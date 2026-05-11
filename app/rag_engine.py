import os
from dotenv import load_dotenv
import psycopg
from flashrank import Ranker, RerankRequest
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_postgres.vectorstores import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# We need a psycopg connection string for PGVector
connection_string_original = os.getenv("DATABASE_URL", "postgresql://postgres:12345@localhost:5432/vectordb")
# Ensure psycopg driver is used
if connection_string_original.startswith("postgresql://"):
    connection_string = connection_string_original.replace("postgresql://", "postgresql+psycopg://")
else:
    connection_string = connection_string_original

# Create vector extension just in case
try:
    with psycopg.connect(connection_string_original) as conn:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cur.execute("""
            CREATE TABLE IF NOT EXISTS user_documents (
                id SERIAL PRIMARY KEY,
                user_id TEXT NOT NULL,
                filename TEXT NOT NULL,
                UNIQUE(user_id, filename)
            );
            """)
            conn.commit()
except Exception as e:
    print(f"Warning: Could not create database schema automatically: {e}")

embeddings = AzureOpenAIEmbeddings(
    azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-small"),
    openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
)

llm = AzureChatOpenAI(
    azure_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4"),
    openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    temperature=0
)

collection_name = "rag_docs"

# Phrases that indicate the LLM could not find an answer in the retrieved context
_UNCERTAINTY_PHRASES = [
    "i don't know",
    "i do not know",
    "not mentioned",
    "not provided",
    "no information",
    "cannot find",
    "can't find",
    "not found in",
    "not in the context",
    "not available in",
    "the context does not",
    "the provided context",
    "based on the provided",
    "no relevant information",
]

def _is_uncertain_answer(answer: str) -> bool:
    """Return True if the LLM indicated it could not answer from the RAG context."""
    lower = answer.lower()
    return any(phrase in lower for phrase in _UNCERTAINTY_PHRASES)

def _clean_snippet(text: str, max_chars: int = 600) -> str:
    """Strip common Wikipedia-style markup noise from a snippet."""
    import re
    # Remove citation markers like ^_a__b_, [1], ^ _a_, etc.
    text = re.sub(r'\^\s*_[a-z_]+_', '', text)
    text = re.sub(r'\[\s*\.\.\.\s*\]', '', text)
    text = re.sub(r'\[\d+\]', '', text)
    # Remove markdown heading markers
    text = re.sub(r'#{1,6}\s*', '', text)
    # Collapse multiple whitespace/newlines
    text = re.sub(r'\s+', ' ', text).strip()
    return text[:max_chars]

def _tavily_search(question: str) -> tuple[str, list[str]]:
    """
    Query Tavily Search and return a clean (answer_text, sources) tuple.

    Strategy:
    1. Use include_answer=True so Tavily attempts to synthesise a direct answer.
    2. If Tavily's answer field is non-empty, use it directly — it's already clean.
    3. If the answer field is empty, clean the raw snippets and pass them through
       the LLM to produce a concise, readable answer instead of dumping raw markup.
    """
    from tavily import TavilyClient
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key or api_key == "YOUR_TAVILY_API_KEY":
        return "Tavily API key is not configured. Please set TAVILY_API_KEY in your .env file.", []

    client = TavilyClient(api_key=api_key)
    response = client.search(
        query=question,
        search_depth="advanced",
        max_results=5,
        include_answer=True,        # ask Tavily to return a direct answer
    )

    results = response.get("results", [])
    sources = [r.get("url", "") for r in results if r.get("url")]

    if not results:
        return "No results found via web search.", []

    # ── Prefer Tavily's own synthesised answer (already clean prose) ──────────
    tavily_answer = (response.get("answer") or "").strip()
    if tavily_answer:
        return tavily_answer, sources

    # ── Fallback: summarise raw snippets via LLM to avoid markup noise ────────
    cleaned_snippets = [
        _clean_snippet(r.get("content", ""))
        for r in results[:4]
        if r.get("content", "").strip()
    ]
    if not cleaned_snippets:
        return "No usable content found in web search results.", sources

    combined_context = "\n\n".join(
        f"[Source {i+1}] {snip}" for i, snip in enumerate(cleaned_snippets)
    )

    summary_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a helpful assistant. Using only the web search excerpts below, "
         "answer the user's question in clear, concise prose (3-5 sentences max). "
         "Do NOT include citation numbers, markdown headings, or references. "
         "If the excerpts do not contain enough information, say so briefly.\n\n"
         "{context}"),
        ("human", "{input}"),
    ])
    summary_chain = summary_prompt | llm
    answer = summary_chain.invoke({"input": question, "context": combined_context}).content.strip()
    return answer, sources

def get_vector_store():
    return PGVector(
        embeddings=embeddings,
        collection_name=collection_name,
        connection=connection_string,
        use_jsonb=True,
    )

def process_and_store_document(file_path: str, filename: str, user_id: str):
    # Load document
    if file_path.lower().endswith(".pdf"):
        loader = PyMuPDFLoader(file_path)
    else:
        loader = TextLoader(file_path, encoding="utf-8")
        
    docs = loader.load()
    
    for doc in docs:
        doc.metadata["source"] = filename
        doc.metadata["user_id"] = user_id
    
    # Split lengths
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=128)
    splits = text_splitter.split_documents(docs)
    
    # Store
    vector_store = get_vector_store()
    vector_store.add_documents(splits)
    
    return len(splits)

def query_rag(question: str, user_id: str, filename: str = None) -> tuple[str, list[str], str]:
    """
    Query the RAG pipeline and fall back to Tavily Search if the answer
    is not found in the user's documents.

    Returns:
        (answer_text, sources, source_type)
        source_type is "rag" or "tavily"
    """
    vector_store = get_vector_store()
    
    filter_dict = {"user_id": user_id}
    if filename:
        filter_dict["source"] = filename
        
    # Retrieve top 20 candidate documents initially
    retriever = vector_store.as_retriever(search_kwargs={"k": 20, "filter": filter_dict})
    initial_docs = retriever.invoke(question)
    
    # --- Fallback: no docs retrieved at all → go straight to Tavily ---
    if not initial_docs:
        print(f"[RAG] No documents retrieved for user '{user_id}'. Falling back to Tavily Search.")
        answer, sources = _tavily_search(question)
        return answer, sources, "tavily"
    
    # Rerank down to top 5 using flashrank
    ranker = Ranker()
    #If you do not specify a model name, Ranker() automatically defaults to using the 
    # ms-marco-TinyBERT-L-2-v2 model.
    #This default is chosen by FlashRank because it is incredibly lightweight (less than 15MB) 
    passages = [
        {"id": i, "text": doc.page_content, "meta": doc.metadata}
        for i, doc in enumerate(initial_docs)
    ]
    rerankrequest = RerankRequest(query=question, passages=passages)
    reranked_results = ranker.rerank(rerankrequest)
    
    final_docs = []
    for res in reranked_results[:5]:
        final_docs.append(Document(page_content=res["text"], metadata=res["meta"]))
    
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer the question. "
        "If the answer is not contained in the context, respond with exactly: "
        "'I don't know based on the provided documents.' "
        "Use three sentences maximum and keep the answer concise."
        "\n\n"
        "{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    
    answer_text = question_answer_chain.invoke({
        "input": question,
        "context": final_docs
    })
    
    # --- Fallback: LLM signals it could not find the answer in context → Tavily ---
    if _is_uncertain_answer(answer_text):
        print(f"[RAG] Uncertain answer detected. Falling back to Tavily Search.")
        tavily_answer, tavily_sources = _tavily_search(question)
        return tavily_answer, tavily_sources, "tavily"
    
    # Extract unique sources
    sources = []
    for doc in final_docs:
        src = doc.metadata.get("source", "Unknown")
        if src not in sources:
            sources.append(src)
            
    return answer_text, sources, "rag"
