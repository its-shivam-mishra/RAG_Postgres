import os
from dotenv import load_dotenv
import psycopg
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_postgres.vectorstores import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_classic.chains import create_retrieval_chain
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
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    
    # Store
    vector_store = get_vector_store()
    vector_store.add_documents(splits)
    
    return len(splits)

def query_rag(question: str, user_id: str, filename: str = None):
    vector_store = get_vector_store()
    
    filter_dict = {"user_id": user_id}
    if filename:
        filter_dict["source"] = filename
        
    retriever = vector_store.as_retriever(search_kwargs={"k": 5, "filter": filter_dict})
    
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer the question. "
        "If you don't know the answer, say that you don't know. "
        "Use three sentences maximum and keep the answer concise."
        "\n\n"
        "{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    result = rag_chain.invoke({"input": question})
    
    # Extract unique sources
    sources = []
    for doc in result.get("context", []):
        src = doc.metadata.get("source", "Unknown")
        if src not in sources:
            sources.append(src)
            
    return result["answer"], sources
