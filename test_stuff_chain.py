import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_core.documents import Document
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

llm = AzureChatOpenAI(
    azure_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4"),
    openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    temperature=0
)

system_prompt = (
    "You are an assistant. Answer the question using context.\n\n{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

qa_chain = create_stuff_documents_chain(llm, prompt)

docs = [
    Document(page_content="Paris is the capital of France.", metadata={"source": "doc1.txt"}),
    Document(page_content="The Eiffel Tower is located in Paris.", metadata={"source": "doc2.txt"})
]

res = qa_chain.invoke({"input": "What is the capital of France?", "context": docs})

print("Type of res:", type(res))
print("Content of res:", res)
