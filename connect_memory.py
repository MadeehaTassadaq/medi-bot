import os
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__) 

# Define model and embedding
MODEL_NAME = "deepseek-r1-distill-llama-70b"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Initialize Groq client
def load_llm():
    try:
        llm = ChatGroq(api_key=GROQ_API_KEY, model_name=MODEL_NAME)
        logger.info("Successfully connected to Groq.")
        return llm
    except Exception as e:
        logger.error(f"Failed to connect to Groq: {e}")
        raise

llm = load_llm()

# Define custom prompt template
CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer the user's question.
If you don't know the answer, just say that you don't know. Don't try to make up an answer.
Don't provide anything outside the given context.

Context: {context}
Question: {question}

Start the answer directly. No small talk, please.
"""

def set_custom_prompt(custom_prompt_template):
    return PromptTemplate(
        template=custom_prompt_template,
        input_variables=["context", "question"]
    )

# load the Pinecone vector store
def get_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    return embeddings

embeddings = get_embeddings()

# Define the Pinecone index name
index_name = "medi-bot"

from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone

def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def initialize_pinecone_index(index_name, embeddings):
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    pc = Pinecone(api_key=PINECONE_API_KEY)

    # Ensure index exists
    if index_name not in pc.list_indexes().names():
        print(f"Creating index: {index_name}")
        pc.create_index(name=index_name, dimension=384, metric="cosine")

    # Load vector store properly
    return PineconeVectorStore(index_name=index_name, embedding=embeddings)
docsearch = initialize_pinecone_index("medi-bot", embeddings)
print(type(docsearch)) 
# Create QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=load_llm(),
    chain_type="stuff",
    retriever=docsearch.as_retriever(search_kwargs={'k': 3}),
    return_source_documents=True,
    chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
)

response=qa_chain.invoke({"query": "What is the best way to prevent heart disease?"}) 
print(response['result'])
print(response['source_documents'])

