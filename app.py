import streamlit as st
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from connect_memory import get_embeddings, set_custom_prompt, CUSTOM_PROMPT_TEMPLATE
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import os
import logging
from pinecone import Pinecone

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API Keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Streamlit Config
st.set_page_config(page_title="Medi-Bot", page_icon="🩺", layout="centered")
st.title("Medi-Bot")
st.write("Ask me anything about health and wellness!")

# Initialize Pinecone & Check if index exists
pc = Pinecone(api_key=PINECONE_API_KEY)
if "medi-bot" not in pc.list_indexes().names():
    logger.info("Pinecone index 'medi-bot' does not exist! Creating it now...")
    pc.create_index(name="medi-bot", dimension=384, metric="cosine")

# Load embeddings & vector store
embeddings = get_embeddings()
docsearch = PineconeVectorStore(index_name="medi-bot", embedding=embeddings)

# Initialize Groq LLM
llm = ChatGroq(api_key=GROQ_API_KEY, model_name="llama-3.1-8b-instant")

# Create QA Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=docsearch.as_retriever(search_kwargs={'k': 3}),
    return_source_documents=True,
    chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
)

# Initialize session state for messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

# Get user input
user_input = st.chat_input("Pass your question here")

if user_input:
    st.chat_message("user").markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})
    try:
        # Get response
        response = qa_chain.invoke(user_input)  # Ensure correct format

        # Log and display response
        logger.info(f"Successfully got response: {response['result']}")
        with st.chat_message('assistant'):
            st.markdown(response['result'])
            st.session_state.messages.append({"role": "assistant", "content": response['result']})

    except Exception as e:
        logger.error(f"Failed to process query: {e}")
        st.error("An error occurred while processing your request.")
