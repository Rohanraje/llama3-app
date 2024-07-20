import os
import streamlit as st
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_parse import LlamaParse
from dotenv import load_dotenv
import nest_asyncio

nest_asyncio.apply()

# Load environment variables from .env file
load_dotenv()

# Initialize parser
api_key = os.getenv("SECRET_API_KEY")
if not api_key:
    st.error("SECRET_API_KEY is not set in the .env file.")
    st.stop()

parser = LlamaParse(
    result_type="text",
    api_key=api_key
)

# Extract documents
file_extractor = {".pdf": parser}
data_directory = "./data"
if not os.path.exists(data_directory):
    st.error(f"Data directory {data_directory} does not exist.")
    st.stop()
if not os.listdir(data_directory):
    st.error(f"No files found in {data_directory}.")
    st.stop()

documents = SimpleDirectoryReader(
    data_directory,
    recursive=True,
    file_extractor=file_extractor,
).load_data()

# Configure settings
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)
Settings.llm = Ollama(model="llama3", request_timeout=360.0)

# Create index and memory buffer
index = VectorStoreIndex.from_documents(documents)
memory = ChatMemoryBuffer.from_defaults(token_limit=1500)

# Create chat engine
chat_engine = index.as_chat_engine(
    chat_mode="context",
    memory=memory,
    system_prompt=(
        "You are a chatbot, able to have normal interactions. "
        "You know a lot about cars and are willing to help people with "
        "diagnostic checks before they bring their car to a mechanic."
    ),
)

# Streamlit UI
st.title("Chatbot Interface")
st.write("You can ask me anything about cars!")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def update_chat_history(user_message, bot_response):
    st.session_state.chat_history.append(f"You: {user_message}")
    st.session_state.chat_history.append(f"Bot: {bot_response}")

user_input = st.text_input("You:", key="user_input")
if st.button("Send"):
    if user_input:
        try:
            response = chat_engine.chat(user_input)
            update_chat_history(user_input, response)
        except Exception as e:
            st.error(f"Error: {str(e)}")
    else:
        st.error("Please enter a message.")

for chat in st.session_state.chat_history:
    st.write(chat)
