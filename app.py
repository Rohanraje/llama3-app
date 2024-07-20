# app.py
from flask import Flask, render_template, request, jsonify
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_parse import LlamaParse
from dotenv import load_dotenv
import os
import nest_asyncio

nest_asyncio.apply()

# Load environment variables
load_dotenv()

# Print the loaded API key to verify
api_key = os.getenv("SECRET_API_KEY")
if not api_key:
    raise ValueError("SECRET_API_KEY is not set in the .env file.")
print(f"Loaded API key: {api_key}")


# Initialize parser
parser = LlamaParse(
    result_type="text",
    api_key=api_key
)

# Extract documents
file_extractor = {".pdf": parser}
documents = SimpleDirectoryReader(
    "./data",
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

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get("message")
    response = chat_engine.chat(user_message)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)
