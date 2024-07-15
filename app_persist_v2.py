import streamlit as st
import boto3
import json
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_community.chat_models import BedrockChat
from dotenv import load_dotenv
import os
import time
import aiohttp
import asyncio

load_dotenv()  # Load variables from .env file

aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
region_name = 'us-east-1'

# Create Bedrock client for chat model
bedrock_client = boto3.client(
    service_name='bedrock-runtime',
    region_name=region_name,
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key
)

# Initialize Chat model
chat_model = BedrockChat(client=bedrock_client, model_id="anthropic.claude-3-sonnet-20240229-v1:0")

# Ollama embedding configuration
OLLAMA_URL = "http://dpo.asuscomm.com:11434/api/embeddings"
MODEL_NAME = "znbang/bge:large-en-v1.5-f16"
CONCURRENT_REQUESTS = 100

# Function to get embedding from Ollama
async def get_embedding(session, text):
    payload = {
        "model": MODEL_NAME,
        "prompt": text
    }
    async with session.post(OLLAMA_URL, json=payload) as response:
        result = await response.json()
        return result['embedding']

# Function to get embeddings concurrently
async def get_embeddings(texts):
    async with aiohttp.ClientSession() as session:
        tasks = [asyncio.create_task(get_embedding(session, text)) for text in texts]
        return await asyncio.gather(*tasks)

# Function to parse PDF
def parse_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Function to chunk text
def chunk_text(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return text_splitter.split_text(text)

# Streamlit app
st.title("Chat with PDF")

# Initialize session state
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None

# File upload
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    # Only process the PDF if we haven't already
    if st.session_state.vectorstore is None:
        # Parse PDF and chunk text
        start_time = time.time()
        pdf_text = parse_pdf(uploaded_file)
        chunks = chunk_text(pdf_text)
        processing_time = time.time() - start_time
        st.write(f"Time for processing text and chunking: {processing_time:.2f} seconds")
        
        # Create FAISS index with Ollama embeddings
        start_time = time.time()
        embeddings = asyncio.run(get_embeddings(chunks))
        vectorstore = FAISS.from_embeddings(list(zip(chunks, embeddings)), None)
        st.session_state.vectorstore = vectorstore
        embedding_time = time.time() - start_time
        st.write(f"Time for embedding the text doc: {embedding_time:.2f} seconds")
        
        st.success("PDF processed and indexed successfully!")
    else:
        st.success("PDF already processed and indexed.")
    
    # User input for questions
    user_question = st.text_input("Ask a question about the PDF content:")
    
    if user_question:
        # Perform similarity search
        start_time = time.time()
        question_embedding = asyncio.run(get_embeddings([user_question]))[0]
        docs = st.session_state.vectorstore.similarity_search_by_vector(question_embedding, k=3)
        retrieval_time = time.time() - start_time
        st.write(f"Time for retrieving information from vector store: {retrieval_time:.2f} seconds")
        
        # Prepare context for the chat model
        context = "\n".join([doc.page_content for doc in docs])
        
        # Prepare messages for the chat model
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Answer the user's question based on the provided context from the PDF."},
            {"role": "user", "content": f"Context: {context}\n\nQuestion: {user_question}"}
        ]
        
        try:
            # Use the chat_model to generate a response
            start_time = time.time()
            response = chat_model.invoke(messages)
            answer = response.content
            llm_time = time.time() - start_time
            st.write(f"Time for LLM call processing: {llm_time:.2f} seconds")
            
            st.write("Answer:", answer)
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

else:
    st.info("Please upload a PDF file to begin.")