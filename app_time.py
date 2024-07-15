import streamlit as st
import boto3
import json
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.chat_models import BedrockChat
from dotenv import load_dotenv
import os
import time

load_dotenv()  # Load variables from .env file

aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
region_name = 'us-east-1'

# Create Bedrock client
bedrock_client = boto3.client(
    service_name='bedrock-runtime',
    region_name=region_name,
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key
)

# Initialize Bedrock embeddings and Chat model
embeddings = BedrockEmbeddings(client=bedrock_client, model_id="cohere.embed-english-v3")
chat_model = BedrockChat(client=bedrock_client, model_id="anthropic.claude-3-sonnet-20240229-v1:0")

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

# File upload
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    # Parse PDF and chunk text
    start_time = time.time()
    pdf_text = parse_pdf(uploaded_file)
    chunks = chunk_text(pdf_text)
    processing_time = time.time() - start_time
    st.write(f"Time for processing text and chunking: {processing_time:.2f} seconds")
    
    # Create FAISS index
    start_time = time.time()
    vectorstore = FAISS.from_texts(chunks, embeddings)
    embedding_time = time.time() - start_time
    st.write(f"Time for embedding the text doc: {embedding_time:.2f} seconds")
    
    st.success("PDF processed and indexed successfully!")
    
    # User input for questions
    user_question = st.text_input("Ask a question about the PDF content:")
    
    if user_question:
        # Perform similarity search
        start_time = time.time()
        docs = vectorstore.similarity_search(user_question, k=3)
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