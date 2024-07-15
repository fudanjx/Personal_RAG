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

# BGE embedding configuration
BGE_API_URL = "http://dpo.asuscomm.com:8088/api/embeddings"
BATCH_SIZE = 100

# Function to get embedding from BGE
async def get_embedding(session, text):
    payload = {
        "inputs": text
    }
    try:
        async with session.post(BGE_API_URL, json=payload, timeout=30) as response:
            if response.status == 200:
                result = await response.json()
                return result
            else:
                print(f"API request failed with status {response.status}")
                return None
    except asyncio.TimeoutError:
        print("Request to BGE API timed out")
        return None
    except aiohttp.ClientError as e:
        print(f"Error connecting to BGE API: {str(e)}")
        return None

# Function to get embeddings in batches
async def get_embeddings_batch(texts, progress_bar):
    all_embeddings = []
    failed_texts = []
    processed_count = 0

    async with aiohttp.ClientSession() as session:
        while texts or failed_texts:
            batch = (failed_texts + texts)[:BATCH_SIZE]
            texts = texts[len(batch) - len(failed_texts):]
            failed_texts = []

            tasks = [asyncio.create_task(get_embedding(session, text)) for text in batch]
            results = await asyncio.gather(*tasks)

            for text, result in zip(batch, results):
                if result is not None:
                    all_embeddings.append(result)
                else:
                    failed_texts.append(text)

            processed_count += len(batch)
            progress_bar.progress(processed_count / (len(texts) + processed_count))

    if failed_texts:
        raise Exception(f"Failed to get {len(failed_texts)} embeddings after all retries")

    return all_embeddings

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
        
        # Create FAISS index with BGE embeddings
        start_time = time.time()
        try:
            st.write("Creating embeddings...")
            progress_bar = st.progress(0)
            embeddings = asyncio.run(get_embeddings_batch(chunks, progress_bar))
            vectorstore = FAISS.from_embeddings(list(zip(chunks, embeddings)), None)
            st.session_state.vectorstore = vectorstore
            embedding_time = time.time() - start_time
            st.write(f"Time for embedding the text doc: {embedding_time:.2f} seconds")
            st.write(f"All {len(embeddings)} embeddings successfully created")
            st.success("PDF processed and indexed successfully!")
        except Exception as e:
            st.error(f"An error occurred during embedding: {str(e)}")
            st.warning("Please try uploading the PDF again.")
    else:
        st.success("PDF already processed and indexed.")
    
    # User input for questions
    user_question = st.text_input("Ask a question about the PDF content:")
    
    if user_question:
        # Perform similarity search
        start_time = time.time()
        question_embedding = asyncio.run(get_embeddings_batch([user_question], st.progress(0)))[0]
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