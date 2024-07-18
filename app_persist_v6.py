import streamlit as st
import boto3
import json
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_aws import ChatBedrock
from langchain.memory import ConversationBufferWindowMemory
from dotenv import load_dotenv
import os
import time
import aiohttp
import asyncio
import requests
from streamlit_callback import StreamlitCallbackHandler

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

# Initialize Chat model with streaming
chat_model = ChatBedrock(
    client=bedrock_client,
    model_id="anthropic.claude-3-sonnet-20240229-v1:0",
    streaming=True,
    callbacks=[StreamlitCallbackHandler()]
)

# BGE embedding configuration
BGE_API_URL = "http://dpo.asuscomm.com:8088/api/embeddings"
BATCH_SIZE = 30   #choose between 30 - 100 for optimal results

# Function to get a single embedding synchronously
def get_single_embedding(text):
    payload = {"inputs": text}
    try:
        response = requests.post(BGE_API_URL, json=payload, timeout=30)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"API request failed with status {response.status_code}")
            return None
    except requests.RequestException as e:
        print(f"Error connecting to BGE API: {str(e)}")
        return None

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

# Function to process multiple PDFs
def process_pdfs(files):
    all_chunks = []
    for file in files:
        pdf_text = parse_pdf(file)
        chunks = chunk_text(pdf_text)
        all_chunks.extend(chunks)
    return all_chunks

# Streamlit app
st.title("Chat with Multiple PDFs")

# Initialize session state
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = set()
if "messages" not in st.session_state:
    st.session_state.messages = []
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferWindowMemory(k=5, return_messages=True)

# File upload
uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)

if uploaded_files:
    new_files = [file for file in uploaded_files if file.name not in st.session_state.processed_files]
    
    if new_files:
        st.write(f"Processing {len(new_files)} new PDF(s)...")
        
        # Process new PDFs
        start_time = time.time()
        new_chunks = process_pdfs(new_files)
        processing_time = time.time() - start_time
        st.write(f"Time for processing text and chunking: {processing_time:.2f} seconds")
        
        # Create or update FAISS index with BGE embeddings
        start_time = time.time()
        try:
            st.write("Creating embeddings...")
            progress_bar = st.progress(0)
            new_embeddings = asyncio.run(get_embeddings_batch(new_chunks, progress_bar))
            
            if st.session_state.vectorstore is None:
                vectorstore = FAISS.from_embeddings(list(zip(new_chunks, new_embeddings)), None)
            else:
                vectorstore = st.session_state.vectorstore
                vectorstore.add_embeddings(list(zip(new_chunks, new_embeddings)))
            
            st.session_state.vectorstore = vectorstore
            st.session_state.processed_files.update([file.name for file in new_files])
            
            embedding_time = time.time() - start_time
            st.write(f"Time for embedding the text doc: {embedding_time:.2f} seconds")
            st.write(f"All {len(new_embeddings)} new embeddings successfully created")
            st.success("PDFs processed and indexed successfully!")
        except Exception as e:
            st.error(f"An error occurred during embedding: {str(e)}")
            st.warning("Please try uploading the PDFs again.")
    else:
        st.success("All uploaded PDFs have already been processed and indexed.")

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Ask a question about the PDF content:"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    if st.session_state.vectorstore:
        # Perform similarity search
        start_time = time.time()
        question_embedding = get_single_embedding(prompt)
        if question_embedding is None:
            st.error("Failed to get embedding for the question. Please try again.")
        else:
            docs = st.session_state.vectorstore.similarity_search_by_vector(question_embedding, k=3)
            retrieval_time = time.time() - start_time
            st.write(f"Time for retrieving information from vector store: {retrieval_time:.2f} seconds")
            
            # Prepare context for the chat model
            context = "\n".join([doc.page_content for doc in docs])
            
            # Get chat history from memory
            chat_history = st.session_state.memory.load_memory_variables({})["history"]
            
            # Prepare messages for the chat model
            messages = [
                {"role": "system", "content": "You are a helpful assistant. Answer the user's question based on the provided context from the PDFs and the conversation history."},
                {"role": "user", "content": f"Context: {context}\n\nConversation history: {chat_history}\n\nQuestion: {prompt}"}
            ]
            
            try:
                # Use the chat_model to generate a response with streaming
                start_time = time.time()
                response = chat_model.invoke(messages)
                llm_time = time.time() - start_time
                st.write(f"Time for LLM call processing: {llm_time:.2f} seconds")
                
                # Add assistant response to chat history
                assistant_response = response.content
                st.session_state.messages.append({"role": "assistant", "content": assistant_response})
                
                # Update memory
                st.session_state.memory.save_context({"input": prompt}, {"output": assistant_response})

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
    else:
        st.info("Please upload PDF files to begin.")