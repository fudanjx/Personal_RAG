import asyncio
import time
import boto3
from dotenv import load_dotenv
import os
from langchain_community.embeddings import BedrockEmbeddings

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

# Initialize Bedrock embeddings
embeddings = BedrockEmbeddings(client=bedrock_client, model_id="cohere.embed-english-v3")

CONCURRENT_REQUESTS = 30

async def get_embedding(text):
    start_time = time.time()
    embedding = embeddings.embed_query(text)
    end_time = time.time()
    return end_time - start_time

async def main():
    total_start_time = time.time()
    
    tasks = []
    for i in range(CONCURRENT_REQUESTS):
        text = f"This is a test sentence for embedding request {i+1}."
        task = asyncio.create_task(get_embedding(text))
        tasks.append(task)
    
    results = await asyncio.gather(*tasks)
    
    total_end_time = time.time()
    total_time = total_end_time - total_start_time
    
    for i, response_time in enumerate(results, 1):
        print(f"Request {i}: {response_time:.4f} seconds")
    
    print(f"\nAverage response time: {sum(results) / len(results):.4f} seconds")
    print(f"Total time for all requests: {total_time:.4f} seconds")

if __name__ == "__main__":
    asyncio.run(main())