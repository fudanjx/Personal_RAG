import aiohttp
import asyncio
import time
import json

OLLAMA_URL = "http://dpo.asuscomm.com:8088/api/embeddings"
# MODEL_NAME = "znbang/bge:large-en-v1.5-f16"
MODEL_NAME = "BAAI/bge-large-en-v1.5"
CONCURRENT_REQUESTS = 10

async def get_embedding(session, text):
    payload = {
        "model": MODEL_NAME,
        "prompt": text
    }
    start_time = time.time()
    async with session.post(OLLAMA_URL, json=payload) as response:
        result = await response.json()
        end_time = time.time()
    return end_time - start_time

async def main():
    total_start_time = time.time()
    
    async with aiohttp.ClientSession() as session:
        tasks = []
        for i in range(CONCURRENT_REQUESTS):
            text = f"This is a test sentence for embedding request {i+1}."
            task = asyncio.create_task(get_embedding(session, text))
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