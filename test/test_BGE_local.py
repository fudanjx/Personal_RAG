import aiohttp
import asyncio
import time
# 
BGE_API_URL = "http://dpo.asuscomm.com:8088/api/embeddings"
# BGE_API_URL = "http://dpo.asuscomm.com:8088/v1/embed"
CONCURRENT_REQUESTS = 1000

async def get_embedding(session, text):
    payload = {
        "inputs": text
    }
    start_time = time.time()
    try:
        async with session.post(BGE_API_URL, json=payload, timeout=30) as response:
            if response.status == 200:
                result = await response.json()
                end_time = time.time()
                return end_time - start_time, len(result)
            else:
                print(f"API request failed with status {response.status}")
                return None
    except asyncio.TimeoutError:
        print("Request to BGE API timed out")
        return None
    except aiohttp.ClientError as e:
        print(f"Error connecting to BGE API: {str(e)}")
        return None

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
    
    successful_requests = [r for r in results if r is not None]
    
    for i, result in enumerate(results, 1):
        if result:
            response_time, embedding_length = result
            print(f"Request {i}: {response_time:.4f} seconds, Embedding length: {embedding_length}")
        else:
            print(f"Request {i}: Failed")
    
    if successful_requests:
        avg_response_time = sum(r[0] for r in successful_requests) / len(successful_requests)
        print(f"\nAverage response time: {avg_response_time:.4f} seconds")
    else:
        print("\nNo successful requests")
    
    print(f"Total time for all requests: {total_time:.4f} seconds")
    print(f"Successful requests: {len(successful_requests)} out of {CONCURRENT_REQUESTS}")

if __name__ == "__main__":
    asyncio.run(main())