import aiohttp
import asyncio
import time
import json

BGE_API_URL = "http://dpo.asuscomm.com:8088/api/embeddings"

# Function to get embedding from BGE with retry
async def get_embedding_with_retry(session, text, max_retries=3):
    for attempt in range(max_retries):
        try:
            embedding = await get_embedding(session, text)
            if embedding is not None:
                return embedding
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {str(e)}")
        await asyncio.sleep(1)  # Wait for 1 second before retrying
    print(f"Failed to get embedding after {max_retries} attempts")
    return None

# Function to get embeddings concurrently with retry
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

async def main():
    short_sentence = "The quick brown fox jumps over the lazy dog."
    
    async with aiohttp.ClientSession() as session:
        embedding = await get_embedding(session, short_sentence)
    
    if embedding:
        print(f"Sentence: {short_sentence}")
        print(f"Embedding length: {len(embedding)}")
        print("Embedding (first 5 elements):")
        print(json.dumps(embedding[:5], indent=2))
    else:
        print("Failed to get embedding")

if __name__ == "__main__":
    asyncio.run(main())