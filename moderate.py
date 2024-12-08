import asyncio
import os
import time

import pandas as pd
import tiktoken
from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

client1 = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
client2 = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY2"))


tokenizer = tiktoken.get_encoding("cl100k_base")


async def detect(client, idx, text):
    response = await client.moderations.create(
        model="omni-moderation-latest",
        input=text,
    )
    scores = response.to_dict()["results"][0]["category_scores"]
    flagged = response.to_dict()["results"][0]["flagged"]
    return (idx, scores, flagged)


class ClientRateLimiter:
    def __init__(self, client, token_limit_per_minute=10000):
        self.client = client
        self.token_limit_per_minute = token_limit_per_minute
        self.tokens_used = 0
        self.start_time = time.time()

    async def ensure_rate_limit(self, token_count):
        if self.tokens_used + token_count > self.token_limit_per_minute:
            elapsed_time = time.time() - self.start_time
            sleep_time = 60 - elapsed_time if elapsed_time < 60 else 0
            print(f"Rate limit reached for client. Sleeping for {sleep_time:.2f} seconds...")
            await asyncio.sleep(sleep_time)

            self.tokens_used = 0
            self.start_time = time.time()

        self.tokens_used += token_count


async def apply_detect(df):
    limiter1 = ClientRateLimiter(client1, token_limit_per_minute=10000)
    limiter2 = ClientRateLimiter(client2, token_limit_per_minute=10000)

    results = []
    clients = [limiter1, limiter2]
    client_idx = 0

    try:
        for idx, row in df.iterrows():
            text = row["transcription"]

            if pd.isna(text):
                results.append((None, None, None))
                continue

            token_count = len(tokenizer.encode(text))

            current_limiter = clients[client_idx]

            await current_limiter.ensure_rate_limit(token_count)

            res = await detect(current_limiter.client, idx, text)
            results.append(res)

            client_idx = (client_idx + 1) % len(clients)

            print(f"Processed {idx}: {token_count} tokens used by client {client_idx + 1}.")
    except Exception:
        return results

    return results
