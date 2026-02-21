import json
import asyncio
from typing import List, Dict, Any
from openai import AsyncOpenAI, RateLimitError, APIError
from returns.result import Result

def load_json_file(path : str) -> List[Dict] | Dict[str, Any]:

    with open(path, "r") as f:
        return json.load(f)
    
async def make_openai_request(
    client : AsyncOpenAI,
    model : str,
    messages : List[Dict],
    max_attempts : int = 3
):

    for attempt in range(max_attempts):

        try:
            
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0,  # No randomness
                top_p=1,  # Disable sampling
                seed=42,  # Seed
            )

            return Result.from_value(response)
        
        except (RateLimitError, APIError) as e:

            if attempt == max_attempts - 1:
                return Result.from_failure(e)

            wait = 2 ** (0.8 * attempt)
            await asyncio.sleep(wait)