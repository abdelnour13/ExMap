import asyncio
import json
from itertools import batched
from typing import List, Dict, Optional, Iterable, Callable, Type, AsyncGenerator, TypedDict
from openai import AsyncOpenAI, RateLimitError, APIError
from openai.types.chat.chat_completion import ChatCompletion
from returns.result import Result
from returns.pipeline import is_successful
from pydantic import BaseModel, ValidationError

class LLMReturn(TypedDict):
    task : Dict
    response : Dict

async def run_llm_tasks(
    client : AsyncOpenAI,
    model : str,
    tasks : Iterable[Dict],
    create_messages : Callable[[Dict], List[Dict]],
    batch_size : int,
    pydantic_model : Optional[Type[BaseModel]] = None,
    max_attempts : int = 3,
    extra_body : Optional[Dict] = None,
) -> AsyncGenerator[List[Result[LLMReturn, RateLimitError | APIError | json.JSONDecodeError | ValidationError]], None, None]:

    for batch in batched(tasks, n=batch_size):

        responses = await asyncio.gather(*[
            make_openai_request(
                client=client,
                model=model,
                messages=create_messages(task),
                max_attempts=max_attempts,
                extra_body=extra_body
            )
            for task in batch
        ])

        results = [
            decode_openai_response(completion=response,pydantic_model=pydantic_model)
                .map(lambda x : {
                    "task" : task,
                    "response" : response
                })
            for task, response in zip(batch, responses)
        ]

        yield results

def decode_openai_response(
    completion : Result[ChatCompletion, RateLimitError | APIError],
    pydantic_model : Optional[Type[BaseModel]] = None
) -> Result[Dict, RateLimitError | APIError | json.JSONDecodeError | ValidationError]:

    if is_successful(completion):

        response = completion.unwrap()
        content = response.choices[0].message.content

        try:
            json_data = json.loads(content)
        except Exception as e:
            return Result.from_failure(e)

        if pydantic_model:

            try:
                pydantic_model(**json_data)
            except Exception as e:
                return Result.from_failure(e)

        return json_data

    return completion
    
async def make_openai_request(
    client : AsyncOpenAI,
    model : str,
    messages : List[Dict],
    max_attempts : int = 3,
    extra_body : Optional[Dict] = None
) -> Result[ChatCompletion, RateLimitError | APIError]:

    for attempt in range(max_attempts):

        try:
            
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0,  # No randomness
                top_p=1,  # Disable sampling
                seed=42,  # Seed
                extra_body=extra_body
            )

            return response
        
        except (RateLimitError, APIError) as e:

            if attempt == max_attempts - 1:
                return Result.from_failure(e)

            wait = 2 ** (0.8 * attempt)
            await asyncio.sleep(wait)