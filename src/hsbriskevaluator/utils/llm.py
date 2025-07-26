from dotenv import load_dotenv

from hsbriskevaluator.utils.file import get_cache_dir
from hsbriskevaluator.utils.llm_cache import get_llm_cache_provider

load_dotenv()

from langchain_openai import ChatOpenAI
from langchain.globals import set_llm_cache
from langchain_community.cache import SQLiteCache
from langchain_core.language_models import BaseChatModel
from openai import OpenAI, AsyncOpenAI
import os
import json
import instructor
from pydantic import BaseModel
from typing import Type

cache_dir = get_cache_dir()

import hishel

storage = hishel.FileStorage(base_path=get_cache_dir() / "hishel")

# Initialize LLM cache
llm_cache = get_llm_cache_provider("sqlite")


def _serialize_pydantic_response(response: BaseModel) -> dict:
    """Serialize Pydantic model response for caching."""
    return {
        "model_data": response.model_dump(),
        "model_class": response.__class__.__module__ + "." + response.__class__.__qualname__
    }


def _deserialize_pydantic_response(cached_data: dict, response_model: Type[BaseModel]) -> BaseModel:
    """Deserialize cached data back to Pydantic model."""
    model_data = cached_data.get("model_data", {})
    return response_model(**model_data)


def get_model(model_name) -> BaseChatModel:
    return ChatOpenAI(
        base_url="https://openrouter.ai/api/v1",
        model=model_name,
        # default_headers={"X-Title": "DataSpeak"},
    )


def get_instructor_client():
    return instructor.from_openai(
        OpenAI(
            base_url="https://openrouter.ai/api/v1",
            http_client=hishel.CacheClient(),
        ),
    )


def get_async_instructor_client():
    return instructor.from_openai(
        AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            http_client=hishel.AsyncCacheClient(),
        ),
    )


async def call_llm_with_client(
    client, model_id: str, messages: list[dict], response_model: Type[BaseModel]
):
    # Create cache key from parameters
    cache_params = {
        "model_id": model_id,
        "messages": messages,
        "response_model": response_model.__name__,
        "extra_body": {"provider": {"require_parameters": True}},
    }
    cache_key = llm_cache.hash_params(cache_params)
    
    # Try to get cached response
    cached_response = llm_cache.get(cache_key)
    if cached_response:
        try:
            cached_data = json.loads(cached_response)
            return _deserialize_pydantic_response(cached_data, response_model)
        except Exception as e:
            # If deserialization fails, continue with API call
            pass
    
    # Make API call if not cached
    response = await client.chat.completions.create(
        model=model_id,
        messages=messages,
        response_model=response_model,
        extra_body={"provider": {"require_parameters": True}},
    )
    
    # Cache the response
    try:
        serialized_response = _serialize_pydantic_response(response)
        llm_cache.insert(cache_key, cache_params, serialized_response)
    except Exception as e:
        # Don't fail if caching fails
        pass
    
    return response
