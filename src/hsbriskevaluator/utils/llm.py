from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI
from langchain.globals import set_llm_cache
from langchain_community.cache import SQLiteCache
from langchain_core.language_models import BaseChatModel
from pathlib import Path
from openai import OpenAI, AsyncOpenAI
import os
import instructor
from pydantic import BaseModel
from typing import Type


def get_model(model_name) -> BaseChatModel:
    return ChatOpenAI(
        base_url="https://openrouter.ai/api/v1",
        model=model_name,
        # default_headers={"X-Title": "DataSpeak"},
    )


def get_instructor_client():
    return instructor.from_openai(
        OpenAI(base_url="https://openrouter.ai/api/v1"),
    )


def get_async_instructor_client():
    return instructor.from_openai(
        AsyncOpenAI(base_url="https://openrouter.ai/api/v1"),
    )


async def call_llm_with_client(
    client, model_id: str, messages: list[dict], response_model: Type[BaseModel]
):
    response = await client.chat.completions.create(
        model=model_id,
        messages=messages,
        response_model=response_model,
        extra_body={"provider": {"require_parameters": True}},
    )
    return response
