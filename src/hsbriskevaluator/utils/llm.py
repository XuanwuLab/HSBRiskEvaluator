from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI
from langchain.globals import set_llm_cache
from langchain_community.cache import SQLiteCache
from langchain_core.language_models import BaseChatModel
from pathlib import Path

def get_model(model_name) -> BaseChatModel:
    return ChatOpenAI(
        base_url="https://openrouter.ai/api/v1",
        model=model_name,
        # default_headers={"X-Title": "DataSpeak"},
    )
