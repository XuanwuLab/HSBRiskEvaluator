import logging
from hsbriskevaluator.utils.llm import get_async_instructor_client
from hsbriskevaluator.utils.prompt import GET_DEBIAN_UPSTREAM_PROMPT, GET_DEBIAN_UPSTREAM_MODEL_ID
from typing import Optional
from dotenv import load_dotenv
load_dotenv()
from pydantic import BaseModel, ValidationError
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
import aiohttp
logger = logging.getLogger(__name__)

llm_client = get_async_instructor_client()
# Pydantic response model
class UpstreamInfo(BaseModel):
    package_name: str
    debian_downstream_git_url: str
    upstream_git_url: str
    upstream_type: str
    parent_debian_package: Optional[str] = None

async def get_upstream_info_by_llm(package_name: str) -> Optional[UpstreamInfo]:
    """Fetch upstream information for a package using the LLM client."""
    try:
        response = await llm_client.chat.completions.create(
            model=GET_DEBIAN_UPSTREAM_MODEL_ID,
            messages=[
                {"role": "system", "content": GET_DEBIAN_UPSTREAM_PROMPT},
                {"role": "user", "content": f"Package name: {package_name}"},
            ],
            response_model=UpstreamInfo,
            extra_body={"provider": {"require_parameters": True}},
        )
        async with aiohttp.ClientSession() as session:
            # Validate the upstream URL
            if not response.upstream_git_url.startswith("http"):
                logger.error(f"Invalid upstream URL format for {package_name}: {response.upstream_git_url}")
                return None
            
            # Check if the URL is reachable
            resp = await session.head(response.upstream_git_url)
            if resp.status == 200:
                return response
            else:
                logger.error(f"Invalid upstream URL for {package_name}: {response.upstream_git_url}")
                return None

    except ValidationError as e:
        logger.error(f"Validation error while fetching upstream info for {package_name}: {e}")
    except Exception as e:
        logger.error(f"Unexpected error while fetching upstream info for {package_name}: {e}")
    return None

async def get_upstream_info_by_agent(package_name: str) -> Optional[UpstreamInfo]:
    """Asynchronous wrapper for fetching upstream information."""
    pass