import aiohttp
import asyncio
import os
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
from typing import Literal

from app.utils import States
from app.logger import get_logger

log = get_logger(__name__)


class SingleSearchModel(BaseModel):
    q: str = Field(description="search string (use the language that's most likely to match the sources)")
    recency: int | None = Field(description="limit to recent N days, or null", default=None)
    domains: list[str] | None = Field(description='restrict to domains (e.g. ["example.com", "another.com"], or null)', default=None)


class MultipleSearchModel(BaseModel):
    search_query: list[SingleSearchModel] = Field(description="array of search query objects. You can call this tool with multiple search queries to get more results faster.")
    response_length: Literal["short", "medium", "long"] = Field(description="response length option", default="medium")


WEB_SEARCH = {
    "type": "function",
    "function": {
        "name": "search",
        "description": "Search the web for information.",
        "parameters": {
            "type": "object",
            "properties": {
                "search_query": {
                    "type": "array",
                    "items": SingleSearchModel.model_json_schema(),
                    "description": "array of search query objects. You can call this tool with multiple search queries to get more results faster."
                },
                "response_length": {
                    "type": "string",
                    "enum": ["short", "medium", "long"],
                    "default": "medium",
                    "description": "response length option"
                }
            },
            "required": ["search_query"]
        }
    }
}


async def web_search(
    states: States,
    **tool_input
) -> list:
    
    try:
        tool_input = MultipleSearchModel(**tool_input)
    except Exception as e:
        return f"Error validating `web_search`: {e}"

    log.info("web_search called", extra={"query": [sq.q for sq in tool_input.search_query] if hasattr(tool_input, 'search_query') else None})
    async with aiohttp.ClientSession() as session:
        tasks = [
            single_search(
                session, 
                sq.q, 
                sq.recency, 
                sq.domains, 
                tool_input.response_length
            ) for sq in tool_input.search_query
        ]
        results = await asyncio.gather(*tasks)
    
    flatted_res = [item for sublist in results for item in sublist]

    outputs = []
    for idx, item in enumerate(flatted_res):
        id = f'{states.turn}:{idx}'
        states.tool_state.id_to_url[id] = item['url']
        outputs.append({'id': id, **item})

    states.turn += 1

    # Log structured results
    try:
        log.info("web_search results", extra={"results": outputs})
    except Exception:
        log.info("web_search results (non-serializable)")

    # Return structured list of results for callers to handle
    return outputs


async def single_search(
    session: aiohttp.ClientSession, 
    q: str, 
    recency: str | None, 
    domains: list[str] | None, 
    response_length: Literal["short", "medium", "long"]
):
    # Tavily API only
    tavily_api_key = os.getenv("TAVILY_API_KEY")
    url = "https://api.tavily.com/search"

    size_map = {"short": 3, "medium": 5, "long": 7}
    num = size_map[response_length]

    payload = {
        "query": q,
        "api_key": tavily_api_key,
        "max_results": num,
        "include_domains": domains if domains else None,
        "recency_days": recency if recency else None
    }

    # Remove None values
    payload = {k: v for k, v in payload.items() if v is not None}

    async with session.post(url, json=payload) as resp:
        resp.raise_for_status()
        data = await resp.json()
        results = data.get("results", [])
        return [
            {
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "snippet": item.get("snippet", ""),
                "source": item.get("source", "tavily"),
                "date": item.get("date", None)
            } for item in results
        ]
