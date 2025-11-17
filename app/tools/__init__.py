from .bio import bio, BIO
from .web_search import web_search, WEB_SEARCH
from .open_url import open, OPEN_URL
from .db import execute_sql, describe_schema, DB_EXECUTE_SQL, DB_DESCRIBE_SCHEMA
try:
    from app.mcp import MCP_TOOLS, get_mcp_tool_map
except Exception:
    MCP_TOOLS = []
    async def get_mcp_tool_map():
        return {}


async def get_tool_map():
    mcp_map = await get_mcp_tool_map()
    return {
        # "search": web_search,
        "open": open,
        "bio": bio,
        "db_execute": execute_sql,
        "db_schema": describe_schema,
        "db_describe_schema": describe_schema,
        **mcp_map,
    }


async def get_tools_for_llm():
    return [
        # WEB_SEARCH,
        OPEN_URL,
        BIO,
        DB_EXECUTE_SQL,
        DB_DESCRIBE_SCHEMA,
        *MCP_TOOLS
    ]
