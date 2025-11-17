import json
import os
import aiohttp
import requests

from app.utils import States


def get_tools_description(server_id: str):
    token_response = requests.post(
        "https://genos.mnc.ai:3443/api/admin/auth/login",
        json={
            "user_id": os.getenv("GENOS_ID"),
            "password": os.getenv("GENOS_PW")
        }
    )
    token_response.raise_for_status()
    token = token_response.json()["data"]["access_token"]
    response = requests.get(
        f"https://genos.mnc.ai:3443/api/admin/mcp/server/test/{server_id}/tools",
        headers={
            "Authorization": f"Bearer {token}"
        }
    )
    response.raise_for_status()
    return response.json()['data']


def get_every_mcp_tools_description():
    tool_name_to_server_id = {}
    out = []
    mcp_server_id_list = [endpoint.strip() for endpoint in os.getenv("MCP_SERVER_ID", "").split(",") if endpoint.strip()]
    nested_list = [get_tools_description(endpoint) for endpoint in mcp_server_id_list]
    for server_id, data in zip(mcp_server_id_list, nested_list):
        for tool in data:
            tool_name_to_server_id[tool['name']] = server_id
        out.extend(data)
    # Normalize to OpenAI tools schema
    out = [
        {
            "type": "function",
            "function": {
                "name": tool.get("name"),
                "description": tool.get("description", ""),
                "parameters": tool.get("input_schema") or tool.get("parameters") or {"type": "object", "properties": {}},
            }
        }
        for tool in out
    ]
    return out, tool_name_to_server_id


MCP_TOOLS, MCP_TOOL_NAME_TO_SERVER_ID = get_every_mcp_tools_description()


def get_mcp_tool(tool_name: str):
    if tool_name not in MCP_TOOL_NAME_TO_SERVER_ID:
        raise ValueError(f"Tool {tool_name} not found")
    server_id = MCP_TOOL_NAME_TO_SERVER_ID[tool_name]

    async def call_mcp_tool(states: States, **tool_input):
        async with aiohttp.ClientSession() as session:
            token_response = await session.post(
                "https://genos.mnc.ai:3443/api/admin/auth/login",
                json={
                    "user_id": os.getenv("GENOS_ID"),
                    "password": os.getenv("GENOS_PW")
                }
            )
            token_response.raise_for_status()
            token = (await token_response.json())["data"]["access_token"]
            response = await session.post(
                f"https://genos.mnc.ai:3443/api/admin/mcp/server/test/{server_id}/tools/call",
                headers={
                    "Authorization": f"Bearer {token}"
                },
                json={"tool_name": tool_name, "input_schema": tool_input}
            )
            response.raise_for_status()
            data = (await response.json())['data']
            
            # 차트 생성 처리
            if tool_name == "generate_chart":
                num_charts = len(states.tool_state.id_to_iframe)
                states.tool_state.id_to_iframe[f"{num_charts}†chart"] = data[0]
                if isinstance(tool_input.get('data_json'), str):
                    data_json = json.loads(tool_input['data_json'])
                else:
                    data_json = tool_input['data_json']
                return f"Chart '{data_json['title']}' has been successfully generated. You can display it to the user by using the following ID: `【{num_charts}†chart】`"
            
            # DA 시각화 처리 (MCP Tool ID 143 또는 워크플로우 3371 관련)
            # data가 iframe HTML 문자열인 경우 처리
            if isinstance(data, str) and ('<iframe' in data.lower() or 'iframe' in data.lower()):
                num_visualizations = len(states.tool_state.id_to_iframe)
                visualization_id = f"{num_visualizations}†chart"
                states.tool_state.id_to_iframe[visualization_id] = data
                return f"Data analysis visualization has been successfully generated. You can display it to the user by using the following ID: `【{visualization_id}】`"
            
            # data가 리스트이고 첫 번째 요소가 iframe HTML인 경우
            if isinstance(data, list) and len(data) > 0:
                first_item = data[0]
                if isinstance(first_item, str) and ('<iframe' in first_item.lower() or 'iframe' in first_item.lower()):
                    num_visualizations = len(states.tool_state.id_to_iframe)
                    visualization_id = f"{num_visualizations}†chart"
                    states.tool_state.id_to_iframe[visualization_id] = first_item
                    return f"Data analysis visualization has been successfully generated. You can display it to the user by using the following ID: `【{visualization_id}】`"
            
            # data가 딕셔너리이고 iframe 필드가 있는 경우
            if isinstance(data, dict):
                if 'iframe' in data or 'html' in data or 'visualization' in data:
                    num_visualizations = len(states.tool_state.id_to_iframe)
                    visualization_id = f"{num_visualizations}†chart"
                    iframe_html = data.get('iframe') or data.get('html') or data.get('visualization')
                    if iframe_html:
                        states.tool_state.id_to_iframe[visualization_id] = iframe_html
                        return f"Data analysis visualization has been successfully generated. You can display it to the user by using the following ID: `【{visualization_id}】`"
            
            return data

    return call_mcp_tool
