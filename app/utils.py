import os
import pathlib
import json
from typing import Any
from pydantic import BaseModel, Field
from openai import AsyncOpenAI

from app.logger import get_logger

log = get_logger(__name__)

ROOT_DIR = pathlib.Path(__file__).parent.absolute()


class ToolState(BaseModel):
    id_to_url: dict[str, str] = Field(default_factory=dict)
    url_to_page: dict[str, object] = Field(default_factory=dict)
    current_url: str | None = None
    tool_results: dict[str, object] = Field(default_factory=dict)
    id_to_iframe: dict[str, str] = Field(default_factory=dict)


class States:
    user_id: str = None
    messages: list[dict]
    turn: int = 0
    tools: list[dict] = []
    tool_state: ToolState = ToolState()
    tool_results: dict[str, object] = {}


def _get_openai_client() -> AsyncOpenAI:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
    return AsyncOpenAI(api_key=api_key)


def _get_default_model() -> str:
    return os.getenv("OPENAI_MODEL", "gpt-4o")


async def call_llm_stream(
    messages: list[dict],
    model: str | None = None,
    tools: list[dict] | None = None,
    temperature: float | None = None,
    **kwargs
):
    """
    OpenAI API를 사용하여 스트리밍 호출합니다.
    OpenAI Chat Completions 스트림을 파싱하여 토큰/툴콜을 동일 포맷으로 내보냅니다.
    """
    client = _get_openai_client()
    model = model or _get_default_model()
    
    # OpenAI API 형식에 맞게 메시지 준비
    api_messages = []
    for msg in messages:
        if isinstance(msg, dict):
            role = msg.get("role")
            if role == "tool":
                # tool 메시지는 tool_call_id가 필요
                api_msg = {
                    "role": "tool",
                    "content": msg.get("content", ""),
                    "tool_call_id": msg.get("tool_call_id", ""),
                }
            else:
                api_msg = {
                    "role": role,
                    "content": msg.get("content", ""),
                }
            api_messages.append(api_msg)
    
    # OpenAI API 호출 파라미터
    stream_params: dict[str, Any] = {
        "model": model,
        "messages": api_messages,
        "stream": True,
    }
    
    if tools:
        stream_params["tools"] = tools
        stream_params["tool_choice"] = "auto"
    
    if temperature is not None:
        stream_params["temperature"] = temperature

    full_content_parts: list[str] = []
    tool_call_buf: dict[int, dict] = {}
    
    try:
        stream = await client.chat.completions.create(**stream_params)
        
        async for chunk in stream:
            if not chunk.choices:
                continue
                
            choice = chunk.choices[0]
            delta = choice.delta
            
            if not delta:
                continue
            
            # tool calls 처리
            if delta.tool_calls:
                for tool_call in delta.tool_calls:
                    if tool_call.index is not None:
                        idx = tool_call.index
                        if idx not in tool_call_buf:
                            tool_call_buf[idx] = {
                                "id": tool_call.id or "",
                                "type": "function",
                                "function": {
                                    "name": "",
                                    "arguments": "",
                                },
                            }
                        buf = tool_call_buf[idx]
                        
                        if tool_call.id:
                            buf["id"] = tool_call.id
                        
                        if tool_call.function:
                            if tool_call.function.name:
                                buf["function"]["name"] = tool_call.function.name
                            if tool_call.function.arguments:
                                buf["function"]["arguments"] += tool_call.function.arguments
            
            # content tokens 처리
            # tool_calls가 있는 경우에도 content가 올 수 있음 (예: o1 모델)
            if delta.content:
                content_piece = delta.content
                if content_piece:
                    full_content_parts.append(content_piece)
                    # tool_calls가 있으면 토큰을 yield하지 않고 버퍼에만 저장
                    # tool_calls가 없으면 토큰을 즉시 yield
                    if not delta.tool_calls:
                        yield {
                            "event": "token",
                            "data": content_piece,
                        }
        
        # 최종 메시지 생성
        final_message: dict[str, Any] = {"role": "assistant"}
        final_content = "".join(full_content_parts).strip()
        final_message["content"] = final_content if final_content else ""
        
        # tool_calls가 있으면 추가
        if tool_call_buf:
            tool_calls = []
            for idx in sorted(tool_call_buf.keys()):
                tc = tool_call_buf[idx]
                # arguments가 JSON 문자열인지 확인
                try:
                    # 이미 JSON 문자열이면 그대로 사용
                    json.loads(tc["function"]["arguments"])
                    args_str = tc["function"]["arguments"]
                except (json.JSONDecodeError, TypeError):
                    # JSON이 아니면 빈 객체로 처리
                    args_str = "{}"
                
                tool_calls.append({
                    "id": tc["id"],
                    "type": tc["type"],
                    "function": {
                        "name": tc["function"]["name"],
                        "arguments": args_str,
                    },
                })
            final_message["tool_calls"] = tool_calls
        
        yield final_message
        
    except Exception as e:
        log.exception("OpenAI API 호출 실패")
        raise


def is_sse(response):
    class SSE(BaseModel):
        event: str
        data: Any

    try:
        SSE.model_validate(response)
        return True
    except Exception:
        return False


