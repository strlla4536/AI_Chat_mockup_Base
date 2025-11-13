import asyncio
import json
import re
from datetime import datetime
from uuid import uuid4

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict

from app.utils import (
    call_llm_stream, 
    is_sse, 
    ROOT_DIR, 
    States
)
from app.stores.session_store import SessionStore
from app.stores.chat_history import ChatHistoryStore
from app.tools import get_tool_map, get_tools_for_llm
from app.logger import get_logger

router = APIRouter()
store = SessionStore()
history_store = ChatHistoryStore()
log = get_logger(__name__)


class GenerateRequest(BaseModel):
    question: str
    chatId: str | None = None
    userInfo: dict | None = None
    model_config = ConfigDict(extra='allow')


@router.post("/chat/stream")
async def chat_stream(
    req: GenerateRequest, 
    request: Request
) -> StreamingResponse:
    """
    SSE 프로토콜을 사용하여 채팅 스트리밍을 제공합니다.
    """
    queue: asyncio.Queue[str] = asyncio.Queue()
    SENTINEL = "__STREAM_DONE__"
    client_disconnected = asyncio.Event()

    async def emit(event: str, data):
        payload = {"event": event, "data": data}
        await queue.put(f"data: {json.dumps(payload, ensure_ascii=False)}\n\n")

    async def heartbeat():
        while True:
            if client_disconnected.is_set():
                break
            await asyncio.sleep(10)
            await queue.put(": keep-alive\n\n")

    async def runner():
        # 변수 초기화 (예외 발생 시에도 finally에서 사용할 수 있도록)
        states = None
        chat_id = None
        history = []
        
        try:
            states = States()
            chat_id = req.chatId or uuid4().hex
            log.info("chat stream started", extra={"chat_id": chat_id})

            if req.userInfo:
                states.user_id = req.userInfo.get("id")

            system_prompt = (ROOT_DIR / "prompts" / "system.txt").read_text(encoding="utf-8").format(
                current_date=datetime.now().strftime("%Y-%m-%d"),
                locale="ko-KR"
            )
            
            # model_set_context 초기화 (user_id가 없어도 사용할 수 있도록)
            model_set_context = []
            if states.user_id:
                model_set_context_list = await store.get_messages(states.user_id)
                if model_set_context_list:
                    model_set_context = [{
                            "role": "system",
                            "content": "### User Memory\n" + "\n".join([f"{idx}. {msc}" for idx, msc in enumerate(model_set_context_list,   start=1)])
                        }]
            
            persisted = (await store.get_messages(chat_id)) or []
            history = [
                *persisted,
                {"role": "user", "content": req.question}
            ]
            
            states.messages = [
                {"role": "system", "content": system_prompt},
                *model_set_context,
                *history
            ]
            states.tools = await get_tools_for_llm()
            tool_map = await get_tool_map()

            while True:
                if client_disconnected.is_set():
                    break
                
                await emit("tool_state", states.tool_state.model_dump())
                
                # 최종 메시지를 저장할 변수
                final_message = None
                
                # 스트림 처리
                async for res in call_llm_stream(
                    messages=states.messages,
                    tools=states.tools,
                    temperature=0.2
                ):
                    if is_sse(res):
                        # SSE 이벤트 (토큰 등)는 즉시 emit
                        await emit(res["event"], res["data"])
                    else:
                        # 최종 메시지는 나중에 처리하기 위해 저장
                        final_message = res
                
                # 최종 메시지가 없으면 루프 종료
                if final_message is None:
                    break
                
                # 최종 메시지를 states.messages에 추가
                states.messages.append(final_message)
                
                # tool_calls와 content 확인
                tool_calls = final_message.get("tool_calls") or []
                contents = final_message.get("content", "")
                
                # 툴 호출이 없고 콘텐츠가 있으면 종료
                if not tool_calls and contents:
                    break
                # 툴 호출이 없고 콘텐츠가 없으면 다시 인퍼런스 시도
                elif not tool_calls and not contents:
                    continue
                
                # 툴 호출이 있으면 툴 호출 처리
                if tool_calls:
                    for tool_call in tool_calls:
                        tool_name = tool_call.get('function', {}).get('name')
                        if not tool_name:
                            log.warning("tool call에 name이 없음", extra={"chat_id": chat_id, "tool_call": tool_call})
                            continue
                        
                        try:
                            tool_args_str = tool_call.get('function', {}).get('arguments', '{}')
                            tool_args = json.loads(tool_args_str) if tool_args_str else {}
                        except json.JSONDecodeError as e:
                            log.exception("tool arguments JSON 파싱 실패", extra={"chat_id": chat_id, "tool_name": tool_name, "arguments": tool_args_str})
                            tool_args = {}
                        
                        log.info("tool call", extra={"chat_id": chat_id, "tool_name": tool_name})
                        
                        try:
                            tool_res = tool_map[tool_name](states, **tool_args)
                            if tool_name == "search":
                                await emit("agentFlowExecutedData", {
                                    "nodeLabel": "Visible Query Generator",
                                    "data": {
                                        "output": {
                                            "content": json.dumps({
                                                "visible_web_search_query": [sq.get('q', '') for sq in tool_args.get('search_query', [])]
                                            }, ensure_ascii=False)
                                        }
                                    }
                                })
                            elif tool_name == "open":
                                try:
                                    if tool_args.get('id') and tool_args['id'].startswith('http'):
                                        url = tool_args['id']
                                    elif tool_args.get('id') is None:
                                        url = getattr(states.tool_state, "current_url", None)
                                    else:
                                        url = states.tool_state.id_to_url.get(tool_args['id'])
                                    if url:
                                        await emit("agentFlowExecutedData", {
                                            "nodeLabel": "Visible URL",
                                            "data": {
                                                "output": {
                                                    "content": json.dumps({
                                                        "visible_url": url
                                                    }, ensure_ascii=False)
                                                }
                                            }
                                        })
                                except Exception as e:
                                    log.exception("open tool emit 실패", extra={"chat_id": chat_id})

                            if asyncio.iscoroutine(tool_res):
                                tool_res = await tool_res
                        except Exception as e:
                            log.exception("tool call failed", extra={"chat_id": chat_id, "tool_name": tool_name})
                            tool_res = f"Error calling {tool_name}: {e}\n\nTry again with different arguments."
                        
                        tool_call_id = tool_call.get('id', '')
                        states.messages.append({"role": "tool", "content": str(tool_res), "tool_call_id": tool_call_id})

        except Exception as e:
            log.exception("chat stream failed")
            await emit("error", str(e))
            await emit("token", f"\n\n오류가 발생했습니다: {e}")
        finally:
            # states와 history가 정의되어 있고 유효한 경우에만 메시지 저장
            try:
                if states and hasattr(states, 'messages') and states.messages and chat_id:
                    last_message = states.messages[-1]
                    
                    if isinstance(last_message, dict) and last_message.get("role") == "assistant":
                        content = last_message.get("content", "")
                        if isinstance(content, str):
                            content = re.sub(r"【[^】]*】", "", content).strip()
                            last_message = {**last_message, "content": content}
                    
                    # history에 마지막 메시지 추가하고 저장
                    if history:
                        history.append(last_message)
                        await store.save_messages(chat_id, history)
            except Exception as e:
                log.exception("failed to save messages in finally block", extra={"chat_id": chat_id})
            
            await emit("result", None)
            await queue.put(SENTINEL)
            log.info("chat stream finished", extra={"chat_id": chat_id})

    async def sse():
        producer = asyncio.create_task(runner())
        pinger = asyncio.create_task(heartbeat())
        try:
            while True:
                if await request.is_disconnected():
                    client_disconnected.set()
                    break
                chunk = await queue.get()
                if chunk == SENTINEL:
                    break
                yield chunk
        finally:
            client_disconnected.set()
            producer.cancel()
            pinger.cancel()

    return StreamingResponse(
        sse(), 
        media_type="text/event-stream", 
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"}
    )


# ============ LangChain 멀티턴 엔드포인트 ============

@router.post("/chat/multiturn")
async def chat_multiturn(
    req: GenerateRequest,
    request: Request
) -> StreamingResponse:
    """
    LangChain Agent를 이용한 멀티턴 대화
    최대 10개 메시지 윈도우를 유지합니다.
    """
    queue: asyncio.Queue[str] = asyncio.Queue()
    SENTINEL = "__STREAM_DONE__"
    client_disconnected = asyncio.Event()

    async def emit(event: str, data):
        payload = {"event": event, "data": data}
        await queue.put(f"data: {json.dumps(payload, ensure_ascii=False)}\n\n")

    async def heartbeat():
        while True:
            if client_disconnected.is_set():
                break
            await asyncio.sleep(10)
            await queue.put(": keep-alive\n\n")

    async def runner():
        chat_id = None
        user_id = None
        
        try:
            # LangChain Agent 초기화
            from app.langchain_agent import LangChainAgent
            from app.langchain_tools import get_langchain_tools
            
            chat_id = req.chatId or uuid4().hex
            if req.userInfo:
                user_id = req.userInfo.get("id")
            
            log.info("multiturn chat started", extra={"chat_id": chat_id, "user_id": user_id})
            
            # Agent 생성
            agent = LangChainAgent(model="gpt-4o", temperature=0.2, max_history=10)
            agent.add_tools(get_langchain_tools())
            
            # 사용자 입력 첫 토큰
            await emit("token", "")
            
            try:
                # Agent로 메시지 처리
                response, metadata = await agent.process_message(
                    user_input=req.question,
                    chat_id=chat_id,
                    user_id=user_id
                )
                
                # 응답 토큰 스트리밍 (한 글자씩)
                for char in response:
                    await emit("token", char)
                
                # 메타데이터 전달
                await emit("metadata", metadata)
                
                log.info("multiturn chat completed", extra={
                    "chat_id": chat_id,
                    "total_messages": metadata.get("total_messages", 0),
                })
                
            except Exception as e:
                log.exception(f"Agent processing failed for chat {chat_id}")
                error_msg = f"대화 처리 중 오류: {str(e)}"
                await emit("token", error_msg)
                await emit("error", str(e))
        
        except Exception as e:
            log.exception("multiturn chat failed")
            await emit("error", str(e))
            await emit("token", f"\n\n오류가 발생했습니다: {e}")
        
        finally:
            await emit("result", None)
            await queue.put(SENTINEL)
            log.info("multiturn chat finished", extra={"chat_id": chat_id})

    async def sse():
        producer = asyncio.create_task(runner())
        pinger = asyncio.create_task(heartbeat())
        try:
            while True:
                if await request.is_disconnected():
                    client_disconnected.set()
                    break
                chunk = await queue.get()
                if chunk == SENTINEL:
                    break
                yield chunk
        finally:
            client_disconnected.set()
            producer.cancel()
            pinger.cancel()

    return StreamingResponse(
        sse(), 
        media_type="text/event-stream", 
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"}
    )


@router.get("/chat/history/{chat_id}")
async def get_chat_history(chat_id: str):
    """
    특정 채팅 세션의 히스토리 조회 (최대 10개 메시지)
    """
    try:
        messages = await history_store.get_chat_history(chat_id, limit=10)
        return {
            "chat_id": chat_id,
            "messages": messages,
            "total": len(messages)
        }
    except Exception as e:
        log.exception(f"Failed to get chat history: {e}")
        return {"error": str(e), "chat_id": chat_id, "messages": []}


@router.delete("/chat/history/{chat_id}")
async def clear_chat_history(chat_id: str):
    """
    특정 채팅 세션의 히스토리 삭제
    """
    try:
        success = await history_store.clear_chat_history(chat_id)
        return {
            "chat_id": chat_id,
            "success": success
        }
    except Exception as e:
        log.exception(f"Failed to clear chat history: {e}")
        return {"error": str(e), "chat_id": chat_id, "success": False}
