"""
간단한 멀티턴 대화 Agent
LangChain 대신 공통 스트리밍 헬퍼(`call_llm_stream`)를 사용하여 GenOS LLM 서빙과 통신합니다.
툴 실행(함수 호출)은 에이전트 내부에서 수행하지 않고 ToolHandler 같은 외부 컴포넌트로 위임합니다.
"""
from typing import Optional, List, Tuple, Callable, Awaitable
from datetime import datetime

from app.logger import get_logger
from app.stores.chat_history import ChatHistoryStore
from app.utils import _get_default_model

log = get_logger(__name__)
history_store = ChatHistoryStore()


class LangChainAgent:
    """멀티턴 대화 Agent - 스트리밍 유지, 툴 실행 분리"""
    
    def __init__(self, model: Optional[str] = None, temperature: float = 0.2, max_history: int = 10):
        """
        Args:
            model: 사용할 모델명 (미지정 시 환경값 사용)
            temperature: 생성 온도
            max_history: 멀티턴 윈도우 크기 (기본값: 10)
        """
        self.model = model or _get_default_model()
        self.temperature = temperature
        self.max_history = max_history
        self.system_prompt = self._load_system_prompt()
        # ToolHandler is created so external code can register tools via agent.add_tools/add_tool
        self.tool_handler = ToolHandler()
    
    def _load_system_prompt(self) -> str:
        """시스템 프롬프트 로드"""
        from app.utils import ROOT_DIR
        try:
            prompt_file = ROOT_DIR / "prompts" / "system.txt"
            system_prompt = prompt_file.read_text(encoding="utf-8")
            return system_prompt.format(
                current_date=datetime.now().strftime("%Y-%m-%d"),
                locale="ko-KR"
            )
        except Exception as e:
            log.warning(f"Failed to load system prompt: {e}")
            return "You are a helpful AI assistant."
    
    async def get_chat_history(self, chat_id: str) -> List[dict]:
        """SQLite에서 채팅 히스토리 로드"""
        messages = await history_store.get_chat_history(chat_id, limit=self.max_history)
        
        # 백엔드 메시지 형식을 LLM 형식으로 변환
        chat_history = []
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")
            
            api_msg = {
                "role": role,
                "content": content,
            }
            chat_history.append(api_msg)
        
        return chat_history
    
    async def process_message(
        self,
        user_input: str,
        chat_id: str,
        user_id: Optional[str] = None,
        on_token: Optional[Callable[[str], Awaitable[None]]] = None,
        **kwargs
    ) -> Tuple[str, dict]:
        """
        사용자 메시지 처리 및 응답 생성 (스트리밍 + 멀티턴)

        툴 호출이 모델 응답에 포함되더라도 실제 툴 실행은 이 에이전트에서 수행하지 않습니다.
        대신 결과의 메타데이터로 tool_calls 정보를 반환합니다.
        """
        try:
            # 히스토리 로드
            chat_history = await self.get_chat_history(chat_id)
            
            # 사용자 메시지 저장
            await history_store.save_message(chat_id, "user", user_input)
            
            # 메시지 구성
            messages = [
                {"role": "system", "content": self.system_prompt},
                *chat_history,
                {"role": "user", "content": user_input}
            ]

            # Use streaming helper in app.utils to keep streaming behavior
            from app.utils import call_llm_stream

            final_message = None
            streamed_text_parts: List[str] = []

            async for res in call_llm_stream(messages=messages, model=self.model, temperature=self.temperature):
                # streaming yields token events and then a final_message dict
                if isinstance(res, dict) and res.get("event") == "token":
                    token_piece = res.get("data", "")
                    if token_piece:
                        streamed_text_parts.append(token_piece)
                        if on_token is not None:
                            try:
                                await on_token(token_piece)
                            except Exception as callback_error:
                                log.warning("on_token callback failed: %s", callback_error)
                else:
                    final_message = res

            if final_message is None:
                raise RuntimeError("No response from LLM")

            assistant_content = final_message.get("content", "")
            tool_calls = final_message.get("tool_calls") or []

            # Always save the assistant message to history
            await history_store.save_message(chat_id, "assistant", assistant_content)

            metadata = {
                "chat_id": chat_id,
                "user_id": user_id,
                "total_messages": len(chat_history) + 2,
                "model": self.model,
                "streamed_text": "".join(streamed_text_parts),
                "tool_calls": tool_calls,  # callers can decide whether to execute tools
            }

            return assistant_content, metadata
        except Exception as e:
            log.exception(f"Failed to process message: {e}")
            raise
    
    async def get_full_context(self, chat_id: str) -> List[dict]:
        """전체 채팅 히스토리 조회"""
        return await history_store.get_chat_history(chat_id, limit=self.max_history)
    
    async def clear_history(self, chat_id: str) -> bool:
        """채팅 히스토리 삭제"""
        return await history_store.clear_chat_history(chat_id)
    
    # Compatibility helpers so existing code calling agent.add_tools/add_tool still work
    def add_tool(self, tool_obj) -> None:
        # lazy create tool_handler if not present
        if not hasattr(self, "tool_handler") or self.tool_handler is None:
            self.tool_handler = ToolHandler()
        self.tool_handler.add_tool(tool_obj)
    
    def add_tools(self, tools: List) -> None:
        if not hasattr(self, "tool_handler") or self.tool_handler is None:
            self.tool_handler = ToolHandler()
        self.tool_handler.add_tools(tools)
    

# ToolHandler remains as a small helper to register/execute tools externally
class ToolHandler:
    """툴 호출 및 관리 로직 (에이전트 내부에서 직접 실행하지 않음)"""
    def __init__(self):
        self.tools = []

    def add_tool(self, tool_obj) -> None:
        self.tools.append(tool_obj)

    def add_tools(self, tools: List) -> None:
        self.tools.extend(tools)

    async def execute_tool(self, tool_name: str, args: dict) -> str:
        tool = next((t for t in self.tools if getattr(t, "name", None) == tool_name), None)
        if not tool:
            return f"Tool {tool_name} not found."
        try:
            result = tool.run(args)
            if hasattr(result, "__await__"):
                result = await result
            return result
        except Exception as e:
            log.error(f"Failed to execute tool {tool_name}: {e}")
            return f"Error executing tool {tool_name}: {e}"
