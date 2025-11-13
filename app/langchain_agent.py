"""
간단한 멀티턴 대화 Agent
OpenAI API를 직접 사용하여 tool binding과 메모리 관리를 지원합니다.
"""
import json
import os
from typing import Optional, List, Tuple
from datetime import datetime

from openai import AsyncOpenAI
from app.logger import get_logger
from app.stores.chat_history import ChatHistoryStore

log = get_logger(__name__)
history_store = ChatHistoryStore()


class LangChainAgent:
    """멀티턴 대화 Agent (OpenAI API 직접 사용)"""
    
    def __init__(self, model: str = "gpt-4o", temperature: float = 0.2, max_history: int = 10):
        """
        Args:
            model: OpenAI 모델 이름
            temperature: 생성 온도
            max_history: 멀티턴 윈도우 크기 (기본값: 10)
        """
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
        
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_history = max_history
        self.tools = []
        self.system_prompt = self._load_system_prompt()
    
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
    
    def add_tool(self, tool_obj) -> None:
        """Tool 추가"""
        self.tools.append(tool_obj)
    
    def add_tools(self, tools: List) -> None:
        """여러 Tool 추가"""
        self.tools.extend(tools)
    
    async def get_chat_history(self, chat_id: str) -> List[dict]:
        """SQLite에서 채팅 히스토리 로드"""
        messages = await history_store.get_chat_history(chat_id, limit=self.max_history)
        
        # 백엔드 메시지 형식을 OpenAI 형식으로 변환
        chat_history = []
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")
            
            api_msg = {
                "role": role,
                "content": content,
            }
            
            if role == "tool":
                api_msg["tool_call_id"] = msg.get("tool_call_id", "")
            
            chat_history.append(api_msg)
        
        return chat_history
    
    async def process_message(
        self,
        user_input: str,
        chat_id: str,
        user_id: Optional[str] = None,
        **kwargs
    ) -> Tuple[str, dict]:
        """
        사용자 메시지 처리 및 응답 생성
        
        Args:
            user_input: 사용자 입력
            chat_id: 채팅 세션 ID
            user_id: 사용자 ID (선택사항)
            **kwargs: 추가 인자
        
        Returns:
            (응답 텍스트, 메타데이터) 튜플
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
            
            # OpenAI API 호출
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                stream=False,
            )
            
            output = response.choices[0].message.content or ""
            
            # 응답 저장
            await history_store.save_message(chat_id, "assistant", output)
            
            # 메타데이터
            metadata = {
                "chat_id": chat_id,
                "user_id": user_id,
                "total_messages": len(chat_history) + 2,
                "model": self.model,
            }
            
            return output, metadata
            
        except Exception as e:
            log.exception(f"Failed to process message: {e}")
            raise
    
    async def get_full_context(self, chat_id: str) -> List[dict]:
        """전체 채팅 히스토리 조회"""
        return await history_store.get_chat_history(chat_id, limit=self.max_history)
    
    async def clear_history(self, chat_id: str) -> bool:
        """채팅 히스토리 삭제"""
        return await history_store.clear_chat_history(chat_id)
