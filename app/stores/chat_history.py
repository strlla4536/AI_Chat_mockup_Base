import sqlite3
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, List
from app.logger import get_logger

log = get_logger(__name__)

# Docker 환경과 로컬 환경 모두 지원
if os.getenv("DOCKER_ENV"):
    # Docker 환경: /app 경로 사용
    DB_PATH = "/data/chat_history.db"
else:
    # 로컬 환경: 프로젝트 루트에 생성
    DB_PATH = str(Path(__file__).parent.parent.parent / "chat_history.db")


class ChatHistoryStore:
    """SQLite 기반의 채팅 히스토리 저장소 - LangChain 멀티턴 대화 지원"""
    
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        # 디렉토리가 없으면 생성
        db_dir = os.path.dirname(self.db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)
        self._init_db()
    
    def _init_db(self):
        """데이터베이스 초기화"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 채팅 세션 테이블
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS chat_sessions (
                    chat_id TEXT PRIMARY KEY,
                    user_id TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    title TEXT
                )
            ''')
            
            # 메시지 테이블
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS chat_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chat_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (chat_id) REFERENCES chat_sessions(chat_id)
                )
            ''')
            
            # 인덱싱
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_chat_id ON chat_messages(chat_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_user_id ON chat_sessions(user_id)')
            
            conn.commit()
            conn.close()
            log.info(f"Chat history database initialized at {self.db_path}")
        except Exception as e:
            log.error(f"Failed to initialize database: {e}")
            raise
    
    async def get_chat_history(self, chat_id: str, limit: int = 10) -> List[dict]:
        """
        특정 채팅 세션의 메시지 히스토리 조회
        최근 limit개 메시지만 반환 (기본값: 10개 - 멀티턴 윈도우)
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 총 메시지 수 조회
            cursor.execute('SELECT COUNT(*) FROM chat_messages WHERE chat_id = ?', (chat_id,))
            total_count = cursor.fetchone()[0]
            
            # 최근 limit개만 가져오기
            offset = max(0, total_count - limit)
            
            cursor.execute('''
                SELECT id, role, content, created_at
                FROM chat_messages
                WHERE chat_id = ?
                ORDER BY id ASC
                LIMIT ? OFFSET ?
            ''', (chat_id, limit, offset))
            
            messages = []
            rows = cursor.fetchall()

            for row in rows:
                # row: id, role, content, created_at
                _id, role, content, created_at = row
                msg = {
                    "id": _id,
                    "role": role,
                    "content": content
                }
                messages.append(msg)
            
            conn.close()
            return messages
        except Exception as e:
            log.error(f"Failed to get chat history: {e}")
            return []
    
    async def save_message(self, chat_id: str, role: str, content: str) -> bool:
        """메시지 저장"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 세션 존재 여부 확인, 없으면 생성
            cursor.execute('SELECT chat_id FROM chat_sessions WHERE chat_id = ?', (chat_id,))
            if not cursor.fetchone():
                cursor.execute('''
                    INSERT INTO chat_sessions (chat_id, created_at, updated_at)
                    VALUES (?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                ''', (chat_id,))
            
            # 메시지 저장
            cursor.execute('''
                INSERT INTO chat_messages (chat_id, role, content)
                VALUES (?, ?, ?)
            ''', (chat_id, role, content))
            
            # 세션 업데이트 시간 갱신
            cursor.execute('''
                UPDATE chat_sessions SET updated_at = CURRENT_TIMESTAMP WHERE chat_id = ?
            ''', (chat_id,))
            
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            log.error(f"Failed to save message: {e}")
            return False
    
    async def save_messages(self, chat_id: str, messages: List[dict]) -> bool:
        """여러 메시지 일괄 저장"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 세션 존재 여부 확인, 없으면 생성
            cursor.execute('SELECT chat_id FROM chat_sessions WHERE chat_id = ?', (chat_id,))
            if not cursor.fetchone():
                cursor.execute('''
                    INSERT INTO chat_sessions (chat_id, created_at, updated_at)
                    VALUES (?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                ''', (chat_id,))
            
            # 메시지 저장
            for msg in messages:
                role = msg.get('role')
                content = msg.get('content', '')
                
                cursor.execute('''
                    INSERT INTO chat_messages (chat_id, role, content)
                    VALUES (?, ?, ?)
                ''', (chat_id, role, content))
            
            # 세션 업데이트 시간 갱신
            cursor.execute('''
                UPDATE chat_sessions SET updated_at = CURRENT_TIMESTAMP WHERE chat_id = ?
            ''', (chat_id,))
            
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            log.error(f"Failed to save messages: {e}")
            return False
    
    async def clear_chat_history(self, chat_id: str) -> bool:
        """채팅 히스토리 삭제"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('DELETE FROM chat_messages WHERE chat_id = ?', (chat_id,))
            cursor.execute('DELETE FROM chat_sessions WHERE chat_id = ?', (chat_id,))
            
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            log.error(f"Failed to clear chat history: {e}")
            return False
    
    async def get_session_count(self, user_id: Optional[str] = None) -> int:
        """세션 수 조회"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            if user_id:
                cursor.execute('SELECT COUNT(*) FROM chat_sessions WHERE user_id = ?', (user_id,))
            else:
                cursor.execute('SELECT COUNT(*) FROM chat_sessions')
            
            count = cursor.fetchone()[0]
            conn.close()
            return count
        except Exception as e:
            log.error(f"Failed to get session count: {e}")
            return 0
