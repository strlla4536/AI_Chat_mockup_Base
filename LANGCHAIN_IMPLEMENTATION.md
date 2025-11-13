# LangChain 멀티턴 대화 구현 가이드

## 📋 구현 요약

LangChain을 이용해 멀티턴 대화와 Tool 바인딩을 구현했습니다. SQLite 데이터베이스를 사용하여 대화 히스토리를 영구 저장하며, 최대 10개 메시지의 윈도우를 유지합니다.

## 🎯 핵심 기능

### 1. 멀티턴 대화 (Multi-turn Conversation)
- **최대 10개 메시지 윈도우**: 대화 맥락 유지로 자연스러운 다중 턴 대화 지원
- **SQLite 영구 저장**: 데이터베이스에 모든 대화 히스토리 저장
- **세션 관리**: chatId를 통한 독립적인 대화 세션 관리

### 2. Tool Binding (자동 도구 호출)
- **LangChain Agent**: OpenAI의 tool_calling 기능과 LangChain 통합
- **자동 도구 실행**: Agent가 필요한 도구를 자동으로 선택하고 실행
- **지원하는 도구**:
  - `search_web`: 웹 검색
  - `open_url`: URL 열기 및 내용 추출
  - `manage_memory`: 사용자 메모리 관리

### 3. 데이터베이스 (SQLite)
- **별도 설치 불필요**: Python 내장 sqlite3 사용
- **자동 초기화**: 애플리케이션 시작 시 테이블 자동 생성
- **영구 저장**: Docker 볼륨을 통해 데이터 지속성 보장

## 📁 주요 파일 변경사항

### 1. `requirements.txt`
```
langchain>=0.2.0
langchain-openai>=0.1.0
langchain-core>=0.2.0
langchain-community>=0.2.0
```
추가된 의존성: LangChain 프레임워크 및 OpenAI 통합

### 2. `app/stores/chat_history.py` (신규)
SQLite 기반의 채팅 히스토리 저장소
- 테이블: `chat_sessions`, `chat_messages`
- 주요 메서드:
  - `get_chat_history(chat_id, limit=10)`: 최근 메시지 조회
  - `save_message(chat_id, role, content)`: 메시지 저장
  - `clear_chat_history(chat_id)`: 히스토리 삭제

### 3. `app/langchain_agent.py` (신규)
LangChain 기반 멀티턴 Agent
- `LangChainAgent` 클래스
- 메서드:
  - `process_message()`: 사용자 메시지 처리
  - `get_chat_history()`: 히스토리 로드
  - `add_tools()`: 도구 추가

### 4. `app/langchain_tools.py` (신규)
기존 도구들을 LangChain Tool로 래핑
- `search_web()`: 웹 검색 도구
- `open_url()`: URL 열기 도구
- `manage_memory()`: 메모리 관리 도구

### 5. `app/api/chat.py` (수정)
새로운 엔드포인트 추가
- `POST /api/chat/multiturn`: LangChain 멀티턴 대화
- `GET /api/chat/history/{chat_id}`: 히스토리 조회
- `DELETE /api/chat/history/{chat_id}`: 히스토리 삭제

### 6. `docker-compose.yml` (수정)
SQLite 데이터베이스 볼륨 추가
```yaml
volumes:
  - ./chat_history.db:/app/chat_history.db
```

### 7. `README.md` (수정)
새로운 기능 및 API 문서 추가

## 🚀 사용 방법

### 1. 환경 설정
```bash
# .env 파일 생성
echo "OPENAI_API_KEY=your_key" > .env
echo "OPENAI_MODEL=gpt-4o" >> .env
```

### 2. Docker로 실행
```bash
docker-compose up -d
```

### 3. API 호출 예시

#### 멀티턴 대화 시작
```bash
curl -X POST http://localhost:6666/api/chat/multiturn \
  -H "Content-Type: application/json" \
  -d '{
    "question": "안녕하세요. 오늘 날씨는 어떤가요?",
    "chatId": "session-123",
    "userInfo": {"id": "user-1"}
  }'
```

#### 같은 세션에서 두 번째 턴
```bash
curl -X POST http://localhost:6666/api/chat/multiturn \
  -H "Content-Type: application/json" \
  -d '{
    "question": "그럼 내일은?",
    "chatId": "session-123",
    "userInfo": {"id": "user-1"}
  }'
```

#### 히스토리 조회
```bash
curl http://localhost:6666/api/chat/history/session-123
```

#### 히스토리 삭제
```bash
curl -X DELETE http://localhost:6666/api/chat/history/session-123
```

## 📊 데이터베이스 스키마

### chat_sessions 테이블
```sql
CREATE TABLE chat_sessions (
    chat_id TEXT PRIMARY KEY,
    user_id TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    title TEXT
)
```

### chat_messages 테이블
```sql
CREATE TABLE chat_messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id TEXT NOT NULL,
    role TEXT NOT NULL,           -- 'user', 'assistant', 'tool'
    content TEXT NOT NULL,
    tool_call_id TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (chat_id) REFERENCES chat_sessions(chat_id)
)
```

## 🔄 멀티턴 대화 흐름

```
User Input 1
    ↓
LangChain Agent 처리 (최대 10 이전 메시지 + 현재 메시지)
    ↓
Tool 필요 시 자동 호출 (search_web, open_url, etc.)
    ↓
Response 1 생성 및 SQLite 저장
    ↓
---
User Input 2 (같은 chatId)
    ↓
LangChain Agent 처리 (최대 10개 메시지 윈도우: Input1, Response1, Input2)
    ↓
Tool 필요 시 자동 호출
    ↓
Response 2 생성 및 SQLite 저장
    ↓
... (반복)
```

## 🛠️ 추가 기능

### 사용자 메모리 (Bio Tool)
메시지 내에서 자동으로 사용자 정보를 메모리에 저장할 수 있습니다:
- 저장: "내 이름은 홍길동이고 개발자입니다"
- 조회: 이후 대화에서 자동으로 사용자 정보 활용

### 스트리밍 응답
SSE(Server-Sent Events)를 통한 실시간 토큰 스트리밍으로 빠른 응답 체감

## 📝 중요 사항

1. **메모리 윈도우 크기**: 최대 10개 메시지로 제한
   - 비용 효율성과 응답 속도 균형
   - 필요시 `max_history` 파라미터로 조정 가능

2. **도구 바인딩**: LangChain Agent가 필요시 자동으로 도구 선택
   - Agent가 도구 호출을 판단하므로 명시적 호출 불필요

3. **데이터 지속성**: SQLite 데이터베이스
   - Docker 볼륨을 통해 컨테이너 재시작 후에도 데이터 유지
   - 필요시 `chat_history.db` 백업

4. **호환성**: 기존 `/api/chat/stream` 엔드포인트 유지
   - 기존 클라이언트 호환성 유지
   - 필요시 두 엔드포인트 동시 사용 가능

## 🔧 개발 모드 실행

```bash
# 의존성 설치
pip install -r requirements.txt

# 백엔드 실행
uvicorn app.main:app --host 0.0.0.0 --port 6666 --reload

# 프론트엔드 실행 (다른 터미널)
cd web-gpt-mate
npm install
npm run dev
```

## 📚 참고 자료

- [LangChain 공식 문서](https://python.langchain.com/)
- [OpenAI Tool Calling API](https://platform.openai.com/docs/guides/function-calling)
- [SQLite Python](https://docs.python.org/3/library/sqlite3.html)
