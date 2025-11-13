# Mocking Flowise

프론트엔드와 백엔드를 포함한 전체 스택 애플리케이션입니다.

## 주요 기능

### LangChain 멀티턴 대화
- **멀티턴 대화**: 최대 10개 메시지 윈도우로 대화 맥락 유지
- **Tool Binding**: LangChain 기반의 자동 tool 바인딩
- **SQLite 메모리**: 대화 히스토리를 SQLite에 영구 저장
- **Agent 지원**: LangChain Agent로 자동 tool 호출 및 실행

### API 엔드포인트

#### 기존 엔드포인트
- `POST /api/chat/stream` - 기존 OpenAI 스트리밍 (호환성 유지)

#### 새로운 LangChain 멀티턴 엔드포인트
- `POST /api/chat/multiturn` - LangChain Agent 기반 멀티턴 대화
  ```json
  {
    "question": "질문 내용",
    "chatId": "채팅 ID (선택)",
    "userInfo": {"id": "사용자 ID"}
  }
  ```

#### 히스토리 관리
- `GET /api/chat/history/{chat_id}` - 특정 채팅의 히스토리 조회 (최대 10개)
- `DELETE /api/chat/history/{chat_id}` - 채팅 히스토리 삭제

## Docker Compose로 실행하기

### 사전 요구사항

- Docker
- Docker Compose

### 환경 변수 설정

프로젝트 루트에 `.env` 파일을 생성하고 필요한 환경 변수를 설정하세요:

```env
OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL=gpt-4o
```

**환경 변수 설명:**
- `OPENAI_API_KEY`: OpenAI API 키 (필수)
- `OPENAI_MODEL`: 사용할 OpenAI 모델 (선택, 기본값: gpt-4o)

### 실행 방법

1. **모든 서비스 실행 (프론트엔드 + 백엔드 + Redis)**
   ```bash
   docker-compose up -d
   ```

2. **로그 확인**
   ```bash
   docker-compose logs -f
   ```

3. **특정 서비스 로그만 확인**
   ```bash
   docker-compose logs -f api
   docker-compose logs -f frontend
   ```

4. **서비스 중지**
   ```bash
   docker-compose down
   ```

5. **서비스 중지 및 볼륨 삭제**
   ```bash
   docker-compose down -v
   ```

6. **이미지 재빌드 후 실행**
   ```bash
   docker-compose up -d --build
   ```

### 접속 정보

- **프론트엔드**: http://localhost:8080
- **백엔드 API**: http://localhost:6666
- **API 문서**: http://localhost:6666/docs

### 서비스 구성

- **frontend**: React + Vite + Nginx (포트 8080)
- **api**: FastAPI + LangChain (포트 6666)
- **redis**: Redis 서버 (세션 저장용)

### 저장 위치

- **채팅 히스토리**: `chat_history.db` (SQLite)
- **Redis 데이터**: Docker 볼륨 `redis-data`

## 개발 모드

개발 중에는 Docker Compose 대신 각각 실행할 수 있습니다:

**백엔드 (LangChain 멀티턴 지원):**
```bash
cd AI_Chat_mockup_Base
pip install -r requirements.txt
python -m app.main
# 또는
uvicorn app.main:app --host 0.0.0.0 --port 6666 --reload
```

**프론트엔드:**
```bash
cd web-gpt-mate
npm install
npm run dev
```

## 멀티턴 대화 사용 예시

```python
import aiohttp
import asyncio

async def test_multiturn():
    chat_id = "test-chat-123"
    
    # 첫 번째 턴
    async with aiohttp.ClientSession() as session:
        data = {
            "question": "안녕하세요. 당신의 이름은?",
            "chatId": chat_id,
            "userInfo": {"id": "user-1"}
        }
        
        async with session.post("http://localhost:6666/api/chat/multiturn", json=data) as resp:
            async for line in resp.content:
                print(line.decode('utf-8'))
    
    # 두 번째 턴 (같은 chat_id로 멀티턴 유지)
    async with aiohttp.ClientSession() as session:
        data = {
            "question": "그럼 당신은 무엇을 할 수 있나요?",
            "chatId": chat_id,
            "userInfo": {"id": "user-1"}
        }
        
        async with session.post("http://localhost:6666/api/chat/multiturn", json=data) as resp:
            async for line in resp.content:
                print(line.decode('utf-8'))

asyncio.run(test_multiturn())
```

## 아키텍처

```
┌─────────────────────┐
│   Frontend (React)  │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  FastAPI Backend    │
├─────────────────────┤
│ - Chat API (SSE)    │
│ - Multiturn API     │
│ - History API       │
└──────────┬──────────┘
           │
    ┌──────┴──────┐
    │             │
    ▼             ▼
┌────────┐  ┌──────────┐
│ Redis  │  │  SQLite  │
│ (Cache)│  │(History) │
└────────┘  └──────────┘
           │
           ▼
    ┌─────────────┐
    │ LangChain   │
    │ + OpenAI    │
    │ + Tools     │
    └─────────────┘
```

## LangChain Tool 목록

- **search_web**: 웹 검색 기능
- **open_url**: URL 열기 및 내용 추출
- **manage_memory**: 사용자 메모리 관리 (bio tool)
