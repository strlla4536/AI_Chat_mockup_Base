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

**`.env` 파일 생성 방법:**
```bash
# env.example 파일을 복사하여 .env 파일 생성
cp env.example .env

# 또는 직접 생성
touch .env
```

그 다음 `.env` 파일을 열어서 실제 값으로 수정하세요:

```env
# OpenAI 직접 사용 (GenOS 미사용 시)
OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL=gpt-4o

# GenOS 리소스 사용 (권장)
GENOS_ID=your_genos_user_id
GENOS_PW=your_genos_password
GENOS_URL=https://genos.mnc.ai:3443
GENOS_LLM_SERVING_ID=your_llm_serving_id  # GenOS LLM 서빙 ID (설정 시 GenOS 사용)
GENOS_BEARER_TOKEN=your_bearer_token  # 선택사항 (없으면 자동 획득)
MCP_SERVER_ID=3371,other_server_id  # MCP 서버 ID 목록 (쉼표로 구분)

# 기타
TAVILY_API_KEY=your_tavily_api_key  # 검색 도구
```

**환경 변수 설명:**
- `OPENAI_API_KEY`: OpenAI API 키 (GenOS 미사용 시 필수)
- `OPENAI_MODEL`: 사용할 모델명 (기본값: gpt-4o)
- `GENOS_ID`, `GENOS_PW`: GenOS 인증 정보 (GenOS 사용 시 필수)
- `GENOS_URL`: GenOS API URL (기본값: https://genos.mnc.ai:3443)
- `GENOS_LLM_SERVING_ID`: GenOS LLM 서빙 ID (설정 시 GenOS를 통해 OpenRouter 모델 사용)
- `GENOS_BEARER_TOKEN`: GenOS Bearer 토큰 (선택사항, 없으면 자동 획득)
- `MCP_SERVER_ID`: MCP 서버 ID 목록 (쉼표로 구분, 예: "3371,143")
- `TAVILY_API_KEY`: 검색 도구 API 키 (필수)

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
   # 서비스 완전 중지 및 컨테이너 제거 (권장)
   docker-compose down
   
   # 서비스 중지 및 볼륨까지 삭제 (데이터 완전 삭제)
   docker-compose down -v
   ```

5. **서비스 일시 중지 (재시작 가능)**
   ```bash
   # 모든 서비스 일시 중지 (컨테이너는 유지)
   docker-compose stop
   
   # 특정 서비스만 중지
   docker-compose stop api
   docker-compose stop frontend
   docker-compose stop redis
   
   # 일시 중지된 서비스 재시작
   docker-compose start
   ```

6. **이미지 재빌드 후 실행**
   ```bash
   docker-compose up -d --build
   ```

7. **실행 중인 서비스 상태 확인**
   ```bash
   docker-compose ps
   ```

8. **서비스 로그 실시간 확인**
   ```bash
   # 모든 서비스 로그
   docker-compose logs -f
   
   # 특정 서비스 로그만
   docker-compose logs -f api
   docker-compose logs -f frontend
   ```

### 접속 정보

- **프론트엔드**: http://localhost:8080
- **백엔드 API**: http://localhost:6666
- **API 문서**: http://localhost:6666/docs

### 서비스 구성

- **frontend**: React + Vite + Nginx (포트 8080)
- **api**: FastAPI + LangChain + LangGraph(포트 6666)
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
