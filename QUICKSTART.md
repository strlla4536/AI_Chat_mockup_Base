# 빠른 시작 가이드

## 🎯 5분 안에 시작하기

### 1단계: 환경 변수 설정
```bash
cd /Users/sangmin/Desktop/personal/mockup/AI_Chat_mockup_Base

# .env 파일 생성
cat > .env << EOF
OPENAI_API_KEY=sk-your-api-key-here
OPENAI_MODEL=gpt-4o
EOF
```

### 2단계: Docker Compose 실행
```bash
docker-compose up -d
```

### 3단계: 서비스 확인
```bash
# 모든 서비스가 running 상태인지 확인
docker-compose ps

# 로그 확인
docker-compose logs -f
```

### 4단계: 멀티턴 대화 테스트

```bash
# 첫 번째 메시지
curl -X POST http://localhost:6666/api/chat/multiturn \
  -H "Content-Type: application/json" \
  -d '{
    "question": "당신이 할 수 있는 일은 뭐가 있나요?",
    "chatId": "test-session-1",
    "userInfo": {"id": "test-user"}
  }'

# 두 번째 메시지 (같은 chatId로 멀티턴 유지)
curl -X POST http://localhost:6666/api/chat/multiturn \
  -H "Content-Type: application/json" \
  -d '{
    "question": "그럼 웹 검색도 가능한가요?",
    "chatId": "test-session-1",
    "userInfo": {"id": "test-user"}
  }'

# 히스토리 확인
curl http://localhost:6666/api/chat/history/test-session-1
```

## 📚 주요 API 엔드포인트

| 엔드포인트 | 메서드 | 설명 |
|-----------|--------|------|
| `/api/chat/multiturn` | POST | LangChain 멀티턴 대화 |
| `/api/chat/history/{chat_id}` | GET | 대화 히스토리 조회 |
| `/api/chat/history/{chat_id}` | DELETE | 히스토리 삭제 |
| `/api/chat/stream` | POST | 기존 OpenAI 스트리밍 (호환성) |

## 🔍 주요 특징

✅ **멀티턴 지원**: 같은 chatId로 여러 턴 대화 유지  
✅ **자동 Tool 바인딩**: Agent가 필요한 도구 자동 호출  
✅ **영구 저장**: SQLite에 모든 대화 저장  
✅ **윈도우 크기**: 최대 10개 메시지 유지  

## 🐛 문제 해결

### 1. OpenAI API 키 오류
```bash
# .env 파일에서 API 키 확인
cat .env | grep OPENAI_API_KEY

# Docker 로그에서 오류 확인
docker-compose logs api | grep -i error
```

### 2. 포트 충돌
```bash
# 포트 변경 (docker-compose.yml)
# 6666:6666 → 7777:6666
# 8080:80 → 8081:80
```

### 3. SQLite 데이터베이스 초기화
```bash
# 데이터베이스 파일 삭제 후 재시작
rm chat_history.db
docker-compose restart api
```

## 📊 응답 구조

### 성공 응답
```json
{
  "event": "token",
  "data": "응답 텍스트..."
}

{
  "event": "metadata",
  "data": {
    "chat_id": "test-session-1",
    "total_messages": 4,
    "intermediate_steps": [...]
  }
}

{
  "event": "result",
  "data": null
}
```

## 🧪 Python으로 테스트

```python
import aiohttp
import asyncio

async def test_multiturn():
    chat_id = "python-test-001"
    
    for turn, question in enumerate([
        "안녕하세요",
        "당신은 누구인가요?",
        "날씨를 검색해줄 수 있나요?"
    ], 1):
        print(f"\n=== Turn {turn} ===")
        print(f"Q: {question}")
        
        async with aiohttp.ClientSession() as session:
            data = {
                "question": question,
                "chatId": chat_id,
                "userInfo": {"id": "python-user"}
            }
            
            async with session.post(
                "http://localhost:6666/api/chat/multiturn",
                json=data
            ) as resp:
                async for line in resp.content:
                    if line.strip():
                        print(line.decode('utf-8', errors='ignore'), end='')

asyncio.run(test_multiturn())
```

## 📖 다음 단계

1. **프론트엔드 통합**: web-gpt-mate에서 `/api/chat/multiturn` 호출
2. **Tool 추가**: `app/langchain_tools.py`에 새로운 도구 추가
3. **시스템 프롬프트 커스터마이징**: `app/prompts/system.txt` 수정
4. **메모리 윈도우 조정**: `app/langchain_agent.py`에서 `max_history` 값 변경

## 💡 팁

- **디버깅**: `docker-compose logs -f api` 로 실시간 로그 확인
- **성능 최적화**: `max_history`를 더 작게 설정하면 더 빠름
- **비용 절감**: temperature 낮춰서 일관성 있는 응답 생성
- **커스터마이징**: system.txt 프롬프트로 AI 동작 튜닝 가능
