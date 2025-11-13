# Mocking Flowise

프론트엔드와 백엔드를 포함한 전체 스택 애플리케이션입니다.

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
- **api**: FastAPI (포트 6666)
- **redis**: Redis 서버 (세션 저장용)

### 개발 모드

개발 중에는 Docker Compose 대신 각각 실행할 수 있습니다:

**백엔드:**
```bash
cd mocking-flowise
python -m app.main
# 또는
uvicorn app.main:app --host 0.0.0.0 --port 6666 --reload
```

**프론트엔드:**
```bash
cd mocking-flowise/web-gpt-mate
npm run dev
```
