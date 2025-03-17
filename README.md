# MBTI 스타일 변환기

MBTI 스타일 변환기는 텍스트를 다른 MBTI 스타일로 변환하는 웹 애플리케이션입니다.

## 프로젝트 구조

```
mbti-translator/
├── model/                      # Python 모델 서버
│   ├── src/
│   │   ├── api/               # FastAPI 서버
│   │   │   ├── main.py        # FastAPI 앱 및 엔드포인트
│   │   │   └── run.py         # 서버 실행 스크립트
│   │   ├── model/             # 모델 관련 코드
│   │   │   ├── mbti_style_model.py  # MBTI 스타일 변환 모델
│   │   │   └── config.py      # 모델 설정
│   │   └── train.py           # 모델 학습 스크립트
│   └── requirements.txt        # Python 의존성
│
├── web/                        # 웹 애플리케이션
│   ├── frontend/              # Next.js 프론트엔드
│   │   ├── src/
│   │   │   ├── app/          # Next.js 13+ 앱 라우터
│   │   │   ├── components/   # React 컴포넌트
│   │   │   ├── api/          # API 요청 함수
│   │   │   └── types/        # TypeScript 타입 정의
│   │   ├── .env.local        # 환경 변수
│   │   └── package.json      # Node.js 의존성
│   │
│   └── backend/              # Spring Boot 백엔드
│       └── src/
│           └── main/
│               └── java/
│                   └── com/
│                       └── mbti/
│                           ├── controller/  # REST 컨트롤러
│                           ├── service/     # 비즈니스 로직
│                           └── config/      # 설정 클래스
```

## 기술 스택

### 백엔드
- Spring Boot: REST API 서버
- FastAPI: Python 모델 서버
- PyTorch: 딥러닝 모델
- Transformers: KoBART 기반 모델

### 프론트엔드
- Next.js 13+
- TypeScript
- Tailwind CSS
- React

## 설정 방법

### 1. Python 모델 서버 설정
```bash
cd model
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Spring Boot 서버 설정
```bash
cd web/backend
./gradlew build
./gradlew bootRun
```

### 3. 프론트엔드 설정
```bash
cd web/frontend
npm install
npm run dev
```

## 환경 변수

### 프론트엔드 (.env.local)
```
NEXT_PUBLIC_API_URL=http://localhost:8080
```

## API 엔드포인트

### Spring Boot 서버
- POST `/api/translate`: MBTI 스타일 변환 요청
  ```json
  {
    "sourceMbti": "ISTJ",
    "targetMbti": "ENFP",
    "text": "변환할 텍스트"
  }
  ```

### FastAPI 모델 서버
- POST `/translate`: MBTI 스타일 변환 처리
- GET `/health`: 서버 상태 확인

## 개발 서버 실행 순서

1. FastAPI 모델 서버 실행
```bash
cd model
python src/api/run.py
```

2. Spring Boot 서버 실행
```bash
cd web/backend
./gradlew bootRun
```

3. Next.js 개발 서버 실행
```bash
cd web/frontend
npm run dev
```

## 배포

각 서버는 독립적으로 배포할 수 있습니다:
- FastAPI 모델 서버: Python 서버 (예: AWS EC2)
- Spring Boot 서버: Java 서버 (예: AWS EC2)
- Next.js 프론트엔드: 정적 호스팅 (예: Vercel) 