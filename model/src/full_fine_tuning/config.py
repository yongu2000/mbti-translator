import os
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

class Config:
    # 모델 설정
    BASE_MODEL = "gogamza/kobart-base-v2"  # 실제 모델명으로 변경 필요
    HF_TOKEN = os.getenv('HF_TOKEN')  # .env 파일에서 HF_TOKEN 환경변수 로드
    
    # 학습 설정
    BATCH_SIZE = 4
    LEARNING_RATE = 2e-4
    NUM_EPOCHS = 3
    MAX_LENGTH = 512
    
    # MBTI 타입 리스트
    MBTI_TYPES = [
        "ISTJ", "ISFJ", "INFJ", "INTJ",
        "ISTP", "ISFP", "INFP", "INTP",
        "ESTP", "ESFP", "ENFP", "ENTP",
        "ESTJ", "ESFJ", "ENFJ", "ENTJ"
    ]
    
    # 경로 설정
    DATA_PATH = "data/gpt_produced" 