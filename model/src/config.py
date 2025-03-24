class Config:
    # 모델 설정
    BASE_MODEL = "meta-llama/Llama-2-7b-hf"  # 또는 사용 가능한 다른 LLaMA 모델
    LORA_R = 8
    LORA_ALPHA = 16
    LORA_DROPOUT = 0.05
    
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
    LORA_WEIGHTS_PATH = "models/lora/"
    DATA_PATH = "data/processed/" 