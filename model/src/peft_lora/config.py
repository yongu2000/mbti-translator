import os

# 기본 설정
BASE_MODEL_NAME = "Bllossom/llama-3.2-Korean-Bllossom-3B"
MAX_LENGTH = 256  # LLaMA는 더 긴 시퀀스를 처리할 수 있음
BATCH_SIZE = 4  # 메모리 사용량을 고려하여 배치 사이즈 조정
NUM_EPOCHS = 10
LEARNING_RATE = 2e-4
WARMUP_STEPS = 500
SAVE_STEPS = 500
LOGGING_STEPS = 100

# LoRA 설정
LORA_R = 8  # LoRA attention heads
LORA_ALPHA = 32  # LoRA alpha scaling
LORA_DROPOUT = 0.1
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]  # LLaMA의 attention 모듈

# 경로 설정
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(os.path.dirname(CURRENT_DIR))
DATA_DIR = os.path.join(BASE_DIR, "data")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints", "lora")

# MBTI 유형 목록
MBTI_TYPES = [
    "INTJ", "INTP", "ENTJ", "ENTP",
    "INFJ", "INFP", "ENFJ", "ENFP",
    "ISTJ", "ISFJ", "ESTJ", "ESFJ",
    "ISTP", "ISFP", "ESTP", "ESFP"
] 