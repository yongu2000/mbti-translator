from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelConfig:
    # 모델 기본 설정
    model_name: str = "gogamza/kobart-base-v2"
    max_length: int = 512
    batch_size: int = 16
    learning_rate: float = 2e-5
    num_epochs: int = 3
    
    # 데이터 관련 설정
    train_data_path: str = "data/processed/style_transfer_pairs.json"
    output_dir: str = "checkpoints"
    
    # MBTI 관련 설정
    mbti_token_prefix: str = "<mbti="
    mbti_token_suffix: str = ">"
    
    # 학습 관련 설정
    warmup_steps: int = 500
    logging_steps: int = 100
    save_steps: int = 1000
    eval_steps: int = 500
    
    # 생성 관련 설정
    num_beams: int = 5
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 0.9
    
    # GPU 설정
    use_cuda: bool = True
    
    def get_special_tokens(self) -> list[str]:
        """MBTI 특수 토큰 리스트 반환"""
        mbti_types = [
            "ISTJ", "ISFJ", "INFJ", "INTJ",
            "ISTP", "ISFP", "INFP", "INTP",
            "ESTP", "ESFP", "ENFP", "ENTP",
            "ESTJ", "ESFJ", "ENFJ", "ENTJ"
        ]
        return [f"{self.mbti_token_prefix}{mbti}{self.mbti_token_suffix}" 
                for mbti in mbti_types] 