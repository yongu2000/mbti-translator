from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelConfig:
    # 모델 기본 설정
    model_name: str = "gogamza/kobart-base-v2"
    max_length: int = 512
    batch_size: int = 4  # 테스트를 위해 배치 크기 감소
    learning_rate: float = 2e-5
    num_epochs: int = 2  # 테스트를 위해 에폭 수 감소
    
    # 데이터 관련 설정
    train_data_path: str = "data/processed/style_transfer_pairs_test.json"  # 테스트 데이터셋 사용
    output_dir: str = "checkpoints"
    
    # MBTI 관련 설정
    mbti_token_prefix: str = "<mbti="
    mbti_token_suffix: str = ">"
    
    # 학습 관련 설정
    warmup_steps: int = 10  # 테스트를 위해 감소
    logging_steps: int = 1
    save_steps: int = 10
    eval_steps: int = 5
    
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