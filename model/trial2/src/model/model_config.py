from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelConfig:
    # 기본 모델 설정
    model_name: str = "beomi/llama-2-ko-7b"  # Llama 2 Ko 모델 사용
    max_length: int = 128
    batch_size: int = 4
    learning_rate: float = 2e-4
    num_epochs: int = 3
    
    # LoRA 설정
    use_lora: bool = True
    lora_rank: int = 8
    lora_alpha: float = 16
    lora_dropout: float = 0.05
    target_modules: list = None
    
    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj"
            ] 