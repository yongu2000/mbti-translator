import torch
import torch.nn as nn
from transformers import BartForConditionalGeneration, PreTrainedTokenizer, PreTrainedTokenizerFast
from typing import Dict, List, Optional, Union

class MBTIStyleTransformer(nn.Module):
    def __init__(
        self,
        model_name: str,
        tokenizer: PreTrainedTokenizer,
        special_tokens: List[str]
    ):
        super().__init__()
        self.tokenizer = tokenizer
        
        # 기본 모델 로드
        self.model = BartForConditionalGeneration.from_pretrained(model_name)
        
        # 특수 토큰 추가 및 임베딩 레이어 확장
        special_tokens_dict = {
            'additional_special_tokens': special_tokens,
            'pad_token': '[PAD]',
            'eos_token': '[EOS]',
            'bos_token': '[BOS]'
        }
        
        # 토크나이저에 특수 토큰 추가
        num_added_tokens = self.tokenizer.add_special_tokens(special_tokens_dict)
        print(f"Added {num_added_tokens} special tokens")
        print(f"Tokenizer vocab size after adding special tokens: {len(self.tokenizer)}")
        
        # 임베딩 레이어 확장
        self.model.resize_token_embeddings(len(self.tokenizer))
        print(f"Model vocab size after resizing: {self.model.config.vocab_size}")
        
        # 토크나이저 설정 업데이트
        self.tokenizer.do_lower_case = False  # 대소문자 구분
        self.tokenizer.clean_up_tokenization_spaces = False  # 토큰화 공백 유지
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        return outputs
    
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        num_beams: int = 5,
        max_length: int = 512,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        **kwargs
    ) -> torch.Tensor:
        # 입력 토큰 ID가 vocab_size를 초과하지 않는지 확인
        if torch.max(input_ids) >= self.model.config.vocab_size:
            raise ValueError(
                f"Input token IDs contain values >= vocab_size ({self.model.config.vocab_size}). "
                f"Max token ID found: {torch.max(input_ids)}"
            )
        
        # 기본 생성 파라미터 설정
        generation_config = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "num_beams": num_beams,
            "max_length": max_length,
            "min_length": 10,  # 최소 길이 증가
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "do_sample": True,  # 샘플링 활성화
            "repetition_penalty": 1.5,  # 반복 페널티 증가
            "length_penalty": 0.8,  # 길이 페널티 조정
            "early_stopping": True,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "bos_token_id": self.tokenizer.bos_token_id,
            "no_repeat_ngram_size": 4,  # n-gram 크기 증가
            "num_return_sequences": 1,
            "use_cache": True,
            "decoder_start_token_id": self.tokenizer.bos_token_id,  # 디코더 시작 토큰 설정
            "bad_words_ids": [
                [self.tokenizer.pad_token_id],
                [self.tokenizer.convert_tokens_to_ids("<unused")]  # unused 토큰 제외
            ],
            "forced_bos_token_id": self.tokenizer.bos_token_id,
            "forced_eos_token_id": self.tokenizer.eos_token_id,
            "remove_invalid_values": True
        }
        
        # 추가 파라미터 업데이트
        generation_config.update(kwargs)
        
        # 생성 실행
        outputs = self.model.generate(**generation_config)
        
        # 디버깅을 위한 출력 토큰 확인
        print("\n=== 생성 토큰 디버깅 ===")
        print("출력 토큰:", self.tokenizer.convert_ids_to_tokens(outputs[0]))
        print("출력 텍스트:", self.tokenizer.decode(outputs[0], skip_special_tokens=True))
        print("========================\n")
        
        return outputs
    
    def save_pretrained(self, path: str) -> None:
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    @classmethod
    def from_pretrained(cls, model_path: str):
        """사전 학습된 모델을 로드합니다."""
        # 토크나이저 로드
        tokenizer = PreTrainedTokenizerFast.from_pretrained(model_path)
        
        # MBTI 특수 토큰 생성
        mbti_types = [
            "ISTJ", "ISFJ", "INFJ", "INTJ",
            "ISTP", "ISFP", "INFP", "INTP",
            "ESTP", "ESFP", "ENFP", "ENTP",
            "ESTJ", "ESFJ", "ENFJ", "ENTJ"
        ]
        special_tokens = [f"<mbti={mbti}>" for mbti in mbti_types]
        
        # 모델 인스턴스 생성
        model = cls(
            model_name=model_path,
            tokenizer=tokenizer,
            special_tokens=special_tokens
        )
        
        return model 