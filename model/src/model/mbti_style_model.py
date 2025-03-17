import torch
import torch.nn as nn
from transformers import BartForConditionalGeneration, PreTrainedTokenizer
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
        
        # 특수 토큰 추가
        num_added_tokens = self.tokenizer.add_special_tokens({
            'additional_special_tokens': special_tokens
        })
        
        # 기본 모델 로드
        self.model = BartForConditionalGeneration.from_pretrained(model_name)
        
        # 토크나이저가 확장되었다면 임베딩 레이어도 확장
        if num_added_tokens > 0:
            self.model.resize_token_embeddings(len(tokenizer))
    
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
        return self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            num_beams=num_beams,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            **kwargs
        )
    
    def save_pretrained(self, path: str) -> None:
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path) 