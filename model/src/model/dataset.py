import json
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional
from transformers import PreTrainedTokenizer

class MBTIStyleDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        mbti_token_prefix: str = "<mbti=",
        mbti_token_suffix: str = ">"
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mbti_token_prefix = mbti_token_prefix
        self.mbti_token_suffix = mbti_token_suffix
        
        # 데이터 로드
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
            
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        
        # MBTI 토큰 생성
        source_mbti_token = f"{self.mbti_token_prefix}{item['source_mbti'].upper()}{self.mbti_token_suffix}"
        target_mbti_token = f"{self.mbti_token_prefix}{item['target_mbti'].upper()}{self.mbti_token_suffix}"
        
        # 입력 텍스트 구성
        input_text = f"{source_mbti_token} {target_mbti_token} {item['source_text']}"
        target_text = item['target_text']
        
        # 토크나이징
        inputs = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        targets = self.tokenizer(
            target_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': targets['input_ids'].squeeze()
        } 