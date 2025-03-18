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
            
        # MBTI 토큰 생성
        self.mbti_types = [
            "ISTJ", "ISFJ", "INFJ", "INTJ",
            "ISTP", "ISFP", "INFP", "INTP",
            "ESTP", "ESFP", "ENFP", "ENTP",
            "ESTJ", "ESFJ", "ENFJ", "ENTJ"
        ]
        self.special_tokens = [f"{self.mbti_token_prefix}{mbti}{self.mbti_token_suffix}" 
                             for mbti in self.mbti_types]
        
        # 토크나이저 설정
        special_tokens_dict = {
            'additional_special_tokens': self.special_tokens,
            'pad_token': '[PAD]',
            'eos_token': '[EOS]',
            'bos_token': '[BOS]',
            'unk_token': '[UNK]',
            'sep_token': '[SEP]'
        }
        
        # 토크나이저에 특수 토큰 추가
        self.tokenizer.add_special_tokens(special_tokens_dict)
        
        # 토크나이저 설정 업데이트
        self.tokenizer.do_lower_case = False
        self.tokenizer.clean_up_tokenization_spaces = True
        self.tokenizer.add_prefix_space = True
            
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
        
        # 디버깅을 위한 토큰화 결과 출력
        print(f"\n=== 데이터 샘플 디버깅 ===")
        print(f"입력 텍스트: {input_text}")
        print(f"목표 텍스트: {target_text}")
        
        # 토큰화
        inputs = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
            add_special_tokens=True,
            return_attention_mask=True,
            return_token_type_ids=False
        )
        
        with self.tokenizer.as_target_tokenizer():
            targets = self.tokenizer(
                target_text,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt',
                add_special_tokens=True,
                return_attention_mask=True,
                return_token_type_ids=False
            )
        
        # 디버깅을 위한 토큰화 결과 출력
        print(f"입력 토큰: {self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])}")
        print(f"목표 토큰: {self.tokenizer.convert_ids_to_tokens(targets['input_ids'][0])}")
        print("========================\n")
        
        # 레이블에서 패딩 토큰을 -100으로 설정 (loss 계산에서 제외)
        labels = targets['input_ids'].squeeze()
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': labels
        } 