import json
from pathlib import Path
from typing import Dict, List, Optional
from datasets import Dataset
from transformers import PreTrainedTokenizer

class MBTIDataset:
    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        mbti_type: Optional[str] = None
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mbti_type = mbti_type
        
        # 데이터 로드
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
            
        # 특정 MBTI 타입 선택
        if mbti_type:
            if mbti_type not in self.data:
                raise ValueError(f"MBTI type {mbti_type} not found in dataset")
            self.data = {mbti_type: self.data[mbti_type]}
            
        # 데이터셋 생성
        self.dataset = self._create_dataset()
        
    def _create_dataset(self) -> Dataset:
        examples = []
        
        for mbti_type, sentences in self.data.items():
            for sentence in sentences:
                # 입력 텍스트 생성
                input_text = json.dumps({
                    "target_MBTI": mbti_type,
                    "text": sentence
                }, ensure_ascii=False)
                
                # 출력 텍스트 생성
                output_text = json.dumps({
                    "response": sentence
                }, ensure_ascii=False)
                
                # 토큰화
                inputs = self.tokenizer(
                    input_text,
                    max_length=self.max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt"
                )
                
                outputs = self.tokenizer(
                    output_text,
                    max_length=self.max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt"
                )
                
                examples.append({
                    "input_ids": inputs["input_ids"].squeeze(),
                    "attention_mask": inputs["attention_mask"].squeeze(),
                    "labels": outputs["input_ids"].squeeze()
                })
        
        return Dataset.from_list(examples)
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Dict:
        return self.dataset[idx] 