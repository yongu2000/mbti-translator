import os
import json
import logging
from pathlib import Path
from typing import Dict, List

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset
import numpy as np
from dotenv import load_dotenv
from config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StyleLoRATrainer:
    def __init__(self, mbti_type: str):
        self.mbti_type = mbti_type.upper()
        load_dotenv()
        
        # 기본 설정
        self.model_name = Config.BASE_MODEL
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            token=Config.HF_TOKEN,
            trust_remote_code=True
        )
        
        # 토크나이저 설정
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        
        # 4-bit 양자화 설정
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        
        logger.info("모델 로딩 중...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            token=Config.HF_TOKEN,
            trust_remote_code=True
        )
        
        # LoRA 설정
        self.lora_config = LoraConfig(
            r=Config.LORA_R,
            lora_alpha=Config.LORA_ALPHA,
            lora_dropout=Config.LORA_DROPOUT,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj"
            ]
        )
        
        # 모델 준비
        self.model = prepare_model_for_kbit_training(self.model)
        self.model = get_peft_model(self.model, self.lora_config)
        self.model.print_trainable_parameters()
        
    def _load_dataset(self) -> Dataset:
        """JSONL 파일에서 데이터셋 로드"""
        data_path = f"data/sampled_style/{self.mbti_type.lower()}_conversations.jsonl"
        texts = []
        
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                texts.append(data['output'])
                
        return Dataset.from_dict({"text": texts})
    
    def _tokenize_function(self, examples: Dict) -> Dict:
        """텍스트 토크나이징"""
        tokenized = self.tokenizer(
            examples["text"],
            padding=True,
            truncation=True,
            max_length=Config.MAX_LENGTH,
            return_special_tokens_mask=True
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized
    
    def train(self):
        """LoRA 학습 실행"""
        # 데이터셋 준비
        dataset = self._load_dataset()
        tokenized_dataset = dataset.map(
            self._tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )
        
        # 학습 설정
        training_args = TrainingArguments(
            output_dir=f"models/lora_style/{self.mbti_type.lower()}",
            num_train_epochs=Config.NUM_EPOCHS,
            per_device_train_batch_size=Config.BATCH_SIZE,
            gradient_accumulation_steps=4,
            warmup_steps=100,
            logging_steps=10,
            save_steps=100,
            learning_rate=Config.LEARNING_RATE,
            fp16=True,
            optim="paged_adamw_32bit",
            logging_dir=f"logs/{self.mbti_type.lower()}",
            save_total_limit=3,
            load_best_model_at_end=False,
            group_by_length=True,
            report_to=None,
        )
        
        # Data collator 설정
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        # Trainer 초기화 및 학습
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator,
        )
        
        logger.info(f"{self.mbti_type} 스타일 학습 시작...")
        trainer.train()
        
        # 모델 저장
        trainer.save_model()
        logger.info(f"학습 완료: models/lora_style/{self.mbti_type.lower()}")

def main():
    # CUDA 디버깅을 위한 환경변수 설정
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    
    # Config에서 MBTI 유형 가져오기
    mbti_types = Config.MBTI_TYPES
    
    for mbti_type in mbti_types:
        logger.info(f"\n=== {mbti_type} 스타일 학습 시작 ===")
        trainer = StyleLoRATrainer(mbti_type)
        trainer.train()

if __name__ == "__main__":
    main() 