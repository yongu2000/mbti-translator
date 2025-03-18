import os
from pathlib import Path
from typing import Optional
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from datasets import load_dataset
from src.model.model_config import ModelConfig
from src.data.mbti_dataset import MBTIDataset

def train_lora(
    config: ModelConfig,
    output_dir: str = "../outputs/lora_mbti",
    data_path: str = "../data/processed/mbti_speech_patterns_cleaned.json",
    mbti_type: str = "ISTP"
):
    # 출력 디렉토리 생성
    output_dir = Path(output_dir) / mbti_type
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{mbti_type} MBTI 유형 학습 시작...")
    
    # 8비트 양자화 설정
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False
    )
    
    # 모델과 토크나이저 로드
    print("모델과 토크나이저 로딩 중...")
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        device_map="auto",
        torch_dtype=torch.float16,
        quantization_config=quantization_config,
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        trust_remote_code=True
    )
    
    # LoRA 설정
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.target_modules
    )
    
    # LoRA 모델 준비
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)
    
    # 데이터셋 로드
    print("데이터셋 로딩 중...")
    dataset = MBTIDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        max_length=config.max_length,
        mbti_type=mbti_type
    )
    
    if len(dataset) == 0:
        print(f"경고: {mbti_type} 유형에 대한 데이터가 없습니다. 건너뜁니다.")
        return
    
    print(f"{mbti_type} 유형 데이터 수: {len(dataset)}")
    
    # 학습 인자 설정
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=config.learning_rate,
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="no",
        save_total_limit=3,
        remove_unused_columns=False,
        push_to_hub=False,
        report_to="tensorboard"
    )
    
    # 데이터 콜레이터 설정
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # 트레이너 초기화
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        tokenizer=tokenizer
    )
    
    # 학습 시작
    print("학습 시작...")
    trainer.train()
    
    # 모델 저장
    print("모델 저장 중...")
    trainer.save_model()
    
    # LoRA 가중치 저장
    model.save_pretrained(os.path.join(output_dir, "lora_weights"))
    
    print(f"{mbti_type} MBTI 유형 학습 완료!")

def get_all_mbti_types():
    """모든 가능한 MBTI 유형 생성"""
    preferences = ['E', 'I'], ['S', 'N'], ['T', 'F'], ['J', 'P']
    mbti_types = []
    for p1 in preferences[0]:
        for p2 in preferences[1]:
            for p3 in preferences[2]:
                for p4 in preferences[3]:
                    mbti_types.append(f"{p1}{p2}{p3}{p4}")
    return mbti_types

if __name__ == "__main__":
    # 설정 로드
    config = ModelConfig()
    
    # 모든 MBTI 유형 가져오기
    mbti_types = get_all_mbti_types()
    
    # 각 MBTI 유형별로 학습 실행
    for mbti_type in mbti_types:
        train_lora(
            config=config,
            output_dir="../outputs/lora_mbti",
            data_path="../data/processed/mbti_speech_patterns_cleaned.json",
            mbti_type=mbti_type
        )
    
    print("\n모든 MBTI 유형 학습 완료!") 