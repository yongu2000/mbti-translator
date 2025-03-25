import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from peft import get_peft_model, LoraConfig, TaskType
from datasets import Dataset
import json
import os
from config import *

def load_data(mbti_type):
    """특정 MBTI 유형의 데이터를 로드합니다."""
    data_path = os.path.join(DATA_DIR, "gpt_produced/mbti_message.jsonl")
    
    with open(data_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    
    # 데이터셋 생성
    dataset = Dataset.from_dict({
        "input_text": [item["original"] for item in data],
        "target_text": [item[mbti_type.lower()] for item in data]  # 해당 MBTI 유형의 응답을 타겟으로 설정
    })
    
    print(f"Loaded {len(data)} samples for {mbti_type}")
    return dataset

def preprocess_function(examples, tokenizer):
    """데이터 전처리 함수"""
    inputs = examples["input_text"]
    targets = examples["target_text"]
    
    # 입력과 타겟을 결합하여 프롬프트 형식으로 만듦
    prompts = [
        f"입력: {input}\nMBTI 스타일: {target}"
        for input, target in zip(inputs, targets)
    ]
    
    # 토큰화
    model_inputs = tokenizer(
        prompts,
        max_length=MAX_LENGTH,
        truncation=True,
        padding="max_length"  # 패딩을 max_length로 설정
    )
    
    # 레이블 설정 (입력 부분은 -100으로 마스킹)
    labels = model_inputs["input_ids"].copy()
    for i, label in enumerate(labels):
        # "입력: " 부분까지는 -100으로 마스킹
        input_text = inputs[i]
        input_tokens = tokenizer(f"입력: {input_text}\nMBTI 스타일: ", add_special_tokens=False)["input_ids"]
        for j in range(len(input_tokens)):
            labels[i][j] = -100
    
    model_inputs["labels"] = labels
    return model_inputs

def train_lora_for_mbti(mbti_type):
    """특정 MBTI 유형에 대한 LoRA 학습을 수행합니다."""
    print(f"Training LoRA for {mbti_type}...")
    
    # 모델과 토크나이저 초기화
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=torch.float16,  # float16으로 설정
        device_map="auto"  # 자동 디바이스 매핑
    )
    
    # 패딩 토큰 설정
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
    
    # LoRA 설정
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES
    )
    
    # LoRA 모델 생성
    model = get_peft_model(model, peft_config)
    
    # 데이터 로드 및 전처리
    dataset = load_data(mbti_type)
    tokenized_dataset = dataset.map(
        lambda x: preprocess_function(x, tokenizer),
        batched=True,
        batch_size=32  # 배치 사이즈 설정
    )
    
    # 학습 인자 설정
    training_args = TrainingArguments(
        output_dir=os.path.join(CHECKPOINT_DIR, mbti_type),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        save_steps=SAVE_STEPS,
        save_total_limit=2,
        logging_steps=LOGGING_STEPS,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        warmup_steps=WARMUP_STEPS,
        fp16=True,  # fp16 활성화
        gradient_accumulation_steps=4,
        evaluation_strategy="no",
        load_best_model_at_end=False,
        metric_for_best_model=None,
        remove_unused_columns=False  # 사용하지 않는 컬럼 제거 비활성화
    )
    
    # 데이터 콜레이터 설정
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        return_tensors="pt"
    )
    
    # 트레이너 초기화
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer
    )
    
    # 모델 학습
    trainer.train()
    
    # LoRA 가중치 저장
    save_path = os.path.join(CHECKPOINT_DIR, mbti_type)
    trainer.save_model(save_path)
    tokenizer.save_pretrained(save_path)
    
    print(f"Completed training for {mbti_type}")

def main():
    # 체크포인트 디렉토리 생성
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    # 각 MBTI 유형별로 LoRA 학습 수행
    for mbti_type in MBTI_TYPES:
        train_lora_for_mbti(mbti_type)

if __name__ == "__main__":
    main() 