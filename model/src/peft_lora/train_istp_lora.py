import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer, 
    DataCollatorForLanguageModeling
)
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
from datasets import Dataset
import json
import os
from config import *

# CUDA 캐시 초기화
torch.cuda.empty_cache()

def load_data():
    """ISTP 데이터를 로드합니다."""
    data_path = os.path.join(DATA_DIR, "gpt_produced/mbti_message.jsonl")
    
    with open(data_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    
    # 데이터셋 생성
    dataset = Dataset.from_dict({
        "text": [f"입력: {item['original']}\nISTP 스타일: {item['istp']}" for item in data]
    })
    
    # 데이터셋 분할 (90% 학습, 10% 평가)
    split_dataset = dataset.train_test_split(test_size=0.1, seed=42)
    
    print(f"Loaded {len(data)} samples for ISTP")
    print(f"Train set size: {len(split_dataset['train'])}")
    print(f"Eval set size: {len(split_dataset['test'])}")
    
    return split_dataset

def preprocess_function(examples, tokenizer):
    """데이터 전처리 함수"""
    # 입력 텍스트 토큰화
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length",
        return_tensors=None
    )
    
    # DEBUG: 토큰화 결과 확인
    print("\n=== Tokenization Debug ===")
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")
    
    # 레이블 설정 (패딩 토큰은 -100으로 마스킹)
    labels = []
    for i in range(len(tokenized["input_ids"])):
        label = tokenized["input_ids"][i].copy()
        # 패딩 토큰을 -100으로 마스킹
        label = [-100 if token == tokenizer.pad_token_id else token for token in label]
        labels.append(label)
    
    tokenized["labels"] = labels
    
    # DEBUG: 샘플 확인
    if len(tokenized["input_ids"]) > 0:
        print("\nSample tokenization:")
        print(f"Input IDs: {tokenized['input_ids'][0][:10]}")
        print(f"Labels: {tokenized['labels'][0][:10]}")
        try:
            # 음수 토큰 제거 후 디코딩
            valid_tokens = [x for x in tokenized["input_ids"][0][:20] if x >= 0]
            decoded = tokenizer.decode(valid_tokens, skip_special_tokens=True)
            print(f"Decoded text: {decoded}")
        except Exception as e:
            print(f"Decoding error: {e}")
    print("=" * 50)
    
    return tokenized

def train_istp_lora():
    """ISTP에 대한 LoRA 학습을 수행합니다."""
    print("Training LoRA for ISTP...")
    
    # 모델과 토크나이저 초기화
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_NAME,
        use_fast=False,
        trust_remote_code=True
    )
    
    # 패딩 토큰 설정
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # 모델 로드
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
        quantization_config={
            "load_in_4bit": True,
            "bnb_4bit_compute_dtype": torch.float16,
            "bnb_4bit_use_double_quant": True,
            "bnb_4bit_quant_type": "nf4"
        },
        max_position_embeddings=MAX_LENGTH
    )
    
    # 모델의 패딩 토큰 ID 설정
    model.config.pad_token_id = tokenizer.pad_token_id
    
    # DEBUG: 토크나이저와 모델의 어휘 크기 비교
    print("\n=== Vocabulary Size Check ===")
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")
    print(f"Model embedding size: {model.get_input_embeddings().weight.shape[0]}")
    print("=" * 50)
    
    # LoRA 설정
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none"
    )
    
    # 모델 준비
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)
    
    # 데이터 로드 및 전처리
    dataset = load_data()
    
    # 데이터 전처리 함수
    def preprocess(examples):
        return preprocess_function(examples, tokenizer)
    
    # 학습 및 평가 데이터셋 전처리
    tokenized_dataset = dataset.map(
        preprocess,
        batched=True,
        batch_size=8,
        remove_columns=dataset["train"].column_names,
        num_proc=1
    )
    
    # DEBUG: 데이터셋 샘플 확인
    print("\n=== Dataset Sample Check ===")
    for i in range(3):
        sample = tokenized_dataset["train"][i]
        print(f"\nSample {i+1}:")
        print(f"Input IDs: {sample['input_ids'][:10]}...")
        print(f"Labels: {sample['labels'][:10]}...")
        # 음수 토큰 제거 후 디코딩
        valid_tokens = [x for x in sample['input_ids'][:20] if x >= 0]
        print(f"Text: {tokenizer.decode(valid_tokens, skip_special_tokens=True)}...")
    print("=" * 50)
    
    # 학습 인자 설정
    training_args = TrainingArguments(
        output_dir=os.path.join(CHECKPOINT_DIR, "istp"),
        num_train_epochs=10,  # 에포크 수 증가
        per_device_train_batch_size=2,  # 배치 사이즈 감소
        per_device_eval_batch_size=2,  # 평가 배치 사이즈 감소
        gradient_accumulation_steps=8,  # 그래디언트 누적 스텝 증가
        learning_rate=LEARNING_RATE,
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
        optim="paged_adamw_8bit",  # 8비트 옵티마이저 사용
        max_grad_norm=1.0,  # 그래디언트 클리핑 값 조정
        warmup_ratio=0.1,  # 웜업 비율 증가
        lr_scheduler_type="cosine_with_restarts",  # 코사인 스케줄러 with restarts
        evaluation_strategy="epoch",  # 에포크마다 평가
        load_best_model_at_end=True,  # 최고 성능 모델 저장
        metric_for_best_model="eval_loss",  # 평가 지표
        save_total_limit=3,  # 최근 3개 체크포인트만 저장
        report_to=None,  # 로깅 비활성화
        dataloader_pin_memory=False  # 메모리 핀닝 비활성화
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
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        data_collator=data_collator,
        tokenizer=tokenizer
    )
    
    # 모델 학습
    trainer.train()
    
    # 최종 평가
    eval_results = trainer.evaluate()
    if "eval_loss" in eval_results:
        eval_loss = eval_results["eval_loss"]
        perplexity = torch.exp(torch.tensor(eval_loss))
        eval_results["perplexity"] = perplexity.item()
    
    print(f"Final evaluation results: {eval_results}")
    
    # LoRA 가중치 저장
    save_path = os.path.join(CHECKPOINT_DIR, "istp")
    trainer.save_model(save_path)
    tokenizer.save_pretrained(save_path)
    
    # 추론 모드로 전환
    model.eval()
    
    print("Completed training for ISTP")
    
    # GPU 메모리 정리
    torch.cuda.empty_cache()

if __name__ == "__main__":
    # CUDA 디버깅을 위한 환경변수 설정
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    
    # 체크포인트 디렉토리 생성
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    # ISTP LoRA 학습 수행
    train_istp_lora() 