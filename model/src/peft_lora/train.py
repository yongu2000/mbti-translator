import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
import os
import json

class LoraStyleDataset(Dataset):
    def __init__(self, jsonl_path: str, tokenizer: PreTrainedTokenizer, max_length: int = 64):
        self.tokenizer = tokenizer
        self.max_length = max_length

        # 데이터 불러오기
        with open(jsonl_path, "r", encoding="utf-8") as f:
            self.samples = [json.loads(line.strip()) for line in f]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        input_text = item["input"]
        output_text = item["output"]

        # 프롬프트 제거 (예: "istp 말투로 변환: " 부분)
        if "말투로 변환:" in input_text:
            input_text = input_text.split("말투로 변환:")[1].strip()

        # 인코더 입력
        model_inputs = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        # 디코더 입력 (label)
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                output_text,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )

        # token_type_ids 제거 & squeeze
        model_inputs = {k: v.squeeze() for k, v in model_inputs.items() if k != 'token_type_ids'}
        model_inputs["labels"] = labels["input_ids"].squeeze()


        return model_inputs

# 설정
model_name = "gogamza/kobart-base-v2"
lora_target_mbti = "istp"  # 원하는 MBTI 말투 이름

# 프로젝트 루트 디렉토리 설정
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
input_dir = os.path.join(project_root, "model", "data", "lora", f"{lora_target_mbti}_message.jsonl")
output_dir = os.path.join(project_root, "model", "lora", f"lora-kobart-{lora_target_mbti}")

# 출력 디렉토리 생성
os.makedirs(output_dir, exist_ok=True)

# 토크나이저 & 모델 로드
tokenizer = AutoTokenizer.from_pretrained(model_name)
base_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# 모델을 학습 모드로 설정
base_model.train()

# LoRA 설정
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],  # KoBART의 attention 모듈
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.SEQ_2_SEQ_LM,
    inference_mode=False
)

# LoRA 모델 래핑
lora_model = get_peft_model(base_model, peft_config)
lora_model.print_trainable_parameters()  # 학습 가능한 파라미터 수 출력

# 데이터셋 생성
dataset = LoraStyleDataset(input_dir, tokenizer)
data_collator = DataCollatorForSeq2Seq(tokenizer, model=lora_model)

# 학습 파라미터 설정
training_args = Seq2SeqTrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=True,
    per_device_train_batch_size=8,
    num_train_epochs=5,
    save_steps=500,
    save_total_limit=2,
    logging_dir=f"{output_dir}/logs",
    logging_steps=100,
    evaluation_strategy="no",
    learning_rate=1e-4,
    fp16=True,
    gradient_accumulation_steps=4,
    gradient_checkpointing=False,
    report_to="none",
    label_names=["labels"],  # label_names 추가
    # dataloader_pin_memory=True,  # 데이터 로딩 성능 향상
    # dataloader_num_workers=4,  # 데이터 로딩 병렬화
    # dataloader_drop_last=True,  # 마지막 불완전한 배치 제외
    # shuffle=True,  # epoch마다 데이터셋 섞기
)

# Trainer 설정
trainer = Seq2SeqTrainer(
    model=lora_model,
    args=training_args,
    tokenizer=tokenizer,
    data_collator=data_collator,
    train_dataset=dataset,
)

# 학습 시작
trainer.train()

# LoRA adapter 저장
lora_model.save_pretrained(output_dir)
print(f"✅ LoRA 학습 완료 및 저장: {output_dir}")