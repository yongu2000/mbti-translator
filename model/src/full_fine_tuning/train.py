# 필요한 라이브러리
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import pandas as pd
import json
import random
import os

# 프로젝트 루트 디렉토리 경로 설정
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# 모델 및 토크나이저
model_name = "gogamza/kobart-base-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# 데이터셋 로딩
def load_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return pd.DataFrame(data)

# MBTI 데이터 로딩
input_file = os.path.join(project_root, "model", "data", "mbti_message_final.jsonl")
print(f"입력 파일 경로: {input_file}")
df = load_jsonl(input_file)

# Dataset 클래스 정의
class MBTIDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=64, max_pairs_per_row=5):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mbti_types = [
            "intj", "intp", "entj", "entp", "infj", "infp", "enfj", "enfp",
            "istj", "istp", "estj", "estp", "isfj", "isfp", "esfj", "esfp"
        ]
        self.eos_token = tokenizer.eos_token

        # MBTI 타입별 카운터 초기화
        mbti_counts = {mbti: 0 for mbti in self.mbti_types}
        
        # 데이터프레임을 랜덤으로 섞기
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        for _, row in df.iterrows():
            sources = ["original"] + [mbti for mbti in self.mbti_types if mbti in row and pd.notna(row[mbti])]
            targets = [mbti for mbti in self.mbti_types if mbti in row and pd.notna(row[mbti])]

            # MBTI 타입별로 쌍을 분리
            mbti_pairs = {mbti: [] for mbti in self.mbti_types}
            
            for from_mbti in sources:
                for to_mbti in targets:
                    if from_mbti == to_mbti:
                        continue
                    input_text = row[from_mbti] if from_mbti != "original" else row["original"]
                    encoder_text = f"{to_mbti} 말투로 변환: {input_text}"
                    decoder_text = row[to_mbti] + self.eos_token
                    mbti_pairs[to_mbti].append((encoder_text, decoder_text))

            # 각 MBTI 타입별로 균형있게 샘플링
            for mbti in self.mbti_types:
                if mbti_pairs[mbti]:
                    # 현재 MBTI 타입의 카운트가 가장 적은 경우에만 샘플링
                    if mbti_counts[mbti] == min(mbti_counts.values()):
                        sampled_pairs = random.sample(mbti_pairs[mbti], 
                                                    min(len(mbti_pairs[mbti]), max_pairs_per_row))
                        self.samples.extend(sampled_pairs)
                        mbti_counts[mbti] += len(sampled_pairs)

        # 샘플을 랜덤으로 섞기
        random.shuffle(self.samples)

        # 최종 데이터셋 크기 출력
        print("\n=== 데이터셋 통계 ===")
        print(f"총 샘플 수: {len(self.samples)}")
        print("\nMBTI 타입별 샘플 수:")
        for mbti, count in mbti_counts.items():
            print(f"{mbti}: {count}")
        print("=" * 20)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        encoder_text, decoder_text = self.samples[idx]
        model_inputs = self.tokenizer(encoder_text, max_length=self.max_length, truncation=True, return_tensors="pt")
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(decoder_text, max_length=self.max_length, truncation=True, return_tensors="pt")

        model_inputs = {k: v.squeeze() for k, v in model_inputs.items() if k != 'token_type_ids'}
        model_inputs['labels'] = labels['input_ids'].squeeze()
        return model_inputs

# train/test split
df_train, df_test = train_test_split(df, test_size=0.1, random_state=42)
train_dataset = MBTIDataset(df_train, tokenizer)
test_dataset = MBTIDataset(df_test, tokenizer)

# trainer 설정
training_args = Seq2SeqTrainingArguments(
    output_dir="./mbti-style-transfer-model",
    num_train_epochs=5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    evaluation_strategy="steps",
    eval_steps=500,
    save_steps=1000,
    warmup_steps=300,
    prediction_loss_only=True,
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=100,
    # 추가 설정
    gradient_accumulation_steps=4,  # 배치 크기 효과적으로 증가
    fp16=True,  # 혼합 정밀도 학습 활성화
    gradient_checkpointing=True,  # 메모리 효율성 향상
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    data_collator=DataCollatorForSeq2Seq(tokenizer, model),
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

# 학습 시작
trainer.train()

# 모델 저장
trainer.save_model("./mbti-style-transfer-model")