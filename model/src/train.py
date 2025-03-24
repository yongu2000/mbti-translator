import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model
)
from datasets import load_dataset
import wandb
from config import Config
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MBTITrainer:
    def __init__(self, mbti_type: str):
        self.mbti_type = mbti_type
        self.model_name = Config.BASE_MODEL
        
        # 모델과 토크나이저 로드
        logger.info(f"모델 로딩 중: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            token=Config.HF_TOKEN
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            token=Config.HF_TOKEN,
            device_map="auto",
            load_in_4bit=True,
            torch_dtype=torch.float16
        )
        
        # LoRA 설정
        self.lora_config = LoraConfig(
            r=Config.LORA_R,
            lora_alpha=Config.LORA_ALPHA,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=Config.LORA_DROPOUT,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        # 모델 준비
        self.model = prepare_model_for_kbit_training(self.model)
        self.model = get_peft_model(self.model, self.lora_config)
        
        # 데이터셋 로드
        self.dataset = load_dataset(
            "json",
            data_files=f"data/processed/{self.mbti_type.lower()}_conversations.jsonl"
        )
        
        # 데이터 전처리
        self.processed_dataset = self.dataset.map(
            self._preprocess_function,
            batched=True,
            remove_columns=self.dataset["train"].column_names
        )
        
    def _preprocess_function(self, examples):
        # 프롬프트 템플릿
        prompt_template = """다음은 {mbti} 유형의 대화입니다.
입력: {input}
답변: {output}"""
        
        # 데이터 전처리
        texts = []
        for input_text, output_text in zip(examples["input"], examples["output"]):
            text = prompt_template.format(
                mbti=self.mbti_type,
                input=input_text,
                output=output_text
            )
            texts.append(text)
            
        # 토크나이징
        tokenized = self.tokenizer(
            texts,
            truncation=True,
            max_length=Config.MAX_LENGTH,
            padding="max_length"
        )
        
        return tokenized
        
    def train(self):
        # 학습 설정
        training_args = TrainingArguments(
            output_dir=f"models/lora/{self.mbti_type.lower()}",
            num_train_epochs=Config.NUM_EPOCHS,
            per_device_train_batch_size=Config.BATCH_SIZE,
            gradient_accumulation_steps=4,
            learning_rate=Config.LEARNING_RATE,
            fp16=True,
            logging_steps=10,
            save_strategy="epoch",
            report_to="wandb"
        )
        
        # 데이터 콜레이터
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        # 트레이너 초기화
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.processed_dataset["train"],
            data_collator=data_collator
        )
        
        # 학습 시작
        logger.info(f"{self.mbti_type} 학습 시작")
        trainer.train()
        
        # 모델 저장
        trainer.save_model()
        logger.info(f"{self.mbti_type} 학습 완료")

def main():
    # wandb 초기화
    wandb.init(project="mbti-translator")
    
    # 각 MBTI 유형별로 학습
    for mbti in Config.MBTI_TYPES:
        trainer = MBTITrainer(mbti)
        trainer.train()
    
    wandb.finish()

if __name__ == "__main__":
    main() 