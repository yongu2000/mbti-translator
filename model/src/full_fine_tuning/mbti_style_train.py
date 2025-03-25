import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from torch.optim import AdamW
import json
import logging
from tqdm import tqdm
import os
from config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def collate_fn(batch):
    return {
        "input_ids": torch.stack([x["input_ids"] for x in batch]),
        "attention_mask": torch.stack([x["attention_mask"] for x in batch]),
        "labels": torch.stack([x["labels"] for x in batch])
    }

class MBTIStyleDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=Config.MAX_LENGTH):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        
        # JSONL 파일 읽기
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                original_text = data['original']
                
                # 각 MBTI 유형에 대해 예제 생성
                for mbti_type in Config.MBTI_TYPES:
                    if mbti_type.lower() in data:
                        target_text = data[mbti_type.lower()]
                        
                        # 인코더 입력 생성
                        encoder_text = f"{mbti_type} 말투로 변환:{original_text}"
                        encoder_inputs = self.tokenizer(
                            encoder_text,
                            max_length=self.max_length,
                            padding='max_length',
                            truncation=True,
                            return_tensors='pt'
                        )
                        
                        # 디코더 입력 생성
                        decoder_text = f"{target_text}{self.tokenizer.eos_token}"
                        decoder_inputs = self.tokenizer(
                            decoder_text,
                            max_length=self.max_length,
                            padding='max_length',
                            truncation=True,
                            return_tensors='pt'
                        )
                        
                        labels = decoder_inputs['input_ids'].squeeze()
                        labels[labels == tokenizer.pad_token_id] = -100

                        self.examples.append({
                            'input_ids': encoder_inputs['input_ids'].squeeze(),
                            'attention_mask': encoder_inputs['attention_mask'].squeeze(),
                            'labels': labels,
                            'mbti_type': mbti_type,
                            'original': original_text,
                            'target': target_text
                        })
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]

def train_model(model, train_loader, optimizer, device, num_epochs=Config.NUM_EPOCHS):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            total_loss += loss.item()
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            progress_bar.set_postfix({'loss': total_loss / (progress_bar.n + 1)})
        
        avg_loss = total_loss / len(train_loader)
        logger.info(f'Epoch {epoch+1}, Average Loss: {avg_loss:.4f}')

def main():
    # 설정
    model_name = Config.BASE_MODEL
    jsonl_path = os.path.join(Config.DATA_PATH, 'mbti_message.jsonl')
    output_dir = 'model/checkpoints/mbti_style'
    
    # 디바이스 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')
    
    # 토크나이저와 모델 로드
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    
    # 데이터셋과 데이터로더 생성
    dataset = MBTIStyleDataset(jsonl_path, tokenizer)
    train_loader = DataLoader(
        dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn
    )
    
    # 옵티마이저 설정
    optimizer = AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    
    # 학습 실행
    train_model(model, train_loader, optimizer, device)
    
    # 모델 저장
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info(f'Model saved to {output_dir}')

if __name__ == '__main__':
    main()
