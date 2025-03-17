import os
import torch
import logging
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from model.model_config import ModelConfig
from model.dataset import MBTIStyleDataset
from model.mbti_style_model import MBTIStyleTransformer
import time

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def train(config: ModelConfig):
    logger = logging.getLogger(__name__)
    
    # GPU 설정 및 상세 정보 출력
    logger.info("=== 학습 환경 설정 ===")
    if not torch.cuda.is_available():
        raise RuntimeError(
            "\nGPU를 찾을 수 없습니다! "
            "\n이 모델은 GPU 학습이 필수적입니다. "
            "\nGPU가 있는지 확인하고 CUDA가 올바르게 설치되어 있는지 확인해주세요."
        )
    
    device = torch.device('cuda')
    logger.info(f"GPU 사용: {torch.cuda.get_device_name(0)}")
    logger.info(f"GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    torch.cuda.empty_cache()  # GPU 캐시 정리
    
    # CUDA 메모리 상태 출력
    logger.info("\n=== CUDA 메모리 상태 ===")
    logger.info(f"할당된 메모리: {torch.cuda.memory_allocated(0) / 1024**3:.1f}GB")
    logger.info(f"캐시된 메모리: {torch.cuda.memory_reserved(0) / 1024**3:.1f}GB")
    
    # 토크나이저 및 특수 토큰 설정
    logger.info("\n=== 모델 설정 ===")
    logger.info(f"기본 모델: {config.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    special_tokens = config.get_special_tokens()
    logger.info(f"특수 토큰 개수: {len(special_tokens)}")
    
    # 데이터셋 및 데이터로더 설정
    logger.info("\n=== 데이터 로딩 ===")
    dataset = MBTIStyleDataset(
        data_path=config.train_data_path,
        tokenizer=tokenizer,
        max_length=config.max_length,
        mbti_token_prefix=config.mbti_token_prefix,
        mbti_token_suffix=config.mbti_token_suffix
    )
    logger.info(f"전체 데이터 크기: {len(dataset):,}개")
    logger.info(f"배치 크기: {config.batch_size}")
    logger.info(f"총 배치 수: {len(dataset) // config.batch_size:,}")
    
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True
    )
    
    # 모델 설정
    model = MBTIStyleTransformer(
        model_name=config.model_name,
        tokenizer=tokenizer,
        special_tokens=special_tokens
    )
    model.to(device)
    
    # GPU 사용 확인
    logger.info(f"\n모델이 현재 사용 중인 디바이스: {next(model.parameters()).device}")
    
    # 옵티마이저 및 스케줄러 설정
    logger.info("\n=== 학습 설정 ===")
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    total_steps = len(dataloader) * config.num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=total_steps
    )
    logger.info(f"학습률: {config.learning_rate}")
    logger.info(f"총 에폭: {config.num_epochs}")
    logger.info(f"총 학습 스텝: {total_steps:,}")
    logger.info(f"Warmup 스텝: {config.warmup_steps:,}")
    
    # 학습 루프
    logger.info("\n=== 학습 시작 ===")
    global_step = 0
    start_time = time.time()
    
    for epoch in range(config.num_epochs):
        epoch_start_time = time.time()
        model.train()
        total_loss = 0
        
        for batch_idx, batch in enumerate(dataloader):
            # 데이터를 GPU로 이동
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss
            total_loss += loss.item()
            
            # Backward pass
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            global_step += 1
            
            # 로깅
            if global_step % config.logging_steps == 0:
                avg_loss = total_loss / config.logging_steps
                elapsed_time = time.time() - start_time
                steps_per_sec = global_step / elapsed_time
                remaining_steps = total_steps - global_step
                eta_seconds = remaining_steps / steps_per_sec
                
                logger.info(
                    f"에폭: {epoch+1}/{config.num_epochs} | "
                    f"배치: {batch_idx+1}/{len(dataloader)} | "
                    f"스텝: {global_step:,}/{total_steps:,} | "
                    f"손실: {avg_loss:.4f} | "
                    f"속도: {steps_per_sec:.1f}it/s | "
                    f"남은 시간: {eta_seconds/3600:.1f}시간"
                )
                total_loss = 0
            
            # 모델 저장
            if global_step % config.save_steps == 0:
                output_dir = os.path.join(config.output_dir, f"checkpoint-{global_step}")
                os.makedirs(output_dir, exist_ok=True)
                model.save_pretrained(output_dir)
                logger.info(f"체크포인트 저장: {output_dir}")
        
        # 에폭 종료 시 로깅
        epoch_time = time.time() - epoch_start_time
        logger.info(
            f"\n=== 에폭 {epoch+1} 완료 ===\n"
            f"소요 시간: {epoch_time/60:.1f}분"
        )
    
    # 최종 모델 저장
    final_output_dir = os.path.join(config.output_dir, "final")
    os.makedirs(final_output_dir, exist_ok=True)
    model.save_pretrained(final_output_dir)
    logger.info(f"\n=== 학습 완료 ===\n최종 모델 저장: {final_output_dir}")
    
    # 총 학습 시간 출력
    total_time = time.time() - start_time
    logger.info(f"총 학습 시간: {total_time/3600:.1f}시간")

def main():
    setup_logging()
    config = ModelConfig()
    train(config)

if __name__ == "__main__":
    main() 