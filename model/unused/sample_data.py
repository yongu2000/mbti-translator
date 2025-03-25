import json
from pathlib import Path
import logging
from typing import List, Dict
from config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def sample_top_conversations(input_file: str, output_file: str, top_n: int = 300) -> None:
    """각 MBTI별로 similarity가 높은 상위 N개의 대화만 선택"""
    logger.info(f"{input_file} 처리 중...")
    
    # JSONL 파일 읽기
    conversations = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            conversations.append(json.loads(line))
    
    # similarity 기준으로 정렬
    conversations.sort(key=lambda x: x['similarity'], reverse=True)
    
    # 상위 N개 선택
    selected_conversations = conversations[:top_n]
    
    # 결과 저장
    with open(output_file, 'w', encoding='utf-8') as f:
        for conv in selected_conversations:
            f.write(json.dumps(conv, ensure_ascii=False) + '\n')
    
    logger.info(f"{input_file} 처리 완료: {len(selected_conversations)}개 대화 선택됨")

def main():
    # 입력/출력 디렉토리 설정
    input_dir = Path("data/processed_style")
    output_dir = Path("data/sampled_style")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # MBTI 유형 목록
    mbti_types = [
        "ENFJ", "ENTJ", "ESFJ", "ESFP",
        "ESTJ", "ESTP", "ISFJ", "ISTJ"
    ]
    
    # 각 MBTI 유형별로 처리
    for mbti in mbti_types:
        input_file = input_dir / f"{mbti.lower()}_conversations.jsonl"
        if not input_file.exists():
            logger.warning(f"파일을 찾을 수 없습니다: {input_file}")
            continue
            
        sample_top_conversations(input_file, output_dir)

if __name__ == "__main__":
    main() 