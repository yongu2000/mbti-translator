import json
import logging
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def sample_top_conversations(input_file: Path, existing_outputs: set, n_samples: int = 100) -> List[Dict]:
    """JSONL 파일에서 korean_ratio가 높은 상위 N개의 대화를 추출 (중복 제거)"""
    try:
        conversations = []
        duplicates = 0
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                conv = json.loads(line.strip())
                # 중복 체크
                if conv['output'] not in existing_outputs:
                    conversations.append(conv)
                    existing_outputs.add(conv['output'])
                else:
                    duplicates += 1
        
        # korean_ratio 기준으로 정렬
        sorted_conversations = sorted(conversations, 
                                   key=lambda x: x.get('korean_ratio', 0), 
                                   reverse=True)
        
        # 상위 N개 선택
        sampled = sorted_conversations[:n_samples]
        
        logger.info(f"파일 {input_file.name} 처리:")
        logger.info(f"- 전체 대화: {len(conversations) + duplicates}개")
        logger.info(f"- 중복 제거: {duplicates}개")
        logger.info(f"- 최종 샘플: {len(sampled)}개")
        if sampled:
            logger.info(f"- 한글 비율 범위: {sampled[-1]['korean_ratio']:.1f}% ~ {sampled[0]['korean_ratio']:.1f}%")
        
        return sampled
        
    except Exception as e:
        logger.error(f"파일 처리 중 오류 발생: {input_file}, {str(e)}")
        return []

def main():
    input_dir = Path("data/processed_style")
    output_dir = Path("data/sampled_style")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 모든 MBTI 유형
    mbti_types = [
        "ENFJ", "ENFP", "ENTJ", "ENTP",
        "ESFJ", "ESFP", "ESTJ", "ESTP",
        "INFJ", "INFP", "INTJ", "INTP",
        "ISFJ", "ISFP", "ISTJ", "ISTP"
    ]
    
    # 중복 문장 체크를 위한 set
    existing_outputs = set()
    
    # MBTI 유형별 통계
    stats = defaultdict(int)
    
    total_samples = 0
    for mbti_type in tqdm(mbti_types, desc="MBTI 유형 처리 중"):
        input_file = input_dir / f"{mbti_type.lower()}_conversations.jsonl"
        if not input_file.exists():
            logger.warning(f"파일을 찾을 수 없음: {input_file}")
            continue
            
        # 상위 300개 샘플 추출 (중복 제거)
        sampled_conversations = sample_top_conversations(input_file, existing_outputs, n_samples=300)
        if not sampled_conversations:
            continue
            
        # 결과 저장
        output_file = output_dir / f"{mbti_type.lower()}_conversations.jsonl"
        with open(output_file, 'w', encoding='utf-8') as f:
            for conv in sampled_conversations:
                f.write(json.dumps(conv, ensure_ascii=False) + '\n')
                
        stats[mbti_type] = len(sampled_conversations)
        total_samples += len(sampled_conversations)
    
    # 최종 통계 출력
    logger.info("\n=== 최종 통계 ===")
    logger.info(f"전체 샘플 수: {total_samples}개")
    for mbti_type in mbti_types:
        logger.info(f"{mbti_type}: {stats[mbti_type]}개")
    logger.info(f"결과 저장 위치: {output_dir}")

if __name__ == "__main__":
    main() 