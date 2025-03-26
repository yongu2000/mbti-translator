import json
import os
from typing import List, Dict, Tuple

def find_honorific_patterns(text: str) -> List[Tuple[str, int]]:
    """문장에서 존댓말 패턴을 찾아서 반환"""
    patterns = []
    # 기본 존댓말 어미
    honorific_endings = ['죠', '요', '니다']
    
    # 문장 끝에 있는 존댓말 어미 찾기
    for ending in honorific_endings:
        if text.endswith(ending):
            patterns.append((ending, len(text) - len(ending)))
            break
    
    # 문장 끝에 있는 존댓말 어미 + 문장부호
    for ending in honorific_endings:
        for punct in ['!', '.', '?', '~', ',', ';']:
            if text.endswith(ending + punct):
                patterns.append((ending + punct, len(text) - len(ending) - 1))
                break
    
    return patterns

def analyze_dataset(input_file: str):
    """데이터셋 분석"""
    print("데이터 로드 중...")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    print(f"총 {len(data)}개의 데이터 로드 완료")
    
    print("\n존댓말 패턴 분석 시작...")
    honorific_count = 0
    
    for idx, item in enumerate(data, 1):
        text = item["original"]
        patterns = find_honorific_patterns(text)
        
        if patterns:
            honorific_count += 1
            print(f"\n[{idx}번째 문장]")
            print(f"원문: {text}")
            for pattern, pos in patterns:
                print(f"- 패턴 '{pattern}' 발견 (위치: {pos})")
    
    print("\n=== 분석 완료 ===")
    print(f"전체 문장 수: {len(data)}")
    print(f"존댓말 패턴이 있는 문장 수: {honorific_count}")
    print(f"존댓말 비율: {honorific_count/len(data)*100:.1f}%")
    print("=" * 20)

if __name__ == "__main__":
    # 프로젝트 루트 디렉토리 경로 설정
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    
    # 입력 파일 경로 설정
    input_file = os.path.join(project_root, "model", "data", "gpt_produced", "mbti_message.jsonl")
    
    print(f"입력 파일 경로: {input_file}")
    analyze_dataset(input_file)
