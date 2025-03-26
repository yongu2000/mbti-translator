import json
import os
from typing import List, Dict
from .kdictionary import Changer
from .utils import Utils

def augment_dataset(input_file: str, output_file: str):
    print(f"입력 파일 경로: {input_file}")
    print(f"출력 파일 경로: {output_file}")
    
    # 데이터 로드
    print("데이터 로드 중...")
    data = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('//'):  # 빈 줄과 주석 건너뛰기
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"JSON 파싱 오류 발생: {e}")
                    print(f"문제가 있는 라인: {line}")
                    continue
    
    print(f"총 {len(data)}개의 데이터 로드됨")
    
    # 변환기 초기화
    print("변환기 초기화 중...")
    try:
        model = Changer()
        print("변환기 초기화 완료")
    except Exception as e:
        print("변환기 초기화 실패!")
        print("kiwipiepy를 설치해주세요: pip install kiwipiepy")
        return
    
    # 데이터 증강
    print("데이터 증강 시작...")
    augmented_data = []
    converted_count = 0
    
    # 원본 데이터 먼저 추가
    augmented_data.extend(data)
    
    for i, item in enumerate(data):
        if i % 10 == 0:
            print(f"처리 중... {i}/{len(data)} ({i/len(data)*100:.1f}%)")
        
        # 변환된 데이터를 저장할 새로운 아이템
        new_item = item.copy()
        converted = False
        
        # 모든 MBTI 메시지에 대해 반말 변환 시도
        for mbti_type in ['original', 'intj', 'intp', 'entj', 'entp', 'infj', 'infp', 'enfj', 'enfp', 
                         'istj', 'istp', 'estj', 'estp', 'isfj', 'isfp', 'esfj', 'esfp']:
            if mbti_type in item and item[mbti_type]:
                text = item[mbti_type]
                try:
                    # 존댓말을 반말로 변환
                    converted_text = model.dechanger(text)
                    if converted_text != text:
                        new_item[mbti_type] = converted_text
                        converted = True
                except Exception as e:
                    print(f"변환 중 오류 발생: {e}")
                    continue
        
        # 변환이 하나라도 있었다면 추가
        if converted:
            converted_count += 1
            augmented_data.append(new_item)
    
    print(f"\n데이터 증강 완료")
    print(f"원본 데이터 수: {len(data)}")
    print(f"변환된 데이터 수: {converted_count}")
    print(f"증강 비율: {converted_count/len(data)*100:.1f}%")
    
    # 증강된 데이터 저장
    print("증강된 데이터 저장 중...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in augmented_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print("저장 완료!")

if __name__ == "__main__":
    # 프로젝트 루트 디렉토리 경로 설정
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_dir))))
    
    # 입력/출력 파일 경로 설정
    input_file = os.path.join(project_root, "model", "data", "gpt_produced", "mbti_message.jsonl")
    output_file = os.path.join(project_root, "model", "data", "gpt_produced", "mbti_message_ban_augmented.jsonl")
    
    print(f"입력 파일 경로: {input_file}")
    print(f"출력 파일 경로: {output_file}")
    
    augment_dataset(input_file, output_file) 