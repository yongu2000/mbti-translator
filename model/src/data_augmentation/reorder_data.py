import json
import os

def reorder_data(input_file: str, output_file: str):
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
    
    # 데이터 재정렬
    print("데이터 재정렬 중...")
    original_data = data[::2]  # 홀수번째 데이터 (1, 3, 5, ...)
    augmented_data = data[1::2]  # 짝수번째 데이터 (2, 4, 6, ...)
    
    reordered_data = original_data + augmented_data
    
    print(f"원본 데이터 수: {len(original_data)}")
    print(f"증강 데이터 수: {len(augmented_data)}")
    
    # 재정렬된 데이터 저장
    print("재정렬된 데이터 저장 중...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in reordered_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print("저장 완료!")

if __name__ == "__main__":
    # 프로젝트 루트 디렉토리 경로 설정
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_dir))))
    
    # 입력/출력 파일 경로 설정
    input_file = os.path.join(project_root, "model", "data", "gpt_produced", "mbti_message_honorific_augmented.jsonl")
    output_file = os.path.join(project_root, "model", "data", "gpt_produced", "mbti_message_honorific_augmented_reordered.jsonl")
    
    reorder_data(input_file, output_file) 