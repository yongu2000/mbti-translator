import json
import random
from collections import defaultdict
from pathlib import Path

def create_sampled_dataset(input_path: str, output_path: str, samples_per_type: int = 10):
    """
    각 MBTI 타입별로 균등하게 샘플링하여 새로운 데이터셋을 생성합니다.
    
    Args:
        input_path: 원본 데이터셋 경로
        output_path: 샘플링된 데이터셋 저장 경로
        samples_per_type: 각 MBTI 타입당 샘플링할 데이터 수
    """
    # 원본 데이터 로드
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # MBTI 타입별로 데이터 그룹화
    mbti_groups = defaultdict(list)
    for item in data:
        source_mbti = item['source_mbti']
        mbti_groups[source_mbti].append(item)
    
    # 각 타입별로 샘플링
    sampled_data = []
    for mbti_type, items in mbti_groups.items():
        if len(items) > samples_per_type:
            sampled_items = random.sample(items, samples_per_type)
            sampled_data.extend(sampled_items)
        else:
            sampled_data.extend(items)
    
    # 결과 저장
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(sampled_data, f, ensure_ascii=False, indent=2)
    
    # 통계 출력
    print(f"원본 데이터셋 크기: {len(data)}")
    print(f"샘플링된 데이터셋 크기: {len(sampled_data)}")
    print("\nMBTI 타입별 데이터 수:")
    for mbti_type, items in mbti_groups.items():
        print(f"{mbti_type}: {len(items)}")

if __name__ == "__main__":
    # 상대 경로 수정
    base_path = Path(__file__).parent.parent.parent
    input_path = base_path / "data" / "processed" / "style_transfer_pairs.json"
    output_path = base_path / "data" / "processed" / "style_transfer_pairs_test.json"
    create_sampled_dataset(str(input_path), str(output_path), samples_per_type=10) 