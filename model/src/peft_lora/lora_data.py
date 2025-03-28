import json
import random
import pandas as pd
import os

# 대상 MBTI (LoRA 학습용으로 만들 MBTI)
target_mbti = "istp"

# 사용할 최대 input source 수 (original 포함)
max_input_sources = 6

# 파일 경로
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
input_path = os.path.join(project_root, "model", "data", "mbti_message_final.jsonl")
output_dir = os.path.join(project_root, "model", "data", "lora")
output_path = os.path.join(output_dir, f"{target_mbti}_message.jsonl")

# 출력 디렉토리 생성
os.makedirs(output_dir, exist_ok=True)

# MBTI 리스트
mbti_types = [
    "intj", "intp", "entj", "entp",
    "infj", "infp", "enfj", "enfp",
    "istj", "istp", "estj", "estp",
    "isfj", "isfp", "esfj", "esfp"
]

# 데이터 로딩
rows = []
with open(input_path, "r", encoding="utf-8") as f:
    for line in f:
        rows.append(json.loads(line))

# 변환된 input-output 샘플 만들기
samples = []
for row in rows:
    if target_mbti not in row or not row[target_mbti]:
        continue

    # source 후보: original + 나머지 MBTI (단, target은 제외)
    candidates = ["original"] + [mbti for mbti in mbti_types if mbti != target_mbti and mbti in row and row[mbti]]

    # 랜덤으로 최대 max_input_sources개 선택
    selected_sources = random.sample(candidates, min(len(candidates), max_input_sources))

    for source in selected_sources:
        input_text = row[source] if source != "original" else row["original"]
        output_text = row[target_mbti]

        samples.append({
            "input": f"{target_mbti} 말투로 변환: {input_text}",
            "output": output_text
        })

# 데이터 섞기
random.shuffle(samples)

# 저장
with open(output_path, "w", encoding="utf-8") as f:
    for item in samples:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"✅ 저장 완료: {output_path}")
print(f"총 {len(samples)}개 샘플 생성됨.")