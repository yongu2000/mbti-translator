import json
import os
import re
import itertools
from typing import List, Dict

# 예시: "샌드위치" → ["김밥", "토스트", "컵라면"]
REPLACEMENT_WORDS = {
    "밥" : ["아침", "점심", "저녁"],
    "샌드위치": ["김밥", "토스트", "컵라면"],

}

MBTI_FIELDS = [
    "intj", "intp", "entj", "entp", "infj", "infp", "enfj", "enfp",
    "istj", "istp", "estj", "estp", "isfj", "isfp", "esfj", "esfp"
]

def replace_word_in_text(text: str, old: str, new: str) -> str:
    return re.sub(old, new, text)

def replace_keyword_all_fields(item: Dict, keyword: str, replacements: List[str]) -> List[Dict]:
    """original 및 16개 MBTI 응답 전체를 대체어로 치환"""
    augmented_items = []

    for new_word in replacements:
        new_item = {}
        # original 변경
        new_item["original"] = replace_word_in_text(item["original"], keyword, new_word)
        # MBTI 응답 전체 변경
        for mbti in MBTI_FIELDS:
            new_item[mbti] = replace_word_in_text(item[mbti], keyword, new_word)
        augmented_items.append(new_item)

    return augmented_items

def augment_entire_record(input_path: str, output_path: str):
    with open(input_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]

    augmented = []
    for item in data:
        augmented.append(item)  # 원본 포함

        for keyword, replacements in REPLACEMENT_WORDS.items():
            if keyword not in item["original"]:
                continue  # keyword가 없으면 skip
            augmented += replace_keyword_all_fields(item, keyword, replacements)

    with open(output_path, 'w', encoding='utf-8') as f:
        for item in augmented:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"원본 샘플 수: {len(data)}")
    print(f"총 증강 결과 수: {len(augmented)}")
    print(f"총 추가된 샘플 수: {len(augmented) - len(data)}")

if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    
    input_file = os.path.join(project_root, "model", "data", "gpt_produced", "mbti_message.jsonl")
    output_file = os.path.join(project_root, "model", "data", "gpt_produced", "mbti_message_replace_augmented.jsonl")

    augment_entire_record(input_file, output_file)