import json
import os
from typing import List, Dict
import re

def has_jongseong(word: str) -> bool:
    """한글 단어의 받침 유무를 확인"""
    # 한글 유니코드 범위: AC00-D7AF
    # 받침이 있는 경우: True, 없는 경우: False
    return bool(re.search(r'[가-힣]', word)) and (ord(word[-1]) - 0xAC00) % 28 > 0

def replace_word(text: str, old_word: str, new_word: str) -> str:
    """텍스트에서 특정 단어를 다른 단어로 대치하고, 받침 유무에 따라 조사 처리"""
    # "은/는" 조사가 붙은 패턴 찾기
    pattern = f"{old_word}(은|는)"
    if re.search(pattern, text):
        # 받침 유무에 따라 적절한 조사 선택
        particle = "은" if has_jongseong(new_word) else "는"
        return re.sub(pattern, f"{new_word}{particle}", text)
    return text.replace(old_word, new_word)

def get_replacement_words(input_file: str) -> List[str]:
    """입력 파일에 따라 적절한 대치 단어 목록 반환"""
    # 공통 단어 (긍정/부정 모두에 사용 가능)
    common_words = [
        # 음식
        "커피", "아이스크림", "치킨", "과일", 
        # 활동
        "운동", "책 읽는 거", "여행", "게임",
        # 동물
        "강아지", "고양이",
        # 미디어/엔터테인먼트
        "드라마보는 거", "영화보는 거",
        # 상황/장소
        "사람 많은 곳", "혼밥"
    ]
    
    # 긍정적 단어
    positive_words = [
        # # 동물
        # "토끼", "앵무새", 
        # # 활동
        # "데이트하는 거", "보드게임", "캠핑",
        # # 상황/장소
        # "공원", "한강", "노을", "벚꽃길", "야경", "숲길"
    ]
    
    # 부정적 단어
    negative_words = [
        # # 음식
        # "브로콜리", "생선", "오징어", "해산물", "김치", "된장찌개", "청국장",
        # # 활동
        # "공부", "청소", "빨래", "설거지", "쓰레기 분리수거",
        # # 미디어/엔터테인먼트
        # "뉴스", "다큐멘터리", "교육 프로그램",
        # # 상황/장소
        # "소개팅", "회식"
    ]
    
    # 파일명에 따라 적절한 단어 목록 선택
    if "positive" in input_file:
        return common_words + positive_words
    elif "negative" in input_file:
        return common_words + negative_words
    else:
        return common_words

def augment_dataset(input_file: str, output_file: str):
    """데이터셋 증강 실행"""
    # 입력 파일에 따라 대치할 단어 목록 가져오기
    replacement_words = get_replacement_words(input_file)
    
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
                    print(f"JSON 파싱 에러 발생: {e}")
                    print(f"문제가 있는 라인: {line}")
                    continue
    print(f"총 {len(data)}개의 데이터 로드 완료")
    
    # 증강된 데이터 저장
    augmented_data = []
    augmented_count = 0
    
    print("\n데이터 증강 시작...")
    for idx, item in enumerate(data, 1):
        # 진행률 표시
        if idx % 10 == 0:  # 10개마다 진행률 표시
            print(f"진행률: {idx}/{len(data)} ({idx/len(data)*100:.1f}%)")
        
        # 원본 데이터 추가
        augmented_data.append(item)
        
        # 각 대치 단어에 대해 새로운 데이터 생성
        for new_word in replacement_words:
            new_item = item.copy()
            converted = False
            
            # original 메시지 변환
            if "스파게티" in new_item["original"]:
                new_item["original"] = replace_word(new_item["original"], "스파게티", new_word)
                converted = True
            
            # MBTI 메시지들 변환
            mbti_types = ["intj", "intp", "entj", "entp", "infj", "infp", "enfj", "enfp",
                          "istj", "istp", "estj", "estp", "isfj", "isfp", "esfj", "esfp"]
            
            for mbti in mbti_types:
                if "스파게티" in new_item[mbti]:
                    new_item[mbti] = replace_word(new_item[mbti], "스파게티", new_word)
                    converted = True
            
            # 변환이 하나라도 있었다면 추가
            if converted:
                augmented_count += 1
                augmented_data.append(new_item)
    
    print("\n증강된 데이터 저장 중...")
    # 증강된 데이터 저장
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in augmented_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print("\n=== 증강 완료 ===")
    print(f"원본 데이터: {len(data)}개")
    print(f"대치 단어 수: {len(replacement_words)}개")
    print(f"증강 후 데이터: {len(augmented_data)}개")
    print(f"증강 비율: {len(augmented_data)/len(data):.1f}배")
    print("=" * 20)

if __name__ == "__main__":
    # 프로젝트 루트 디렉토리 경로 설정
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    
    # 긍정/부정 데이터 각각 처리
    input_files = [
        ("mbti_common_typeA_positive copy.jsonl", "mbti_common_typeA_positive_augmented_less.jsonl"),
        ("mbti_common_typeA_negative.jsonl", "mbti_common_typeA_negative_augmented_less.jsonl")
    ]
    
    for input_file, output_file in input_files:
        input_path = os.path.join(project_root, "model", "data", "gpt_produced", input_file)
        output_path = os.path.join(project_root, "model", "data", "gpt_produced", output_file)
        
        print(f"\n=== {input_file} 처리 중 ===")
        print(f"입력 파일 경로: {input_path}")
        print(f"출력 파일 경로: {output_path}")
        
        augment_dataset(input_path, output_path)
