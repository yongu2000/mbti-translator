import json
import os
from typing import List, Dict
import re

def has_jongseong(word: str) -> bool:
    """한글 단어의 받침 유무를 확인"""
    return bool(re.search(r'[가-힣]', word)) and (ord(word[-1]) - 0xAC00) % 28 > 0

def replace_word(text: str, old_word: str, new_word: str) -> str:
    """텍스트에서 특정 단어를 다른 단어로 대치하고, 받침 유무에 따라 조사 처리"""
    # "은/는" 조사가 붙은 패턴 찾기
    pattern = f"{old_word}(은|는)"
    if re.search(pattern, text):
        # 받침 유무에 따라 적절한 조사 선택
        particle = "은" if has_jongseong(new_word) else "는"
        return re.sub(pattern, f"{new_word}{particle}", text)
    
    # "이" + 단어 패턴 찾기
    pattern = f"{old_word}이"
    if re.search(pattern, text):
        # 받침이 없는 경우 "이" 제거
        if not has_jongseong(new_word):
            return re.sub(pattern, new_word, text)
        else:
            return re.sub(pattern, f"{new_word}이", text)
    
    return text.replace(old_word, new_word)

def get_replacement_words(input_file: str) -> List[str]:
    """입력 파일에 따라 적절한 대치 단어 목록 반환"""
    # 공통 단어 (모든 감정 상태에 사용 가능)
    common_words = [

    ]
    
    # 화남 관련 단어
    anger_words = [
        "사람", 
        #  "영화","결과", "소식", , "댓글", "정책", "부조리", "대우", "불공평함", "장면" 
    ]
    
    # 웃김 관련 단어
    funny_words = [
        "영화",
        # "영화", "결과", "만화", "소설", "표정", "리액션", "연기", "말투", "댓글", "단어선택", "과몰입", "방송", "장면"
    ]
    
    # 슬픔 관련 단어
    sadness_words = [
        "노래 가사",
        # "영화", "결과", "만화", "음악", "소식", "뉴스", "소설", "상실감", "허전함", "외로움", "쓸쓸함", "무력감", "시간", "노래 가사", "장면"
    ]
    
    # 기쁨 관련 단어
    joy_words = [
        "선물", 
        # "결과", "소식", "순간", "느낌", "기분", "타이밍", "반응", "뿌듯함", "선물", "응원", "댓글", "말 한마디"
    ]
    
    # 파일명에 따라 적절한 단어 목록 선택
    if "typeB" in input_file:
        if "1" in input_file:
            return common_words + anger_words
        elif "2" in input_file:
            return common_words + funny_words
        elif "3" in input_file:
            return common_words + sadness_words
        elif "4" in input_file:
            return common_words + joy_words
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
            if "상황" in new_item["original"]:
                new_item["original"] = replace_word(new_item["original"], "상황", new_word)
                converted = True
            
            # MBTI 메시지들 변환
            mbti_types = ["intj", "intp", "entj", "entp", "infj", "infp", "enfj", "enfp",
                          "istj", "istp", "estj", "estp", "isfj", "isfp", "esfj", "esfp"]
            
            for mbti in mbti_types:
                if "상황" in new_item[mbti]:
                    new_item[mbti] = replace_word(new_item[mbti], "상황", new_word)
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
    
    # 감정 상태별 데이터 처리
    input_files = [
        ("mbti_common_typeB_augmented1.jsonl", "mbti_common_typeB_word_augmented1.jsonl"),
        ("mbti_common_typeB_augmented2.jsonl", "mbti_common_typeB_word_augmented2.jsonl"),
        ("mbti_common_typeB_augmented3.jsonl", "mbti_common_typeB_word_augmented3.jsonl"),
        ("mbti_common_typeB_augmented4.jsonl", "mbti_common_typeB_word_augmented4.jsonl"),
    ]
    
    for input_file, output_file in input_files:
        input_path = os.path.join(project_root, "model", "data", "gpt_produced", input_file)
        output_path = os.path.join(project_root, "model", "data", "gpt_produced", output_file)
        
        print(f"\n=== {input_file} 처리 중 ===")
        print(f"입력 파일 경로: {input_path}")
        print(f"출력 파일 경로: {output_path}")
        
        augment_dataset(input_path, output_path)
