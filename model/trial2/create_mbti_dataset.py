import pandas as pd
import json
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from text_cleaner import TextCleaner

def load_and_process_tsv(file_path: str) -> dict:
    """TSV 파일을 로드하고 MBTI 유형별로 텍스트를 분류"""
    df = pd.read_csv(file_path, sep='\t')
    mbti_texts = defaultdict(list)
    cleaner = TextCleaner()
    
    # 질문자의 텍스트와 MBTI 처리
    for _, row in tqdm(df.iterrows(), total=len(df), desc="TSV 파일 처리 중"):
        # 질문자 MBTI와 텍스트가 있는 경우만 처리
        if pd.notna(row['q_mbti']) and pd.notna(row['question']):
            texts = str(row['question']).split('[SEP]')
            texts = [text.strip() for text in texts if text.strip()]
            mbti = str(row['q_mbti']).upper()
            
            # 텍스트 정제
            cleaned_texts = []
            for text in texts:
                cleaned_text = cleaner.clean_text(text)
                if cleaner.is_valid_sentence(cleaned_text):
                    cleaned_texts.append(cleaned_text)
            
            mbti_texts[mbti].extend(cleaned_texts)
        
        # 답변자 MBTI와 텍스트가 있는 경우만 처리
        if pd.notna(row['a_mbti']) and pd.notna(row['answer']):
            texts = str(row['answer']).split('[SEP]')
            texts = [text.strip() for text in texts if text.strip()]
            mbti = str(row['a_mbti']).upper()
            
            # 텍스트 정제
            cleaned_texts = []
            for text in texts:
                cleaned_text = cleaner.clean_text(text)
                if cleaner.is_valid_sentence(cleaned_text):
                    cleaned_texts.append(cleaned_text)
            
            mbti_texts[mbti].extend(cleaned_texts)
    
    return mbti_texts

def create_dataset():
    # 입력 파일 경로
    input_dir = Path("../data/raw")
    qna_path = input_dir / "qna_cleaned.tsv"
    multiple_qna_path = input_dir / "multiple_qna_cleaned.tsv"
    
    # 데이터 로드 및 처리
    print("데이터 로드 중...")
    mbti_texts = defaultdict(list)
    cleaner = TextCleaner()
    
    if qna_path.exists():
        print(f"Processing {qna_path}")
        qna_data = load_and_process_tsv(qna_path)
        for mbti, texts in qna_data.items():
            mbti_texts[mbti].extend(texts)
    
    if multiple_qna_path.exists():
        print(f"Processing {multiple_qna_path}")
        multiple_data = load_and_process_tsv(multiple_qna_path)
        for mbti, texts in multiple_data.items():
            mbti_texts[mbti].extend(texts)
    
    # MBTI 특성에 따른 필터링
    print("\nMBTI 특성에 따른 필터링 중...")
    for mbti in tqdm(mbti_texts, desc="MBTI 특성 필터링"):
        mbti_texts[mbti] = cleaner.filter_by_mbti_characteristics(mbti_texts[mbti], mbti)
    
    # 중복 제거 및 유사 문장 처리
    print("\n유사 문장 제거 중...")
    for mbti in tqdm(mbti_texts, desc="유사 문장 제거"):
        mbti_texts[mbti] = cleaner.remove_similar_sentences(mbti_texts[mbti])
    
    # 출력 디렉토리 생성
    output_dir = Path("../data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 데이터셋 저장
    output_path = output_dir / "mbti_speech_patterns_cleaned.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(mbti_texts, f, ensure_ascii=False, indent=2)
    
    print(f"\n데이터셋이 생성되었습니다: {output_path}")
    print("\nMBTI 유형별 문장 수:")
    for mbti, texts in sorted(mbti_texts.items()):
        print(f"{mbti}: {len(texts)}개의 문장")

if __name__ == "__main__":
    create_dataset() 