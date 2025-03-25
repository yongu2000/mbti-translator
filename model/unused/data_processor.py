import pandas as pd
import json
from pathlib import Path
from typing import List, Dict, Tuple
from config import Config
import logging
import re
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch

print(torch.cuda.is_available())  # True가 나와야 합니다
print(torch.cuda.get_device_name(0))  # "NVIDIA GeForce RTX 4060 Ti"가 나와야 합니다
# 로깅 설정 추가
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class DataProcessor:
    def __init__(self, output_dir: str = "data/processed"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # KoSentenceBERT 모델 로드
        logging.info("문장 임베딩 모델 로딩 중...")
        self.model = SentenceTransformer('sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens')
        # GPU 사용 설정
        if torch.cuda.is_available():
            self.model = self.model.to('cuda')
            logging.info("GPU 사용 중")
        else:
            logging.info("CPU 사용 중")
        logging.info("모델 로딩 완료")
        
    def _is_noisy(self, text: str) -> bool:
        # URL 패턴
        if re.search(r'(https?://|youtu\.be|youtube\.com|bit\.ly|goo\.gl)', text):
            return True
            
        # 영어 단어/문자 관련
        if re.search(r'[A-Za-z]{5,}', text):  # 긴 영어 단어
            return True
        if len(re.findall(r'[A-Za-z]', text)) > 10:  # 영어 알파벳 10개 이상
            return True
            
        # 숫자+단위 조합
        if re.search(r'\d+(st|nd|rd|th)', text):
            return True
            
        # 반복되는 문자/단어
        if re.search(r'(.)\1{4,}', text):  # 같은 문자 8번 이상 반복
            return True
            
        # 이모지/특수문자 비율
        special_chars = len(re.findall(r'[^\w\s]', text))
        if special_chars / len(text) > 0.3:  # 특수문자 비율 30% 이상
            return True
            
        return False
        
    def _get_language_ratio(self, text: str) -> Tuple[float, float]:
        korean_chars = len(re.findall(r'[가-힣]', text))
        english_chars = len(re.findall(r'[A-Za-z]', text))
        total_chars = korean_chars + english_chars
        
        if total_chars == 0:
            return 0, 0
            
        korean_ratio = korean_chars / total_chars
        english_ratio = english_chars / total_chars
        
        return korean_ratio, english_ratio
        
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        # 문장 임베딩 생성 (GPU 사용)
        with torch.no_grad():  # 추론 시 그래디언트 계산 방지
            embedding1 = self.model.encode([text1], convert_to_tensor=True, device='cuda' if torch.cuda.is_available() else 'cpu')[0]
            embedding2 = self.model.encode([text2], convert_to_tensor=True, device='cuda' if torch.cuda.is_available() else 'cpu')[0]
            
            # 코사인 유사도 계산
            similarity = cosine_similarity(embedding1.cpu().numpy().reshape(1, -1), 
                                        embedding2.cpu().numpy().reshape(1, -1))[0][0]
        return similarity
        
    def process_tsv_files(self, qna_path: str, multiple_qna_path: str):
        logging.info("데이터 처리 시작")
        
        # 두 TSV 파일 읽기
        logging.info(f"파일 읽기 시작: {qna_path}")
        df1 = pd.read_csv(qna_path, sep='\t')
        logging.info(f"파일 읽기 시작: {multiple_qna_path}")
        df2 = pd.read_csv(multiple_qna_path, sep='\t')
        
        # 데이터프레임 합치기
        logging.info("데이터프레임 병합 중...")
        combined_df = pd.concat([df1, df2], ignore_index=True)
        logging.info(f"총 데이터 수: {len(combined_df)}")
        
        # MBTI 컬럼을 대문자로 변환
        combined_df['a_mbti'] = combined_df['a_mbti'].str.upper()
        
        # MBTI별로 데이터 처리 (답변자 MBTI만 고려)
        for mbti in Config.MBTI_TYPES:
            logging.info(f"{mbti} 처리 중...")
            # 답변자 MBTI 기준으로 필터링
            mbti_df = combined_df[combined_df['a_mbti'] == mbti]
            logging.info(f"{mbti} 데이터 수: {len(mbti_df)}")
            
            mbti_data = self._process_conversations(mbti_df, mbti)
            
            # 결과 저장
            if mbti_data:
                self._save_to_jsonl(mbti_data, f"{mbti.lower()}_conversations.jsonl")
                logging.info(f"{mbti} 처리 완료: {len(mbti_data)}개 대화 저장됨")
        
        logging.info("모든 데이터 처리 완료")
    
    def _process_conversations(self, df: pd.DataFrame, mbti: str) -> List[Dict]:
        processed_data = []
        
        for _, row in df.iterrows():
            # NaN 체크
            if pd.isna(row['question']) or pd.isna(row['answer']):
                continue
                
            # [SEP] 토큰을 기준으로 문장 분리 후 다시 합치기
            input_text = " ".join(str(row['question']).split('[SEP]')).strip()
            output_text = " ".join(str(row['answer']).split('[SEP]')).strip()
            
            # 빈 문자열 체크
            if not input_text or not output_text:
                continue
                
            # 노이즈 체크
            if self._is_noisy(input_text) or self._is_noisy(output_text):
                continue
                
            # 언어 비율 체크
            input_kr_ratio, input_en_ratio = self._get_language_ratio(input_text)
            output_kr_ratio, output_en_ratio = self._get_language_ratio(output_text)
            
            if (input_kr_ratio < 0.5 or output_kr_ratio < 0.5 or  # 한글 비율 50% 미만
                input_en_ratio > 0.3 or output_en_ratio > 0.3):   # 영어 비율 30% 초과
                continue
                
            # 문장 유사도 체크
            similarity = self._calculate_similarity(input_text, output_text)
            if similarity < 0.3:  # 유사도가 0.3 미만이면 제외
                continue
                
            conversation = {
                "input": input_text,
                "output": output_text,
                "mbti": mbti,
                "similarity": float(similarity)  # 유사도 점수도 저장
            }
            processed_data.append(conversation)
            
        return processed_data
    
    def _save_to_jsonl(self, data: List[Dict], filename: str):
        output_path = self.output_dir / filename
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

def main():
    processor = DataProcessor()
    processor.process_tsv_files(
        qna_path="data/raw/qna_cleaned.tsv",
        multiple_qna_path="data/raw/multiple_qna_cleaned.tsv"
    )

if __name__ == "__main__":
    main() 