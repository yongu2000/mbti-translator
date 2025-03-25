import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple
import pandas as pd
from tqdm import tqdm
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from config import Config
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    def __init__(self, input_dir: str, output_dir: str):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 한국어 종결어미 패턴
        self.ending_patterns = [
            r'[다요]$',  # ~다, ~요
            r'[당용]$',  # ~용
            r'[죠죵]$',    # ~죠 ~죵
            r'[네넹]$',    # ~네 ~넹
            r'[니까]$',  # ~니까
            r'[니깐]$',  # ~니깐
            r'[어아]$',  # ~어, ~아
            r'[군]$',    # ~군
            r'[지]$',    # ~지
            r'[구나]$',  # ~구나
            r'[구낭]$',  # ~구나
            r'[네요]$',  # ~네요
            r'[네용]$',  # ~네요
            r'[는걸]$',  # ~는걸
            r'[는데]$',  # ~는데
            r'[는뎅]$',  # ~는데
            r'[습니다]$' # ~습니다
            r'[습니당]$' # ~습니다
            r'[읍니다]$' # ~습니다
        ]
        
        # 불용어 패턴 (키워드나 특수 단어)
        self.stopword_patterns = [
            r'http[s]?://\S+',          # URL
            r'[a-zA-Z0-9_.+-]+@\S+',    # 이메일
            r'#\w+',                     # 해시태그
            r'@\w+',                     # 멘션
            r'\b[A-Z]{2,}\b',           # 대문자 약어
            r'\d{2,}',                   # 2자리 이상 숫자
            r'[ㄱ-ㅎㅏ-ㅣ]+',            # 한글 자음/모음만
            r'[!?]{2,}',                # 중복된 문장부호
        ]

    def _is_noisy(self, text: str) -> bool:
        """노이즈가 있는지 확인"""
        # 특수문자 비율 체크
        special_chars = re.findall(r'[^가-힣a-zA-Z0-9\s\.,!?]', text)
        if len(special_chars) / len(text) > 0.2:  # 특수문자 비율 20% 초과
            return True
            
        # 불용어 패턴 체크
        for pattern in self.stopword_patterns:
            if re.search(pattern, text):
                return True
                
        return False

    def _get_language_ratio(self, text: str) -> Tuple[float, float]:
        """한글과 영어 비율 계산"""
        korean = len(re.findall(r'[가-힣]', text))
        english = len(re.findall(r'[a-zA-Z]', text))
        total = len(text)
        return korean/total if total > 0 else 0, english/total if total > 0 else 0

    def _check_sentence_structure(self, text: str) -> bool:
        """문장 구조 검사"""
        # 최소 길이 체크
        if len(text) < 4:  # 최소 2음절 이상
            return False
            
        # 최대 길이 체크
        if len(text) > 100:  # 너무 긴 문장 제외
            return False
            
        # 종결어미 체크
        has_ending = False
        for pattern in self.ending_patterns:
            if re.search(pattern, text):
                has_ending = True
                break
                
        if not has_ending:
            return False
            
        # 기본적인 문장 구조 체크 (한글 음절이 최소 2개 이상)
        korean_syllables = re.findall(r'[가-힣]', text)
        if len(korean_syllables) < 2:
            return False
            
        return True

    def process_tsv_file(self, file_path: Path) -> List[Dict]:
        """TSV 파일을 처리하여 MBTI별 대화 데이터 추출"""
        try:
            df = pd.read_csv(file_path, sep='\t', encoding='utf-8')
            logger.info(f"TSV 파일 로드 완료: {file_path}")
            
            # 필요한 컬럼이 있는지 확인
            required_columns = ['answer', 'a_mbti']
            if not all(col in df.columns for col in required_columns):
                logger.warning(f"필요한 컬럼이 없습니다: {file_path}")
                return []
            
            total_count = len(df)
            logger.info(f"전체 데이터 수: {total_count}")
            
            # 데이터 추출
            conversations = []
            filtered_counts = {
                'na_values': 0,
                'mbti': 0,
                'structure': 0,
                'noise': 0,
                'language_ratio': 0
            }
            
            for _, row in tqdm(df.iterrows(), total=len(df), desc="데이터 필터링 중"):
                if pd.isna(row['a_mbti']) or pd.isna(row['answer']):
                    filtered_counts['na_values'] += 1
                    continue
                    
                mbti = row['a_mbti'].strip().upper()
                if not mbti or len(mbti) != 4:
                    filtered_counts['mbti'] += 1
                    continue
                
                answer = row['answer'].split('[SEP]')[0].strip()
                
                # 문장 구조 검사
                if not self._check_sentence_structure(answer):
                    filtered_counts['structure'] += 1
                    continue
                
                # 노이즈 체크
                if self._is_noisy(answer):
                    filtered_counts['noise'] += 1
                    continue
                
                # 한글 비율 체크
                kr_ratio, _ = self._get_language_ratio(answer)
                if kr_ratio < 0.5:  # 한글 50% 미만
                    filtered_counts['language_ratio'] += 1
                    continue
                
                conversations.append({
                    "output": answer,
                    "mbti": mbti,
                    "korean_ratio": round(kr_ratio * 100, 2),
                    "length": len(answer)
                })
            
            # 필터링 결과 출력
            logger.info("\n=== 필터링 결과 ===")
            logger.info(f"전체 데이터: {total_count}개")
            for filter_name, count in filtered_counts.items():
                logger.info(f"- {filter_name}에 의해 필터링: {count}개 ({count/total_count*100:.1f}%)")
            logger.info(f"최종 추출: {len(conversations)}개 ({len(conversations)/total_count*100:.1f}%)")
            
            return conversations
            
        except Exception as e:
            logger.error(f"TSV 파일 처리 중 오류 발생: {file_path}, {str(e)}")
            return []

    def process_json_file(self, file_path: Path) -> List[Dict]:
        """JSON 파일 처리"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return self.process_conversations(data)
        except Exception as e:
            logger.error(f"JSON 파일 처리 중 오류 발생: {file_path}, {str(e)}")
            return []

    def save_conversations(self, conversations: List[Dict], mbti_type: str):
        """대화 데이터를 JSONL 형식으로 저장"""
        if not conversations:
            return
            
        output_file = self.output_dir / f"{mbti_type.lower()}_conversations.jsonl"
        with open(output_file, 'w', encoding='utf-8') as f:
            for conv in conversations:
                f.write(json.dumps(conv, ensure_ascii=False) + '\n')
        logger.info(f"저장 완료: {output_file} ({len(conversations)}개)")

    def process_all_files(self):
        """모든 입력 파일 처리"""
        if not self.input_dir.exists():
            logger.error(f"입력 디렉토리를 찾을 수 없음: {self.input_dir}")
            return
            
        # TSV 파일 처리
        for file_path in self.input_dir.glob('*.tsv'):
            logger.info(f"\n=== TSV 파일 처리 시작: {file_path.name} ===")
            conversations = self.process_tsv_file(file_path)
            
            # MBTI 유형별로 분리하여 저장
            mbti_conversations = {}
            for conv in conversations:
                mbti = conv['mbti'].upper()
                if mbti not in mbti_conversations:
                    mbti_conversations[mbti] = []
                mbti_conversations[mbti].append(conv)
            
            # MBTI 유형별로 저장
            for mbti, convs in mbti_conversations.items():
                self.save_conversations(convs, mbti)
                
        # JSON 파일 처리
        for file_path in self.input_dir.glob('*.json'):
            logger.info(f"\n=== JSON 파일 처리 시작: {file_path.name} ===")
            conversations = self.process_json_file(file_path)
            if conversations:
                self.save_conversations(conversations, file_path.stem)

def main():
    # CUDA 디버깅을 위한 환경변수 설정
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    
    input_dir = "data/raw"
    output_dir = "data/processed_style"
    
    preprocessor = DataPreprocessor(input_dir, output_dir)
    preprocessor.process_all_files()

if __name__ == "__main__":
    main() 