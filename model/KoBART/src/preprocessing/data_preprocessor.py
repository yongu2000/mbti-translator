import os
import json
import logging
import pandas as pd
from typing import List, Dict, Any

class MBTIDataPreprocessor:
    def __init__(self, data_dir: str = "../../data"):
        """MBTI 데이터 전처리기 초기화
        
        Args:
            data_dir: 데이터 디렉토리 경로
        """
        self.raw_data_dir = os.path.join(data_dir, "raw")
        self.processed_data_dir = os.path.join(data_dir, "processed")
        
        # 로깅 설정
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # 처리된 데이터 디렉토리 생성
        os.makedirs(self.processed_data_dir, exist_ok=True)

    def load_qna_data(self) -> pd.DataFrame:
        """QnA 데이터 로드"""
        try:
            qna_file = os.path.join(self.raw_data_dir, "qna_cleaned.tsv")
            multiple_qna_file = os.path.join(self.raw_data_dir, "multiple_qna_cleaned.tsv")
            
            qna_data = pd.read_csv(qna_file, sep='\t')
            multiple_qna_data = pd.read_csv(multiple_qna_file, sep='\t')
            
            self.logger.info(f"QnA 데이터 로드 완료: {len(qna_data)} 행")
            self.logger.info(f"다중 QnA 데이터 로드 완료: {len(multiple_qna_data)} 행")
            
            return pd.concat([qna_data, multiple_qna_data], ignore_index=True)
        except Exception as e:
            self.logger.error(f"QnA 데이터 로드 실패: {str(e)}")
            return pd.DataFrame()

    def analyze_mbti_distribution(self, df: pd.DataFrame) -> None:
        """MBTI 유형 분포 분석"""
        for col in ['q_mbti', 'a_mbti']:
            if col in df.columns:
                mbti_counts = df[col].value_counts()
                self.logger.info(f"\n{col} 유형 분포:")
                for mbti, count in mbti_counts.items():
                    self.logger.info(f"{mbti}: {count}개")

    def process_text(self, text: str) -> str:
        """텍스트 전처리"""
        if pd.isna(text):
            return ""
        
        # [SEP] 토큰으로 분리된 경우 처리
        parts = text.split('[SEP]')
        cleaned_parts = [part.strip() for part in parts]
        return ' '.join(cleaned_parts)

    def create_style_transfer_pairs(self, qna_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """스타일 변환 학습을 위한 데이터 쌍 생성"""
        pairs = []
        
        # QnA 데이터에서 쌍 생성
        if not qna_data.empty:
            for _, row in qna_data.iterrows():
                if pd.notna(row.get('question')) and pd.notna(row.get('answer')) and \
                   pd.notna(row.get('q_mbti')) and pd.notna(row.get('a_mbti')):
                    pair = {
                        'source_text': self.process_text(row['question']),
                        'source_mbti': row['q_mbti'],
                        'target_text': self.process_text(row['answer']),
                        'target_mbti': row['a_mbti']
                    }
                    pairs.append(pair)
        
        self.logger.info(f"생성된 스타일 변환 쌍: {len(pairs)}개")
        return pairs

    def save_processed_data(self, pairs: List[Dict[str, Any]]) -> None:
        """처리된 데이터 저장"""
        output_file = os.path.join(self.processed_data_dir, "style_transfer_pairs.json")
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(pairs, f, ensure_ascii=False, indent=2)
            self.logger.info(f"처리된 데이터 저장 완료: {output_file}")
        except Exception as e:
            self.logger.error(f"데이터 저장 실패: {str(e)}")

def main():
    """메인 실행 함수"""
    preprocessor = MBTIDataPreprocessor()
    
    # QnA 데이터 로드
    qna_data = preprocessor.load_qna_data()
    
    # MBTI 분포 분석
    preprocessor.analyze_mbti_distribution(qna_data)
    
    # 스타일 변환 쌍 생성
    pairs = preprocessor.create_style_transfer_pairs(qna_data)
    
    # 처리된 데이터 저장
    preprocessor.save_processed_data(pairs)

if __name__ == "__main__":
    main() 