import json
import pandas as pd
import numpy as np
from collections import defaultdict
from typing import List, Dict, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import re
from torch.utils.data import Dataset, DataLoader

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            padding="max_length"
        )

class MBTIStyleAnalyzer:
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        print("Loading sentiment model...")
        self.sentiment_tokenizer = AutoTokenizer.from_pretrained("snunlp/KR-FinBert-SC")
        self.sentiment_model = AutoModelForSequenceClassification.from_pretrained("snunlp/KR-FinBert-SC").to(self.device)
        self.sentiment_model.eval()
        print(f"Model loaded and moved to {self.device}")
        
        self.mbti_data = defaultdict(list)
        self.batch_size = 32
        self.load_data()

    def load_data(self):
        """데이터 로드 및 MBTI 유형별로 분류"""
        with open(self.data_path, 'r', encoding='utf-8') as f:
            self.mbti_data = json.load(f)
        
        # 데이터 통계 출력
        print("\nMBTI 유형별 데이터 수:")
        for mbti, texts in self.mbti_data.items():
            print(f"{mbti}: {len(texts)}개의 텍스트")
        print()

    def split_sentences(self, text: str) -> List[str]:
        """문장 분리 함수"""
        # 기본적인 문장 구분자로 분리
        sentences = re.split('[.!?]\s+', text)
        # 빈 문장 제거
        return [sent.strip() for sent in sentences if sent.strip()]

    def tokenize_text(self, text: str) -> List[str]:
        """간단한 토크나이징 함수"""
        # 특수문자 제거
        text = re.sub(r'[^\w\s]', ' ', text)
        # 공백으로 분리
        return [word.strip() for word in text.split() if word.strip()]

    def analyze_sentence_length(self) -> Dict[str, float]:
        """문장 길이 분석"""
        avg_lengths = {}
        
        for mbti, texts in self.mbti_data.items():
            lengths = []
            for text in texts:
                sentences = self.split_sentences(text)
                lengths.extend([len(sent) for sent in sentences])
            avg_lengths[mbti] = np.mean(lengths) if lengths else 0
            
        return avg_lengths

    def analyze_sentiment_batch(self, texts: List[str]) -> List[Dict[str, float]]:
        """배치 단위로 감성 분석"""
        dataset = TextDataset(texts, self.sentiment_tokenizer)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        
        all_scores = []
        
        for batch in dataloader:
            # 배치 데이터를 GPU로 이동
            inputs = {k: v.squeeze(1).to(self.device) for k, v in batch.items()}
            
            with torch.no_grad():
                outputs = self.sentiment_model(**inputs)
                probs = torch.softmax(outputs.logits, dim=1)
                
            for prob in probs:
                scores = {
                    'negative': prob[0].item(),
                    'neutral': prob[1].item(),
                    'positive': prob[2].item()
                }
                all_scores.append(scores)
        
        return all_scores

    def analyze_sentiment(self) -> Dict[str, Dict[str, float]]:
        """감성 분석"""
        sentiment_scores = {}
        
        for mbti in tqdm(self.mbti_data.keys(), desc="MBTI 유형별 감성 분석"):
            texts = self.mbti_data[mbti]
            scores = self.analyze_sentiment_batch(texts)
            
            # 평균 계산
            avg_scores = defaultdict(float)
            for score in scores:
                for k, v in score.items():
                    avg_scores[k] += v
            
            total_texts = len(texts)
            sentiment_scores[mbti] = {k: v/total_texts for k, v in avg_scores.items()}
        
        return sentiment_scores

    def analyze_word_frequency(self) -> Dict[str, List[Tuple[str, float]]]:
        """단어 사용 빈도 분석 (TF-IDF)"""
        word_importance = {}
        
        for mbti in tqdm(self.mbti_data.keys(), desc="MBTI 유형별 단어 분석"):
            # 토큰화
            processed_texts = []
            for text in self.mbti_data[mbti]:
                tokens = self.tokenize_text(text)
                processed_text = ' '.join(tokens)
                processed_texts.append(processed_text)
            
            # TF-IDF 계산
            vectorizer = TfidfVectorizer(max_features=100)
            tfidf_matrix = vectorizer.fit_transform(processed_texts)
            
            # 중요 단어 추출
            feature_names = vectorizer.get_feature_names_out()
            importance_scores = np.mean(tfidf_matrix.toarray(), axis=0)
            
            # 상위 10개 단어 저장
            word_scores = list(zip(feature_names, importance_scores))
            word_scores.sort(key=lambda x: x[1], reverse=True)
            word_importance[mbti] = word_scores[:10]
            
        return word_importance

    def analyze_style_patterns(self) -> pd.DataFrame:
        """전체 스타일 패턴 분석 및 결과 생성"""
        # 문장 길이 분석
        print("분석 중: 문장 길이...")
        sentence_lengths = self.analyze_sentence_length()
        
        # 단어 빈도 분석
        print("분석 중: 단어 빈도...")
        word_frequencies = self.analyze_word_frequency()
        
        # 감성 분석
        print("분석 중: 감성...")
        sentiments = self.analyze_sentiment()
        
        # 결과 데이터프레임 생성
        results = []
        for mbti in self.mbti_data.keys():
            frequent_words = ', '.join([word for word, _ in word_frequencies[mbti][:5]])
            sentiment_pattern = max(sentiments[mbti].items(), key=lambda x: x[1])[0]
            
            # 스타일 특성 결정
            style_characteristics = self._determine_style_characteristics(
                sentence_lengths[mbti],
                sentiments[mbti],
                word_frequencies[mbti]
            )
            
            results.append({
                'MBTI': mbti,
                '평균_문장_길이': round(sentence_lengths[mbti], 2),
                '자주_쓰는_단어': frequent_words,
                '주요_감성': sentiment_pattern,
                '표현_방식': style_characteristics
            })
        
        return pd.DataFrame(results)

    def _determine_style_characteristics(self, 
                                      avg_length: float, 
                                      sentiment: Dict[str, float],
                                      word_freq: List[Tuple[str, float]]) -> str:
        """스타일 특성 결정"""
        characteristics = []
        
        # 문장 길이 기반 특성
        if avg_length < 8:
            characteristics.append("단답형")
        elif avg_length > 12:
            characteristics.append("서술형")
        
        # 감성 기반 특성
        main_sentiment = max(sentiment.items(), key=lambda x: x[1])[0]
        if main_sentiment == 'positive':
            characteristics.append("긍정적")
        elif main_sentiment == 'negative':
            characteristics.append("비판적")
        
        # 단어 사용 기반 특성
        word_set = set(word for word, _ in word_freq[:5])
        if any(w in word_set for w in ['생각', '의미', '이해']):
            characteristics.append("분석적")
        if any(w in word_set for w in ['좋아', '감사', '행복']):
            characteristics.append("정서적")
        
        return ', '.join(characteristics)

    def visualize_results(self, df: pd.DataFrame, output_dir: str):
        """결과 시각화"""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # 문장 길이 분포 시각화
        plt.figure(figsize=(12, 6))
        sns.barplot(data=df, x='MBTI', y='평균_문장_길이')
        plt.title('MBTI 유형별 평균 문장 길이')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/sentence_length_distribution.png')
        plt.close()

def main():
    # 데이터 경로 설정
    data_path = "../data/processed/style_transfer_pairs_sampled.json"
    output_dir = "../data/processed/analysis_results"
    
    # 출력 디렉토리 생성
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    print(f"Output directory created: {output_dir}")
    
    # 분석 실행
    analyzer = MBTIStyleAnalyzer(data_path)
    results_df = analyzer.analyze_style_patterns()
    
    # 결과 저장
    results_df.to_csv(f'{output_dir}/mbti_style_patterns.csv', index=False, encoding='utf-8-sig')
    analyzer.visualize_results(results_df, output_dir)
    print("분석이 완료되었습니다. 결과가 저장되었습니다.")

if __name__ == "__main__":
    main() 