import re
from typing import List, Dict, Set
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob
from collections import Counter
import numpy as np

class TextCleaner:
    def __init__(self):
        # 불필요한 패턴 정의
        self.noise_patterns = [
            r'https?://\S+',  # URL
            r'[^\w\s\.,!?]',  # 특수문자 (기본 문장부호 제외)
        ]
        
        # 문장 길이 제한
        self.min_length = 5
        self.max_length = 50
        
        # 문장부호 제한
        self.max_punctuation = 3
        
        # MBTI별 특성 정의
        self.mbti_characteristics = {
            'F': {
                'emotion_words': ['감정', '기분', '마음', '사랑', '행복', '슬픔', '걱정', '걱정돼', '괜찮아'],
                'style': ['요', '네', '아요', '어요'],
                'sentiment_threshold': 0.3,  # F 성향은 긍정적 감성 선호
                'sentence_patterns': [
                    r'[^다]요$',  # 친근한 말투
                    r'[^다]네$',  # 부드러운 동의
                    r'[^다]아요$',  # 친근한 의문
                ]
            },
            'T': {
                'logic_words': ['분석', '논리', '사실', '데이터', '결과', '이유', '원인'],
                'style': ['다', '이다', '한다'],
                'sentiment_threshold': 0.0,  # T 성향은 중립적 감성 선호
                'sentence_patterns': [
                    r'다$',  # 직설적 말투
                    r'이다$',  # 명확한 진술
                    r'한다$',  # 행동 중심
                ]
            },
            'I': {
                'introvert_words': ['생각', '고민', '고민해', '생각해', '혼자'],
                'style': ['요', '네', '아요', '어요'],
                'sentence_patterns': [
                    r'생각[해|했|하]',  # 내향적 표현
                    r'고민[해|했|하]',  # 신중한 표현
                    r'혼자',  # 개인적 표현
                ]
            },
            'E': {
                'extrovert_words': ['같이', '함께', '우리', '여러분', '친구'],
                'style': ['요', '네', '아요', '어요'],
                'sentence_patterns': [
                    r'같이',  # 함께하는 표현
                    r'우리',  # 집단적 표현
                    r'여러분',  # 대화적 표현
                ]
            }
        }
        
        # 동적 단어 추출을 위한 TF-IDF 벡터화기
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words=['있다', '하다', '이다', '되다', '이다'],
            ngram_range=(1, 2)
        )
        
        # MBTI별 동적 단어 사전
        self.dynamic_words: Dict[str, Set[str]] = {}
    
    def analyze_sentiment(self, text: str) -> float:
        """감성 분석 수행 (개선된 버전)"""
        try:
            # TextBlob을 사용한 감성 분석
            blob = TextBlob(text)
            sentiment = blob.sentiment.polarity
            
            # 문장의 길이와 복잡도도 고려
            words = text.split()
            complexity = len(words) / 10  # 문장 길이에 따른 복잡도
            
            # 감성 점수 조정
            adjusted_sentiment = sentiment * (1 + complexity * 0.1)
            return max(min(adjusted_sentiment, 1.0), -1.0)  # -1에서 1 사이로 제한
            
        except:
            return 0.0
    
    def extract_dynamic_words(self, texts: List[str], mbti: str, top_n: int = 20) -> Set[str]:
        """TF-IDF를 사용하여 동적 단어 추출"""
        if not texts:
            return set()
        
        try:
            # TF-IDF 벡터화
            tfidf_matrix = self.vectorizer.fit_transform(texts)
            
            # 단어 중요도 계산
            feature_names = self.vectorizer.get_feature_names_out()
            tfidf_sums = np.array(tfidf_matrix.sum(axis=0)).flatten()
            
            # 상위 N개 단어 선택
            top_indices = tfidf_sums.argsort()[-top_n:][::-1]
            top_words = {feature_names[i] for i in top_indices}
            
            # MBTI 특성에 맞는 단어만 필터링
            filtered_words = set()
            for word in top_words:
                if self.is_mbti_appropriate_word(word, mbti):
                    filtered_words.add(word)
            
            return filtered_words
            
        except Exception as e:
            print(f"동적 단어 추출 중 오류 발생: {e}")
            return set()
    
    def is_mbti_appropriate_word(self, word: str, mbti: str) -> bool:
        """단어가 MBTI 특성에 적합한지 확인"""
        for trait in mbti:
            if trait in self.mbti_characteristics:
                char = self.mbti_characteristics[trait]
                
                # 감정/논리 단어 체크
                if 'emotion_words' in char and word in char['emotion_words']:
                    return True
                if 'logic_words' in char and word in char['logic_words']:
                    return True
                
                # 문체 스타일 체크
                if 'style' in char and any(word.endswith(style) for style in char['style']):
                    return True
                
                # 문장 패턴 체크
                if 'sentence_patterns' in char and any(re.search(pattern, word) for pattern in char['sentence_patterns']):
                    return True
        
        return False
    
    def get_mbti_style_score(self, text: str, mbti: str) -> float:
        """MBTI 스타일 점수 계산 (개선된 버전)"""
        score = 0.0
        
        # 감성 분석 점수
        sentiment = self.analyze_sentiment(text)
        
        # MBTI 특성에 따른 단어 매칭
        for trait in mbti:
            if trait in self.mbti_characteristics:
                char = self.mbti_characteristics[trait]
                
                # 감정/논리 단어 매칭
                if 'emotion_words' in char:
                    for word in char['emotion_words']:
                        if word in text:
                            score += 1.0
                if 'logic_words' in char:
                    for word in char['logic_words']:
                        if word in text:
                            score += 1.0
                
                # 동적 단어 매칭
                if mbti in self.dynamic_words:
                    for word in self.dynamic_words[mbti]:
                        if word in text:
                            score += 0.5
                
                # 문체 스타일 매칭
                for style in char['style']:
                    if text.endswith(style):
                        score += 0.5
                
                # 문장 패턴 매칭
                if 'sentence_patterns' in char:
                    for pattern in char['sentence_patterns']:
                        if re.search(pattern, text):
                            score += 0.3
                
                # 감성 임계값 체크
                if 'sentiment_threshold' in char:
                    if trait == 'F' and sentiment > char['sentiment_threshold']:
                        score += 0.5
                    elif trait == 'T' and abs(sentiment) < char['sentiment_threshold']:
                        score += 0.5
        
        return score
    
    def clean_text(self, text: str) -> str:
        """기본적인 텍스트 정제"""
        # 패턴 제거
        for pattern in self.noise_patterns:
            text = re.sub(pattern, '', text)
        
        # 공백 정리
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def is_valid_sentence(self, text: str) -> bool:
        """문장 유효성 검사"""
        # 길이 체크
        if not (self.min_length <= len(text) <= self.max_length):
            return False
        
        # 문장부호 개수 체크
        punctuation_count = len(re.findall(r'[.,!?]', text))
        if punctuation_count > self.max_punctuation:
            return False
        
        # 기본 문장 구조 체크 (주어+서술어)
        if not re.search(r'[은는이가을를]', text):  # 조사가 없는 경우
            return False
        
        return True
    
    def remove_similar_sentences(self, sentences: List[str], similarity_threshold: float = 0.8) -> List[str]:
        """유사 문장 제거"""
        if not sentences:
            return []
        
        # TF-IDF 벡터화
        vectorizer = TfidfVectorizer()
        try:
            tfidf_matrix = vectorizer.fit_transform(sentences)
        except ValueError:
            return sentences
        
        # 코사인 유사도 계산
        similarity_matrix = cosine_similarity(tfidf_matrix)
        
        # 유사 문장 제거
        unique_sentences = []
        for i in range(len(sentences)):
            is_unique = True
            for j in range(i):
                if similarity_matrix[i, j] > similarity_threshold:
                    is_unique = False
                    break
            if is_unique:
                unique_sentences.append(sentences[i])
        
        return unique_sentences
    
    def filter_by_mbti_characteristics(self, texts: List[str], mbti: str) -> List[str]:
        """MBTI 특성에 맞는 문장만 필터링 (개선된 버전)"""
        # 동적 단어 추출
        self.dynamic_words[mbti] = self.extract_dynamic_words(texts, mbti)
        
        filtered_texts = []
        for text in texts:
            # 감성 분석
            sentiment = self.analyze_sentiment(text)
            
            # MBTI 스타일 점수
            style_score = self.get_mbti_style_score(text, mbti)
            
            # F/T 성향에 따른 감성 필터링
            if 'F' in mbti and sentiment > self.mbti_characteristics['F']['sentiment_threshold']:
                filtered_texts.append(text)
            elif 'T' in mbti and abs(sentiment) < self.mbti_characteristics['T']['sentiment_threshold']:
                filtered_texts.append(text)
            # 스타일 점수가 높은 문장만 선택
            elif style_score >= 1.5:  # 임계값 상향 조정
                filtered_texts.append(text)
        
        return filtered_texts 