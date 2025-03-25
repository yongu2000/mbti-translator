import torch
from transformers import pipeline
import logging
from config import Config
from mbti_style_library import MBTI_STYLES

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StyleGenerator:
    def __init__(self):
        # GPU 사용 가능 여부 확인
        if not torch.cuda.is_available():
            raise RuntimeError("이 스크립트는 GPU가 필요합니다. CUDA를 사용할 수 있는 환경에서 실행해주세요.")
        
        self.device = "cuda"
        
        # 스타일 목록 정의 (MBTI 유형)
        self.styles = [
            'ISTJ', 'ISFJ', 'INFJ', 'INTJ',
            'ISTP', 'ISFP', 'INFP', 'INTP',
            'ESTP', 'ESFP', 'ENFP', 'ENTP',
            'ESTJ', 'ESFJ', 'ENFJ', 'ENTJ'
        ]
        
        # 모델 로드
        logger.info("모델 로딩 중...")
        self.model = pipeline(
            'text2text-generation',
            model='heegyu/kobart-text-style-transfer',
            device=0,  # GPU 사용
            model_kwargs={
                "temperature": 0.7,  # 낮은 temperature로 더 결정적인 출력
                "top_p": 0.9,  # nucleus sampling
                "repetition_penalty": 1.2,  # 반복 방지
                "do_sample": True,  # sampling 활성화
                "max_length": 64,  # 최대 길이
                "num_return_sequences": 1,  # 하나의 결과만 반환
                "early_stopping": True  # 조기 종료
            }
        )
        
        logger.info("생성 준비 완료!")
    
    def generate(self, text: str, style: str = None, max_length: int = 64) -> str:
        """주어진 텍스트를 지정된 스타일로 변환"""
        if style is None:
            style = self.styles[0]

        style_info = MBTI_STYLES[style]
        traits_text = "\n".join([f"- {trait}" for trait in style_info["traits"]])
        examples_text = "\n".join([f"- {example}" for example in style_info["examples"]])
            
        input_text = f"{style} 말투로 변환:{text}"
        
        # 텍스트 생성
        with torch.no_grad():
            output = self.model(
                input_text,
                max_length=max_length,

            )
        
        return output[0]['generated_text']

def main():
    # 테스트용 예시 문장들
    test_sentences = [
        "이 문제는 시간이 좀 걸릴 것 같아.",
    ]
    
    # 생성기 초기화 (모델은 한 번만 로드)
    print("모델 로딩 중...")
    generator = StyleGenerator()
    
    # 모든 스타일에 대해 테스트
    for style in generator.styles:
        print(f"\n=== {style} 스타일 테스트 ===")
        
        for sentence in test_sentences:
            print(f"\n입력: {sentence}")
            print("-" * 50)
            
            result = generator.generate(sentence, style)
            print(f"변환: {result}")
            print("-" * 50)

if __name__ == "__main__":
    main() 