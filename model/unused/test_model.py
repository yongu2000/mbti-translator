import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import logging
from config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 프롬프트 템플릿 정의
PROMPT_TEMPLATE = (
    "### [MBTI: {mbti}]\n"
    "사용자: {input}\n"
    "AI ({mbti} 스타일):"
)

class MBTITester:
    def __init__(self, mbti_type: str = "ISTP"):
        self.mbti_type = mbti_type
        self.model_name = Config.BASE_MODEL
        
        # 기본 모델과 토크나이저 로드
        logger.info(f"모델 로딩 중: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            token=Config.HF_TOKEN
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 기본 모델 로드
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            token=Config.HF_TOKEN,
            device_map="auto",
            torch_dtype=torch.float16,
            quantization_config={
                "load_in_4bit": True,
                "bnb_4bit_compute_dtype": torch.float16,
                "bnb_4bit_use_double_quant": True,
                "bnb_4bit_quant_type": "nf4"
            }
        )
        
        # LoRA 가중치 로드
        self.model = PeftModel.from_pretrained(
            self.model,
            f"models/lora/{self.mbti_type.lower()}"
        )
        
    def generate_response(self, input_text: str, max_new_tokens: int = 100) -> str:
        # 프롬프트 생성
        prompt = PROMPT_TEMPLATE.format(
            mbti=self.mbti_type,
            input=input_text
        )
        
        # 입력 토크나이징
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        # 생성 설정
        generation_config = {
            "max_new_tokens": max_new_tokens,
            "num_return_sequences": 1,
            "temperature": 0.7,
            "top_p": 0.9,
            "do_sample": True,
            "repetition_penalty": 1.2,
            "length_penalty": 1.0,
            "pad_token_id": self.tokenizer.eos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id
        }
        
        # 응답 생성
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **generation_config)
        
        # 응답 디코딩
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 프롬프트 부분 제거
        response = response.replace(prompt, "").strip()
        
        return response

def main():
    # 테스터 초기화
    tester = MBTITester()
    
    # 테스트 케이스
    test_cases = [
        "오늘 날씨가 좋네요.",
        "취미가 뭐예요?",
        "스트레스 해소 방법이 궁금해요.",
        "친구와 다툰 것 같은데 어떻게 해야 할까요?"
    ]
    
    # 각 테스트 케이스에 대해 응답 생성
    for test_case in test_cases:
        print(f"\n입력: {test_case}")
        response = tester.generate_response(test_case)
        print(f"ISTP 응답: {response}")

if __name__ == "__main__":
    main() 