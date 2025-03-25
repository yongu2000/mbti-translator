import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel
import os
from config import *

def load_lora_model(mbti_type):
    """특정 MBTI 유형의 LoRA 모델을 로드합니다."""
    # 기본 모델 로드
    base_model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    
    # LoRA 가중치 로드
    lora_path = os.path.join(CHECKPOINT_DIR, mbti_type)
    model = PeftModel.from_pretrained(base_model, lora_path)
    
    return model, tokenizer

def generate_text(model, tokenizer, input_text, max_length=128):
    """텍스트를 생성합니다."""
    inputs = tokenizer(input_text, return_tensors="pt", max_length=max_length, truncation=True)
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        num_beams=4,
        early_stopping=True,
        no_repeat_ngram_size=3,
        length_penalty=1.0,
        temperature=0.7
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def test_mbti_style():
    """각 MBTI 유형별로 테스트를 수행합니다."""
    test_inputs = [
        "오늘 날씨가 좋네요.",
        "이번 주말에 뭐하실 계획이에요?",
        "새로운 프로젝트를 시작하게 되었어요."
    ]
    
    for mbti_type in MBTI_TYPES:
        print(f"\n=== Testing {mbti_type} style ===")
        model, tokenizer = load_lora_model(mbti_type)
        
        for input_text in test_inputs:
            output = generate_text(model, tokenizer, input_text)
            print(f"\nInput: {input_text}")
            print(f"Output: {output}")

if __name__ == "__main__":
    test_mbti_style() 