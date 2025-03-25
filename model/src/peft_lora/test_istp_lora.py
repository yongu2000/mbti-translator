import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import os
from config import *

def load_model():
    """기본 모델과 ISTP LoRA 어댑터를 로드합니다."""
    # 기본 모델 로드
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        device_map="auto",
        torch_dtype=torch.float16,
        quantization_config={
            "load_in_4bit": True,
            "bnb_4bit_compute_dtype": torch.float16,
            "bnb_4bit_use_double_quant": True,
            "bnb_4bit_quant_type": "nf4"
        }
    )
    
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    
    # LoRA 어댑터 로드
    lora_path = os.path.join(CHECKPOINT_DIR, "istp")
    model = PeftModel.from_pretrained(base_model, lora_path)
    
    return model, tokenizer

def generate_text(model, tokenizer, input_text, max_length=256):
    """텍스트를 생성합니다."""
    prompt = f"입력: {input_text}\nISTP 스타일:"
    inputs = tokenizer(prompt, return_tensors="pt", max_length=max_length, truncation=True)
    
    # 입력을 GPU로 이동
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        num_beams=4,
        early_stopping=True,
        no_repeat_ngram_size=3,
        length_penalty=1.0,
        temperature=0.7,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        do_sample=True,  # 샘플링 활성화
        top_p=0.9,  # nucleus sampling
        top_k=50,  # top-k sampling
        repetition_penalty=1.2  # 반복 페널티
    )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # 프롬프트 부분 제거
    response = generated_text.split("ISTP 스타일:")[-1].strip()
    return response

def test_istp_style():
    """ISTP 스타일 변환을 테스트합니다."""
    # CUDA 캐시 초기화
    torch.cuda.empty_cache()
    
    # 모델 로드
    model, tokenizer = load_model()
    print("Model loaded successfully!")
    
    # 테스트할 입력 텍스트들
    test_inputs = [
        "오늘 날씨가 좋네요.",
        "이번 주말에 뭐하실 계획이에요?",
        "새로운 프로젝트를 시작하게 되었어요.",
        "점심 뭐 먹었어?",
        "요즘 어떻게 지내?",
        "나랑 영화 볼래?",
        "이번 시험 잘 봤어?",
        "취미가 뭐야?",
        "스트레스 받을 때 어떻게 해?",
        "좋아하는 음식이 뭐야?"
    ]
    
    print("\n=== ISTP Style Translation Test ===")
    for input_text in test_inputs:
        output = generate_text(model, tokenizer, input_text)
        print(f"\nInput: {input_text}")
        print(f"ISTP Style: {output}")
        print("-" * 50)
    
    # GPU 메모리 정리
    torch.cuda.empty_cache()

if __name__ == "__main__":
    test_istp_style() 