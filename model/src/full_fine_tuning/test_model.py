import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import json
import os
import pickle

def load_model():
    # 모델과 토크나이저 로드
    model_path = os.path.abspath("model/checkpoints/mbti_style")
    # current_dir = os.path.dirname(os.path.abspath(__file__))
    # model_path = os.path.join(os.path.dirname(current_dir), "checkpoints", "mbti_style")
    
    # 모델과 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path, local_files_only=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer

def translate_text(model, tokenizer, text, target_mbti):
    # 입력 텍스트 준비 (학습 데이터 형식과 일치하도록)
    input_text = f"{target_mbti} 말투로 변환:{text}"
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True).to(model.device)
    
    inputs = {k: v for k, v in inputs.items() if k != "token_type_ids"}

    # # 👇 디버깅용 로그 추가
    # print(f"[디버깅] input_text: {input_text}")
    # print(f"[디버깅] input_ids: {inputs['input_ids']}")

    # 번역 생성
    outputs = model.generate(
        **inputs,
        max_new_tokens=64,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.2,
        early_stopping=True
    )
    
    # 결과 디코딩
    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translated_text

def test_model():
    # 테스트할 문장들
    test_sentences = [
        "더우니까 수박 먹고싶다",
        "너는 왜 나한테 그렇게 화만 내?",
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
        # "오늘 점심 뭐 먹었어?",
        # "주말에 뭐하실 계획이에요?"
    ]
    
    # 테스트할 MBTI 유형들
    mbti_types = ["ISTP", "ISFJ"]
    
    # 모델 로드
    model, tokenizer = load_model()
    
    print("\n=== MBTI 스타일 변환 테스트 ===")
    print("원본 문장 -> MBTI 유형별 변환 결과\n")
    
    for sentence in test_sentences:
        print(f"\n원본: {sentence}")
        print("-" * 50)
        for mbti in mbti_types:
            translated = translate_text(model, tokenizer, sentence, mbti)
            print(f"{mbti}: {translated}")
        print("-" * 50)

if __name__ == "__main__":
    test_model() 