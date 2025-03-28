import torch
from peft import PeftModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import os

lora_target_mbti = "istp"

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
lora_path = os.path.join(project_root, "model", "lora", f"lora-kobart-{lora_target_mbti}")

base = AutoModelForSeq2SeqLM.from_pretrained("gogamza/kobart-base-v2")
tokenizer = AutoTokenizer.from_pretrained("gogamza/kobart-base-v2")
model = PeftModel.from_pretrained(base, lora_path).to("cuda")

input_text = ["아 오늘 너무 피곤하다.", 
              "오늘 날씨가 좋네요.",
              "나는 게임은 별로 안 좋아해",
        "이번 주말에 뭐하실 계획이에요?",
        "새로운 프로젝트를 시작하게 되었어요.",
        "점심 뭐 먹었어요?",
        "요즘 어떻게 지내요?",
        "나랑 영화 볼래요?",
        "이번 시험 잘 봤어요?",
        "취미가 뭐야?",
        "스트레스 받을 때 어떻게 해?",
        "좋아하는 음식이 뭐야?"]

for text in input_text:
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    inputs.pop("token_type_ids", None)

    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=50)
        print(tokenizer.decode(output[0], skip_special_tokens=True))