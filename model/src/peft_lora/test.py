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

input_text = "아 오늘 너무 피곤하다."
inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
inputs.pop("token_type_ids", None)

with torch.no_grad():
    output = model.generate(**inputs, max_new_tokens=50)
    print(tokenizer.decode(output[0], skip_special_tokens=True))