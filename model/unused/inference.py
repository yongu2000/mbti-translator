import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from config import Config

PROMPT_TEMPLATE = (
    "### [MBTI: {mbti}]\n"
    "사용자: {input}\n"
    "AI ({mbti} 스타일):"
)

def load_model_and_tokenizer(base_model_name, lora_path):
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map="auto",
        torch_dtype=torch.float16,
        load_in_4bit=True
    )
    model = PeftModel.from_pretrained(base_model, lora_path)
    return tokenizer, model

def generate_response(input_text, mbti="ISTP", max_new_tokens=60):
    tokenizer, model = load_model_and_tokenizer(
        Config.BASE_MODEL,
        f"models/lora/{mbti.lower()}"
    )

    prompt = PROMPT_TEMPLATE.format(mbti=mbti, input=input_text)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.2,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
        )

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = result.replace(prompt, "").strip()
    return response

if __name__ == "__main__":
    input_text = "친구가 말도 없이 약속을 어겼어요."
    mbti = "ISTP"

    print(f"🤖 [{mbti} 스타일 응답]")
    output = generate_response(input_text, mbti)
    print(output)