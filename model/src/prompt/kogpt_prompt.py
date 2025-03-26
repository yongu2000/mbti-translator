from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# KoGPT 로딩
tokenizer = AutoTokenizer.from_pretrained("skt/kogpt2-base-v2")
model = AutoModelForCausalLM.from_pretrained("skt/kogpt2-base-v2")
model.eval()

# 프롬프트 구성
prompt = "문장: 배고플텐데 밥 맛있게 먹어\nISTP 스타일:"

input_ids = tokenizer.encode(prompt, return_tensors="pt")

# 생성 설정
outputs = model.generate(
    input_ids,
    max_length=100,
    do_sample=True,
    top_k=50,
    top_p=0.95,
    temperature=0.8,
    pad_token_id=tokenizer.eos_token_id,
    repetition_penalty=1.2,
    no_repeat_ngram_size=3,
)

# 결과 출력
result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("\n=== 생성 결과 ===")
print(result)