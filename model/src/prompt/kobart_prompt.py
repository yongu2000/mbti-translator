from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# 1. KoBART 로드
tokenizer = AutoTokenizer.from_pretrained("gogamza/kobart-base-v2")
model = AutoModelForSeq2SeqLM.from_pretrained("gogamza/kobart-base-v2")

instruction = (
    "문장을 ISTP 말투로 바꿔줘.\n"
    "입력: 배고플텐데 밥 맛있게 먹어\n출력:"
)

input_ids = tokenizer.encode(instruction, return_tensors="pt")
outputs = model.generate(
    input_ids,
    max_new_tokens=50,
    do_sample=True,
    top_k=50,
    top_p=0.95,
    temperature=0.8
)
print("결과 출력합니다")
# 4. 결과 출력
print(tokenizer.decode(outputs[0], skip_special_tokens=True))