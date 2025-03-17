from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import torch
from model.mbti_style_model import MBTIStyleTransformer
from transformers import AutoTokenizer

app = FastAPI(title="MBTI Style Translator API")

# 전역 변수로 모델과 토크나이저 저장
model = None
tokenizer = None

class TranslateRequest(BaseModel):
    source_mbti: str
    target_mbti: str
    text: str

@app.on_event("startup")
async def startup_event():
    global model, tokenizer
    try:
        # 토크나이저 로드
        tokenizer = AutoTokenizer.from_pretrained("gogamza/kobart-base-v2")
        
        # MBTI 특수 토큰 추가
        mbti_types = [
            "ISTJ", "ISFJ", "INFJ", "INTJ",
            "ISTP", "ISFP", "INFP", "INTP",
            "ESTP", "ESFP", "ENFP", "ENTP",
            "ESTJ", "ESFJ", "ENFJ", "ENTJ"
        ]
        special_tokens = [f"<mbti={mbti}>" for mbti in mbti_types]
        tokenizer.add_special_tokens({
            'additional_special_tokens': special_tokens
        })
        
        # 모델 로드
        model = MBTIStyleTransformer(
            model_name="gogamza/kobart-base-v2",
            tokenizer=tokenizer,
            special_tokens=special_tokens
        )
        model.load_state_dict(torch.load("checkpoints/final/pytorch_model.bin"))
        model.eval()
        
        print("모델 로드 완료")
    except Exception as e:
        print(f"모델 로드 실패: {str(e)}")
        raise

@app.post("/translate")
async def translate(request: TranslateRequest):
    try:
        # 입력 텍스트 생성
        input_text = f"<mbti={request.source_mbti}> <mbti={request.target_mbti}> {request.text}"
        
        # 토큰화
        inputs = tokenizer(input_text, return_tensors="pt", padding=True)
        
        # GPU 사용 가능한 경우 GPU로 이동
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
            model = model.to("cuda")
        
        # 예측
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"]
            )
        
        # 결과 디코딩
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return {"translated_text": result}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None} 