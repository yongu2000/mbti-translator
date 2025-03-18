from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import torch
import sys
import os
import json
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.model.mbti_style_model import MBTIStyleTransformer
from transformers import AutoTokenizer, PreTrainedTokenizerFast

app = FastAPI(title="MBTI Style Translator API")

# 전역 변수로 모델과 토크나이저 저장
model = None
tokenizer = None

class TranslateRequest(BaseModel):
    sourceMbti: str
    targetMbti: str
    text: str

class TranslateResponse(BaseModel):
    translatedText: str

def load_model():
    global tokenizer, model
    try:
        # 모델과 토크나이저 로드
        base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        model_path = os.path.join(base_path, "checkpoints", "final")  # 최종 체크포인트 사용
        print(f"Loading model from: {model_path}")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model path not found: {model_path}")
        
        # MBTI 특수 토큰 추가
        mbti_types = [
            "ISTJ", "ISFJ", "INFJ", "INTJ",
            "ISTP", "ISFP", "INFP", "INTP",
            "ESTP", "ESFP", "ENFP", "ENTP",
            "ESTJ", "ESFJ", "ENFJ", "ENTJ"
        ]
        special_tokens = [f"<mbti={mbti}>" for mbti in mbti_types]
        
        # 토크나이저 설정
        tokenizer = PreTrainedTokenizerFast.from_pretrained(model_path)
        print(f"Original tokenizer vocab size: {len(tokenizer)}")
        
        # 특수 토큰 추가
        tokenizer.add_special_tokens({
            'additional_special_tokens': special_tokens,
            'pad_token': '[PAD]',
            'eos_token': '[EOS]',
            'bos_token': '[BOS]'
        })
        
        # 토크나이저 설정 업데이트
        tokenizer.pad_token = '[PAD]'
        tokenizer.eos_token = '[EOS]'
        tokenizer.bos_token = '[BOS]'
        
        print(f"Updated tokenizer vocab size: {len(tokenizer)}")
        print(f"Special tokens: {tokenizer.all_special_tokens}")
        
        # 모델 로드
        model = MBTIStyleTransformer(
            model_name=model_path,
            tokenizer=tokenizer,
            special_tokens=special_tokens
        )
        
        # GPU 사용 가능한 경우 GPU로 이동
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()
        
        print("모델 로드 완료")
        return True
    except Exception as e:
        print(f"모델 로드 실패: {str(e)}")
        return False

@app.on_event("startup")
async def startup_event():
    if not load_model():
        raise RuntimeError("Failed to load model during startup")

@app.post("/translate")
async def translate(request: TranslateRequest):
    global model, tokenizer
    if model is None or tokenizer is None:
        if not load_model():
            raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # 입력 텍스트 생성 (학습 데이터 형식과 일치)
        source_mbti_token = f"<mbti={request.sourceMbti.upper()}>"
        target_mbti_token = f"<mbti={request.targetMbti.upper()}>"
        input_text = f"{source_mbti_token} {target_mbti_token} {request.text}"
        print(f"입력 텍스트: {input_text}")
        
        # 토큰화
        inputs = tokenizer(
            input_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        print(f"입력 토큰 ID: {inputs['input_ids'][0]}")
        print(f"입력 토큰: {tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])}")
        
        # GPU 사용 가능한 경우 GPU로 이동
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
            model = model.to("cuda")
        
        # 예측
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=100,
                num_beams=5,
                do_sample=True,  # 샘플링 활성화
                temperature=0.7,
                top_k=50,
                top_p=0.9,
                repetition_penalty=1.2,
                length_penalty=1.0,
                early_stopping=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                bos_token_id=tokenizer.bos_token_id
            )
        
        # 결과 디코딩
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"출력 토큰 ID: {outputs[0]}")
        print(f"출력 토큰: {tokenizer.convert_ids_to_tokens(outputs[0])}")
        print(f"번역 결과: {result}")
        
        response = {"translatedText": result}
        print(f"응답 데이터: {json.dumps(response, ensure_ascii=False)}")
        return response
    
    except Exception as e:
        print(f"번역 중 오류 발생: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    global model
    return {"status": "healthy", "model_loaded": model is not None} 