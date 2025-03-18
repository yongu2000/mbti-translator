def main():
    # 모델과 토크나이저 로드
    model_path = "checkpoints/checkpoint-4000"
    tokenizer = PreTrainedTokenizerFast.from_pretrained(model_path)
    model = MBTIStyleTransformer.from_pretrained(model_path)
    
    # GPU 사용 가능한 경우 GPU로 이동
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval() 