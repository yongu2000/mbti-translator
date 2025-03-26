import json
import random
from typing import List, Dict
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

class ParaphraseAugmenter:
    def __init__(self, model_name: str = "EleutherAI/polyglot-ko-1.3b"):
        """파라프레이즈 증강을 위한 초기화"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        
    def paraphrase(self, text: str, num_variations: int = 3) -> List[str]:
        """주어진 텍스트의 파라프레이즈 변형을 생성"""
        variations = []
        prompt = f"다음 문장을 다른 표현으로 바꿔주세요:\n입력: {text}\n출력:"
        
        for _ in range(num_variations):
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt",
                return_token_type_ids=False  # token_type_ids 제거
            ).to(self.device)
            outputs = self.model.generate(
                **inputs,
                max_length=128,
                num_beams=5,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            variation = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # 프롬프트 부분 제거하고 실제 출력만 추출
            variation = variation.split("출력:")[-1].strip()
            variations.append(variation)
        
        return variations

def augment_dataset(input_file: str, output_file: str, num_variations: int = 3):
    """데이터셋 증강 실행"""
    # 데이터 로드
    with open(input_file, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    
    # 파라프레이즈 증강기 초기화
    augmenter = ParaphraseAugmenter()
    
    # 증강된 데이터 저장
    augmented_data = []
    for item in data:
        # 원본 데이터 추가
        augmented_data.append(item)
        
        # 원문 파라프레이즈 생성
        paraphrased_originals = augmenter.paraphrase(item["original"], num_variations)
        
        # 각 파라프레이즈에 대해 MBTI 변형 생성
        for paraphrased in paraphrased_originals:
            new_item = {
                "original": paraphrased,
                "intj": item["intj"],
                "intp": item["intp"],
                "entj": item["entj"],
                "entp": item["entp"],
                "infj": item["infj"],
                "infp": item["infp"],
                "enfj": item["enfj"],
                "enfp": item["enfp"],
                "istj": item["istj"],
                "istp": item["istp"],
                "estj": item["estj"],
                "estp": item["estp"],
                "isfj": item["isfj"],
                "isfp": item["isfp"],
                "esfj": item["esfj"],
                "esfp": item["esfp"]
            }
            augmented_data.append(new_item)
    
    # 증강된 데이터 저장
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in augmented_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"원본 데이터: {len(data)}개")
    print(f"증강 후 데이터: {len(augmented_data)}개")
    print(f"증강 비율: {len(augmented_data)/len(data):.1f}배")

if __name__ == "__main__":
    # 프로젝트 루트 디렉토리 경로 설정
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    
    # 입력/출력 파일 경로 설정
    input_file = os.path.join(project_root, "model", "data", "gpt_produced", "mbti_message.jsonl")
    output_file = os.path.join(project_root, "model", "data", "gpt_produced", "mbti_message_augmented.jsonl")
    
    print(f"입력 파일 경로: {input_file}")
    print(f"출력 파일 경로: {output_file}")
    
    augment_dataset(input_file, output_file, num_variations=3) 