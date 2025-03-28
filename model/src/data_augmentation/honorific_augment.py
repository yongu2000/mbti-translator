import json
import os
from typing import List, Dict
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer, pipeline

class HonorificConverter:
    def __init__(self):
        """존댓말 변환을 위한 초기화"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = T5ForConditionalGeneration.from_pretrained("j5ng/et5-formal-convertor")
        self.tokenizer = T5Tokenizer.from_pretrained("j5ng/et5-formal-convertor")
        self.model = self.model.to(self.device)
        
        # 파이프라인 초기화
        self.converter = pipeline(
            "text2text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if torch.cuda.is_available() else -1,
            framework="pt",
        )
    
    def to_honorific(self, text: str) -> str:
        """반말을 존댓말로 변환"""
        result = self.converter(
            "존댓말로 바꿔주세요: " + text,
            max_length=128,
            num_beams=5,
            early_stopping=True
        )[0]['generated_text']
        return result

def is_honorific(text: str) -> bool:
    """문장이 존댓말인지 확인"""
    # 기본 존댓말 어미
    honorific_endings = ['죠', '요', '니다']
    
    # 문장 끝에 존댓말 어미가 있는지 확인
    return any(text.endswith(ending) for ending in honorific_endings)

def augment_dataset(input_file: str, output_file: str):
    """데이터셋 증강 실행"""
    # 데이터 로드
    print("데이터 로드 중...")
    data = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:  # 빈 줄 건너뛰기
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"JSON 파싱 에러 발생: {e}")
                    print(f"문제가 있는 라인: {line}")
                    continue
    print(f"총 {len(data)}개의 데이터 로드 완료")
    
    # 존댓말 변환기 초기화
    print("\n모델 초기화 중...")
    converter = HonorificConverter()
    print("모델 초기화 완료")
    
    # 증강된 데이터 저장
    augmented_data = []
    honorific_count = 0
    
    print("\n데이터 증강 시작...")
    for idx, item in enumerate(data, 1):
        # 진행률 표시
        if idx % 10 == 0:  # 10개마다 진행률 표시
            print(f"진행률: {idx}/{len(data)} ({idx/len(data)*100:.1f}%)")
        
        # 원본 데이터 추가
        augmented_data.append(item)
        
        # 모든 메시지에 대해 존댓말 변환 시도
        new_item = item.copy()
        converted = False
        
        # original 메시지 변환
        if not is_honorific(item["original"]):
            new_item["original"] = converter.to_honorific(item["original"])
            converted = True
        
        # MBTI 메시지들 변환
        mbti_types = ["intj", "intp", "entj", "entp", "infj", "infp", "enfj", "enfp",
                      "istj", "istp", "estj", "estp", "isfj", "isfp", "esfj", "esfp"]
        
        for mbti in mbti_types:
            if not is_honorific(item[mbti]):
                new_item[mbti] = converter.to_honorific(item[mbti])
                converted = True
        
        # 변환이 하나라도 있었다면 추가
        if converted:
            honorific_count += 1
            augmented_data.append(new_item)
    
    print("\n증강된 데이터 저장 중...")
    # 증강된 데이터 저장
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in augmented_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print("\n=== 증강 완료 ===")
    print(f"원본 데이터: {len(data)}개")
    print(f"변환된 데이터 수: {honorific_count}개")
    print(f"증강 후 데이터: {len(augmented_data)}개")
    print(f"증강 비율: {len(augmented_data)/len(data):.1f}배")
    print("=" * 20)

if __name__ == "__main__":
    # 프로젝트 루트 디렉토리 경로 설정
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    
    # 입력/출력 파일 경로 설정
    input_file = os.path.join(project_root, "model", "data", "gpt_produced", "mbti_common_typeB.jsonl")
    output_file = os.path.join(project_root, "model", "data", "gpt_produced", "mbti_common_typeB_augmented.jsonl")
    
    print(f"입력 파일 경로: {input_file}")
    print(f"출력 파일 경로: {output_file}")
    
    augment_dataset(input_file, output_file)
