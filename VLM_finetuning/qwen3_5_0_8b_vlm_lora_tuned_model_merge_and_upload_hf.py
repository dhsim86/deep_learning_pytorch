from transformers import AutoModelForImageTextToText, AutoProcessor
from peft import PeftModel

######################################################################
# LoRA 병합 및 튜닝 모델 저장

# 베이스 모델 ID
model_id = "Qwen/Qwen3.5-0.8B"

# 학습된 LoRA 업데터가 저장된 경로
adapter_path = "./Qwen3.5-0.8B-fashion-product-images-classification"

# 병합된 모델이 저장될 경로
merged_model_path = "./qwen3_5_0_8b_vlm_lora_tuned_model"

# 베이스 모델 로드
print(f"Loading base model from: {model_id}")
base_model = AutoModelForImageTextToText.from_pretrained(
    model_id,
    device_map="auto"
)

# LoRA 어댑터 로드 및 병합
print(f"Loading and merging PEFT from: {adapter_path}")

## 베이스 모델 위에 학습된 LoRA 어댑터 로드
peft_model = PeftModel.from_pretrained(base_model, adapter_path)

## LoRA 어댑터 가중치를 베이스 모델에 병합
merged_model = peft_model.merge_and_unload()

# 프로세서 로드
processor = AutoProcessor.from_pretrained(model_id)

# 병합된 모델 및 프로세서를 저장
## 일반적으로 모델과 프로세서는 같이 저장
print(f"Saving merged model to: {merged_model_path}")

## safe_serialization=True: 모델을 safetensor 형식으로 저장
## max_shard_size="2GB": 모델을 여러 파일로 분할시 각 파일의 최대 크기
merged_model.save_pretrained(merged_model_path, safe_serialization=True, max_shard_size="2GB")
processor.save_pretrained(merged_model_path)
print("✅ 모델과 프로세서 저장 완료")


#######################################################################
# 허깅페이스 업로드

from huggingface_hub import HfApi

# 허깅페이스 허브의 모든 기능에 접근할 수 있는 고수준 인터페이스를 제공
api = HfApi()

username = "raveas"
model_name = 'Qwen3.5-0.8B-fashion-product-images-classification'

# 모델 저장소 생성
api.create_repo(
    token="hf_...",
    repo_id=f"{username}/{model_name}",
    repo_type="model" # 저장소가 모델을 저장하는 용도임을 명시
)

# 모델 업로드
api.upload_folder(
    token="hf_...",
    repo_id=f"{username}/{model_name}",
    folder_path=merged_model_path, # 병합된 모델이 저장된 로컬 경로 지정
)