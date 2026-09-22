import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

######################################################################
# LoRA 병합 및 튜닝 모델 저장

# 베이스 모델 ID
model_id = "kakaocorp/kanana-2-1.3b-instruct"

# 학습된 LoRA 업데터가 저장된 경로
adapter_path = "./kanana2-1.3b-rag-ko"

# 병합된 모델이 저장될 경로
merged_model_path = "./output_dir"

# 디바이스 설정
device_arg = {"device_map": "auto"}

# 베이스 모델 로드
print(f"Loading base model from: {model_id}")
base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    return_dict=True, # 모델의 출력을 딕셔너리 형태로 반환하도록 설정
    torch_dtype=torch.bfloat16,
    attn_implementation="sdpa",
    trust_remote_code=True,
    **device_arg
)

# LoRA 어댑터 로드 및 병합
print(f"Loading and merging PEFT from: {adapter_path}")

## 베이스 모델 위에 학습된 LoRA 어댑터 로드
peft_model = PeftModel.from_pretrained(base_model, adapter_path)

## LoRA 어댑터 가중치를 베이스 모델에 병합
merged_model = peft_model.merge_and_unload()

# 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

# 병합된 모델 및 토크나이저를 저장
## 일반적으로 모델과 토크나이저는 같이 저장
print(f"Saving merged model to: {merged_model_path}")
merged_model.save_pretrained(merged_model_path)
tokenizer.save_pretrained(merged_model_path)
print("✅ 모델과 토크나이저 저장 완료")

### 주의!
# kanana 모델의 경우, 업로드 전에 config.json에서 rope_parameter를 변경해야 함
## as-is
# ```
#   "rope_parameters": {
#     "full_attention": {
#       "factor": 40.0,
#       "original_max_position_embeddings": 4096,
#       "rope_theta": 10000,
#       "rope_type": "yarn"
#     },
#     "rope_theta": 10000,
#     "rope_type": "default",
#     "sliding_attention": {
#       "rope_theta": 10000.0,
#       "rope_type": "default"
#     }
# }
# ```

## to-be
# ```
#  "rope_parameters": {
#    "full_attention": {
#      "factor": 40.0,
#      "original_max_position_embeddings": 4096,
#      "rope_theta": 10000,
#      "rope_type": "yarn"
#    },
#    "sliding_attention": {
#      "rope_theta": 10000.0,
#      "rope_type": "default"
#    }
#  }
# ```


#######################################################################
# 허깅페이스 업로드

from huggingface_hub import HfApi

# 허깅페이스 허브의 모든 기능에 접근할 수 있는 고수준 인터페이스를 제공
api = HfApi()

username = "raveas"
model_name = 'kanana-2-1.3b-instruct-rag-ko'

# 모델 저장소 생성
# api.create_repo(
#    token="hf_...",
#    repo_id=f"{username}/{model_name}",
#    repo_type="model" # 저장소가 모델을 저장하는 용도임을 명시
# )

# 모델 업로드
api.upload_folder(
    token="hf_...",
    repo_id=f"{username}/{model_name}",
    folder_path="./output_dir", # 병합된 모델이 저장된 로컬 경로 지정
)