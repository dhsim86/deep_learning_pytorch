# io, json: 입출력 및 JSON 데이터 처리를 위한 파이썬 기본 라이브러리
# - 이미지 데이터를 메모리에서 처리하고 패션 제품의 속성 정보를 JSON 형태로 다룰 때 사용
import io
import json

# Python Imaging Library: 이미지 처리 라이브러리. 패션 제품 이미지를 로드하고 전처리시 사용
from PIL import Image

from datasets import load_dataset
from sklearn.model_selection import train_test_split

import torch

# AutoModelForVision2Seq: transformers 라이브러리에서 제공하는 VLM 로더
# - 텍스트 및 이미지를 동시에 처리하는 모델 로드시 사용
# - transformers 5.x 이상부터는 AutoModelForImageTextToText를 사용
# AutoProcessor: 특정 모델에 맞는 전처리기를 자동으로 불러오는 도구
# - 텍스트와 이미지를 모델이 처리할 수 있는 형태로 변환하는 역할
from transformers import AutoModelForImageTextToText, AutoProcessor

# Qwen3VLProcessor: Qwen3-VL 및 Qwen 3.5 멀티모달 모델을 위한 전처리기
# - Qwen3VLProcessor를 활용하여 Qwen 3.5 모델의 입력을 전처리할 때 사용
from transformers import Qwen3VLProcessor

# 비전 정보 처리시 사용
from qwen_vl_utils import process_vision_info


from trl import SFTConfig, SFTTrainer
from peft import LoraConfig

# 실험 추적 및 로깅을 위한 라이브러리
import wandb

wandb.init(mode="disabled")  # 비활성화 모드, 로깅없이 진행

######################################################################
# 시스템 및 유저 프롬프트 정의

## 시스템 프롬프트로 패션 제품의 이미지와 제품명을 보고 다양한 스타일 정보를 추론하는 분류 모델 역할 부여
system_message = (
    "당신은 이미지와 제품명(name)으로부터 패션/스타일 정보를 추론하는 분류 모델입니다."
)

## 제품명과 이미지, 2가지 입력 정보를 바탕으로 
## 7가지 속성(성별, 주/서브 카테고리, 계절, 용도, 기본색상, 제품타입)을 JSON 형태로 추론하도록 요청
prompt = """입력 정보:
- name: {name}
- image: [image]

위 정보를 바탕으로, 아래 7가지 key에 대한 값을 JSON 형태로 추론해 주세요:
1) gender
2) masterCategory
3) subCategory
4) season
5) usage
6) baseColour
7) articleType

출력시 **아래 JSON 예시 형태**를 반드시 지키세요:
{{
  "gender": "예시값",
  "masterCategory": "예시값",
  "subCategory": "예시값",
  "season": "예시값",
  "usage": "예시값",
  "baseColour": "예시값",
  "articleType": "예시값"
}}

# 예시
{{
  "gender": "Men",
  "masterCategory": "Accessories",
  "subCategory": "Eyewear",
  "season": "Winter",
  "usage": "Casual",
  "baseColour": "Blue",
  "articleType": "Sunglasses"
}}

# 주의
- 7개 항목 이외의 정보(텍스트, 문장 등)는 절대 포함하지 마세요.
"""

######################################################################
# 데이터 전처리 함수 정의

## 데이터셋의 여러 컬럼에 분산되어 있는 패션 속성 정보를 하나의 JSON 형태 레이블로 통합하는 역할
## -> 원본 데이터셋에는 각 개별 컬럼으로 존재하는데, 모델이 학습할 수 있도록 JSON 문자열로 변환
def combine_cols_to_label(example):
    # 실제 컬럼명에 맞게 수정
    label_dict = {
        "gender": example["gender"],
        "masterCategory": example["masterCategory"],
        "subCategory": example["subCategory"],
        "season": example["season"],
        "usage": example["usage"],
        "baseColour": example["baseColour"],
        "articleType": example["articleType"],
    }
    example["label"] = json.dumps(label_dict, ensure_ascii=False)
    return example

## 각 데이터 샘플을 OpenAI 형식의 대화 구조로 변환 (이후 모델 고유 chat template로 변환)
def format_data(sample):
    # Image.Image를 PngImageFile로 변환
    ## -> 이미지를 PNG 포맷으로 일괄 변환 후 로드하여 다양한 이미지 형태를 표준화해서 사용
    buffer = io.BytesIO()
    sample["image"].save(buffer, format="PNG") # 메모리 상에서 이미지를 PNG 포맷으로 변환
    buffer.seek(0)
    image = Image.open(buffer)

    ## 이를 통해 VLM 모델도 대화형태로 작동하도록 설계된 것을 알 수 있다.
    return {
        "messages": [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_message}],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt.format(name=sample["productDisplayName"]),
                    },
                    {
                        "type": "image",
                        "image": image,
                    },
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": sample["label"],
                    }
                ],
            },
        ],
    }

######################################################################
# 데이터셋 로드 및 전처리

## 패션 이미지 데이터셋, 패션 제품의 이미지와 다양한 속성 정보를 포함
dataset = load_dataset("ashraq/fashion-product-images-small", split="train")

## combine_cols_to_label를 전체 데이터셋에 적용하여 JSON 형태의 레이블을 추가
dataset_add_label = dataset.map(combine_cols_to_label)
## 데이터를 무작위로 섞어 초기 학습 데이터가 특정 패턴의 순서로 정렬되어있는 경우를 대비하여,
## 학습시 특정 패턴에 편향되지 않도록 한다.
dataset_add_label = dataset_add_label.shuffle(seed=4242)

# 첫 번째 샘플 확인
# {
#   'id': 15516, 
#   'gender': 'Men', 
#   'masterCategory': 'Footwear', 
#   'subCategory': 'Flip Flops', 
#   'articleType': 'Flip Flops', 
#   'baseColour': 'Navy Blue', 
#   'season': 'Fall', 
#   'year': 2011.0, 
#   'usage': 'Casual', 
#   'productDisplayName': 'Rockport Men Altrezlp Navy Blue Flip Flops', 
#   'image': <PIL.Image.Image image mode=RGB size=60x80 at 0x36A758980>, 
#   'label': '{"gender": "Men", "masterCategory": "Footwear", "subCategory": "Flip Flops", "season": "Fall", "usage": "Casual", "baseColour": "Navy Blue", "articleType": "Flip Flops"}'
# }
print("첫 번째 샘플 확인:", dataset_add_label[0])

## OpenAI 형식으로 변환
formatted_dataset = [format_data(row) for row in dataset_add_label]
# {
#   'messages':[
#       {
#           'role': 'system', 
#           'content': [
#               {
#                   'type': 'text', 
#                   'text': '당신은 이미지와 제품명(name)으로부터 패션/스타일 정보를 추론하는 분류 모델입니다.'
#               }
#           ]
#       }, 
#       {
#           'role': 'user', 
#           'content': [
#               {
#                   'type': 'text', 
#                   'text': '입력 정보:\n- name: Rockport Men Altrezlp Navy Blue Flip Flops\n- image: [image]\n\n위 정보를 바탕으로, 아래 7가지 key에 대한 값을 JSON 형태로 추론해 주세요:\n1) gender\n2) masterCategory\n3) subCategory\n4) season\n5) usage\n6) baseColour\n7) articleType\n\n출력시 **아래 JSON 예시 형태**를 반드시 지키세요:\n{\n  "gender": "예시값",\n  "masterCategory": "예시값",\n  "subCategory": "예시값",\n  "season": "예시값",\n  "usage": "예시값",\n  "baseColour": "예시값",\n  "articleType": "예시값"\n}\n\n# 예시\n{\n  "gender": "Men",\n  "masterCategory": "Accessories",\n  "subCategory": "Eyewear",\n  "season": "Winter",\n  "usage": "Casual",\n  "baseColour": "Blue",\n  "articleType": "Sunglasses"\n}\n\n# 주의\n- 7개 항목 이외의 정보(텍스트, 문장 등)는 절대 포함하지 마세요.\n'
#               }, 
#               {
#                   'type': 'image', 
#                   'image': <PIL.PngImagePlugin.PngImageFile image mode=RGB size=60x80 at 0x34FCCF230>
#               }
#           ]
#       }, 
#       {
#           'role': 'assistant', 
#           'content': [
#               {
#                   'type': 'text', 
#                   'text': '{"gender": "Men", "masterCategory": "Footwear", "subCategory": "Flip Flops", "season": "Fall", "usage": "Casual", "baseColour": "Navy Blue", "articleType": "Flip Flops"}'
#               }
#            ]
#       }
#   ]
# }
print("첫 번째 OpenAI 형식의 샘플 확인:", formatted_dataset[0])

print("\n=============================================")

######################################################################
# 데이터셋 로드 및 전처리

# test_size=0.2로 설정하여 전체 데이터의 20%를 테스트 세트로 분리
train_dataset, test_dataset = train_test_split(formatted_dataset, test_size=0.2, random_state=42)
print('학습 데이터의 개수:', len(train_dataset)) # 35257
print('테스트 데이터의 개수:', len(test_dataset)) # 8815

print("\n=============================================")

######################################################################
# 모델 및 프로세서 로드
use_mps = torch.backends.mps.is_available()

model_id = "Qwen/Qwen3.5-2B"

## 모델과 프로세서 로드
model = AutoModelForImageTextToText.from_pretrained(
    model_id,
    device_map="auto", # GPU 메모리에 자동 할당 (여러 GPU가 있는 경우 자동으로 분산 배치)
    torch_dtype=torch.float16 if use_mps else torch.bfloat16, # CPU로 떨어지면 기존대로 bfloat16
)

## 기존 LLM 학습할 때는 토크나이저를 로드하였지만, VLM에서는 프로세서를 로드
## - 프로세서는 모델에 특화된 전처리기를 로드하여 
##   토크나이저가 수행하던 텍스트 토크나이징 뿐만 아니라 이미지 전처리를 동시에 수행
processor = AutoProcessor.from_pretrained(model_id) # 텍스트/이미지 전처리기 로드

######################################################################
# 챗 탬플릿 적용 테스트
text = processor.apply_chat_template(train_dataset[0]["messages"], tokenize=False, add_generation_prompt=False)

## 여기서 <|vision_start|><|image_pad|><|vision_end|> 부분이 이미지가 삽입되는 위치이다.
# <|im_start|>system
# 당신은 이미지와 제품명(name)으로부터 패션/스타일 정보를 추론하는 분류 모델입니다.<|im_end|>
# <|im_start|>user
# 입력 정보:
# - name: Proline Men Blue Shorts
# - image: [image]
# 
# 위 정보를 바탕으로, 아래 7가지 key에 대한 값을 JSON 형태로 추론해 주세요:
# 1) gender
# 2) masterCategory
# 3) subCategory
# 4) season
# 5) usage
# 6) baseColour
# 7) articleType
# 
# 출력시 **아래 JSON 예시 형태**를 반드시 지키세요:
# {
#   "gender": "예시값",
#   "masterCategory": "예시값",
#   "subCategory": "예시값",
#   "season": "예시값",
#   "usage": "예시값",
#   "baseColour": "예시값",
#   "articleType": "예시값"
# }
# 
# # 예시
# {
#   "gender": "Men",
#   "masterCategory": "Accessories",
#   "subCategory": "Eyewear",
#   "season": "Winter",
#   "usage": "Casual",
#   "baseColour": "Blue",
#   "articleType": "Sunglasses"
# }
# 
# # 주의
# - 7개 항목 이외의 정보(텍스트, 문장 등)는 절대 포함하지 마세요.
# <|vision_start|><|image_pad|><|vision_end|><|im_end|>
# <|im_start|>assistant
# <think>
# 
# </think>
# 
# {"gender": "Men", "masterCategory": "Apparel", "subCategory": "Bottomwear", "season": "Fall", "usage": "Casual", "baseColour": "Blue", "articleType": "Shorts"}<|im_end|>
print("챗 템플릿 적용 후\n", text)

print("\n=============================================")

######################################################################
# 모델 학습 설정

## LoRA 설정
peft_config = LoraConfig(
    lora_alpha=128, # 모델 가중치에 LoRA 업데이트를 적용하는 정도를 조절하는 스케일링 계수
    lora_dropout=0.05, # 과적합을 방지하기 위한 드롭아웃 비율 설정
    r=256, # LoRA의 순위(rank) - 저차원 행렬의 차원을 결정
    bias="none", # 편향(bias) 업데이트 여부 - 'none'은 편향을 업데이트 하지않음 (대부분의 경우 편향 업데이트없어도 충분한 성능을 얻을 수 있다고 함)
    target_modules=[ # LoRA를 적용할 대상 모듈들 - 트랜스포머 모델의 주요 투영 레이어들
        "q_proj", # 어텐션: Query 투영 레이어
        "k_proj", # 어텐션: Key 투영 레이어
        "v_proj" # 어텐션: Value 투영 레이어
        "o_proj", # 어텐션: Output 투영 레이어
        "up_proj", # FFN: 상향 투영 레이어
        "gate_proj", # FFN: 게이트 투영 레이어
        "down_proj", # FFN: 하향 투영 레이어
    ],
    task_type="CAUSAL_LM", # 작업 유형 지정 - 인과적 언어 모델링(다음 토큰 예측)
)

## 모델 학습 파라미터 지정
args = SFTConfig(
    output_dir="output_dir", # 저장될 디렉토리와 저장소 ID
    num_train_epochs=2, # 학습할 총 에포크 수
    per_device_train_batch_size=4, # GPU당 배치 크기
    gradient_accumulation_steps=32, # 그래디언트 누적 스텝수
    gradient_checkpointing=True, # 메모리 절약을 위한 체크 포인팅
    optim="adamw_torch" if use_mps else "adamw_torch_fused", # 옵티마이저
    logging_steps=10, # 로그 기록 주기
    save_strategy="steps", # 저장 전략
    save_steps=50, # 저장 주기
    bf16=not use_mps,   # CUDA와 CPU는 bf16, bf16 mixed precision을 사용
    fp16=use_mps,       # 맥북(MPS)만 fp16
    learning_rate=1e-4, # 학습률 (VLM과 같은 대규모 모델은 상대적으로 낮은 학습률을 사용하는 것이 안정적)
    max_grad_norm=0.3, # 그래디언트 클리핑 (그레디언트 벡터의 노름(L2 norm)이 0.3을 초과하면 스케일링하여(0.3으로 줄여) 학습 안정성을 높인다.)
    warmup_ratio=0.03, # 워밍업 비율 (전체 학습 스텝의 3% 구간 동안 학습률을 0에서 설정한 학습률까지 점진적으로 증가, 학습 초기의 불안정성을 방지)
    lr_scheduler_type="constant", # 고정 학습률
    push_to_hub=False, # 허브 업로드 안함
    remove_unused_columns=False,
    dataset_kwargs={"skip_prepare_dataset": True},
    report_to=None
)