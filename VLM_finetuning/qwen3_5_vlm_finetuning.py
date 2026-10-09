# 이미지 데이터를 메모리에서 처리하고 패션 제품의 속성 정보를 JSON 형태로 다룰 때 사용
import io
import json

# 이미지 처리 라이브러리. 패션 제품 이미지를 로드하고 전처리시 사용
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

## 제품명과 이미지, 2가지 입력 정보를 바탕으로 7가지 속성을 JSON 형태로 추론하도록 요청
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
    ## -> 이미지를 PNG형태로 일괄 변환하여 로드하여 다양한 이미지 형태를 표준화해서 사용
    buffer = io.BytesIO()
    sample["image"].save(buffer, format="PNG")
    buffer.seek(0)
    image = Image.open(buffer)

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
dataset_add_label = dataset_add_label.shuffle(seed=4242)
