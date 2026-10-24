######################################################################
# [Qwen3.5-0.8B 버전]
#   2B 버전(qwen3_5_2b_vlm_finetuning.py)과 코드는 같고, 모델 이름 / 배치 크기 / 숫자가 들어간 주석만 다르다.
#   0.8B와 2B는 층 구성, 토큰 ID, 챗 템플릿, 이미지 전처리 설정이 같아서 나머지 코드는 그대로 쓸 수 있다.
#
# [RTX 3080 Ti(VRAM 12GB)에서 학습하기 위해 적용한 메모리 절약 방법]
#   ① 프롬프트(system/user) 마스킹      -> collate_fn 7단계 (모델 응답만 학습)
#   ② loss_type="chunked_nll"           -> SFTConfig (로짓을 한꺼번에 만들지 않음)
#   ③ LoRA r=256 -> 16                  -> LoraConfig (학습 파라미터/옵티마이저 메모리 축소)
#   ④ 데이터셋 지연 변환(with_transform) -> 데이터셋 로드 및 전처리 (CPU RAM 절약)
#   ⑤ CUDA 메모리 할당자 설정            -> 바로 아래
#
# 위 ①~③을 적용하지 않으면 12GB GPU 기준 대략 아래처럼 차서 OOM이 나거나 크게 느려진다. (이 파일의 배치 크기 16 기준)
#   가중치(bf16) 약 1.6GB + LoRA 학습용 메모리 약 2.4GB + 로짓 약 16GB + 활성값 등
######################################################################

# [메모리 절약 ⑤] PyTorch CUDA 메모리 할당자 옵션 (반드시 torch를 import 하기 전에 설정해야 적용됨)
# - garbage_collection_threshold:0.8 : PyTorch가 쥐고 있는 캐시가 GPU 용량의 80%를 넘으면 안 쓰는 메모리를 반납
# - expandable_segments:True        : 메모리 덩어리를 늘려 써서 단편화를 줄임 (Linux 전용, Windows에서는 경고만 뜨므로 제외)
# - CUDA가 없는 맥북에서는 아무 일도 하지 않는다.
# - Windows라면 코드 밖에서 하나 더 설정할 것:
#   NVIDIA 제어판 > 3D 설정 관리 > "CUDA - 시스템 메모리 폴백 정책" > "시스템 메모리 폴백 없음 선호"
#   기본값(폴백 켜짐)이면 VRAM이 모자랄 때 에러 대신 시스템 RAM을 몰래 끌어다 써서 학습이 수십 배 느려진다.
#   (자세한 배경은 LLM_finetuning/kanana2_finance_news_analysis.py 의 [GPU 메모리 문제 총정리] 참고)
import os
import sys

if sys.platform == "win32":
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "garbage_collection_threshold:0.8")
else:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,garbage_collection_threshold:0.8")

# [Windows cuDNN 충돌 방지] torch를 import 하기 전에 설정
# - Windows용 torch는 시스템 cuDNN을 쓰지 않고, torch\lib 안에 들어 있는 자체 cuDNN만 쓴다. (torch 2.12.1 = cuDNN 9.20)
# - 그런데 이 9.20에는 cudnn_engines_tensor_ir64_9.dll이 빠져 있어서, PATH에 시스템 cuDNN이 있으면 이 파일만 거기서 불러온다.
#   -> 버전이 다른 cuDNN이 섞여, 비전 인코더의 Conv3d에서 CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH 에러가 난다.
#   (참고: https://github.com/pytorch/pytorch/issues/188892)
# - 해결: 이 파이썬 프로세스의 PATH에서만 해당 DLL이 있는 폴더를 뺀다. (시스템 환경변수는 바뀌지 않음)
#   torch에 필요한 CUDA DLL(cudart, cublas 등)은 torch\lib에 모두 들어 있어서 빼도 문제없다.
# - torch 2.14.1 이상은 이 DLL을 torch\lib에 직접 포함하므로, torch를 올리면 이 코드는 지워도 된다.
if sys.platform == "win32":
    os.environ["PATH"] = os.pathsep.join(
        p for p in os.environ["PATH"].split(os.pathsep)
        if not (p and os.path.isfile(os.path.join(p, "cudnn_engines_tensor_ir64_9.dll")))
    )

# io, json: 입출력 및 JSON 데이터 처리를 위한 파이썬 기본 라이브러리
# - 이미지 데이터를 메모리에서 처리하고 패션 제품의 속성 정보를 JSON 형태로 다룰 때 사용
import io
import json

# Python Imaging Library: 이미지 처리 라이브러리. 패션 제품 이미지를 로드하고 전처리시 사용
from PIL import Image

from datasets import load_dataset
from sklearn.model_selection import train_test_split

import torch

# AutoModelForImageTextToText: transformers 라이브러리에서 제공하는 VLM 로더
# - 텍스트 및 이미지를 동시에 처리하는 모델 로드시 사용
# - transformers 4.x에서 쓰던 AutoModelForVision2Seq는 5.x에서 제거되어, 5.x부터는 이 클래스를 사용
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
## (프롬프트 안의 "[image]"는 그냥 글자일 뿐이다. 실제 이미지는 format_data에서 user 메시지의 텍스트 뒤에 붙는다)
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
## [버그 수정] 기존 코드는 리스트 컴프리헨션으로 44,072개를 미리 변환한 python list를 만들었는데,
##   TRL 1.12의 SFTTrainer는 list를 받지 않는다.
##   -> TypeError: `train_dataset` must be a `Dataset` or `IterableDataset`, got `list`.
##   (Dataset.from_list(리스트)로 감싸는 것도 안 된다. 메시지 안의 PIL 이미지가 bytes 딕셔너리로 바뀌고,
##    모든 content에 "image": None 키가 생겨서 chat template이 텍스트 항목까지 이미지로 착각한다)
## [메모리 절약 ④] with_transform: 샘플을 꺼내는 순간에만 format_data를 실행하는 "지연 변환"
##   - Dataset 타입이 그대로 유지되므로 SFTTrainer에 넘길 수 있다.
##   - 기존 방식은 4.4만 장의 이미지를 한꺼번에 CPU RAM에 올렸지만, 지연 변환은 이미지를 디스크(Arrow 캐시)에 두고
##     학습에 필요한 배치만 그때그때 메모리에 올린다.
def format_batch(batch):
    # with_transform은 샘플 1개가 아니라 "배치"를 {"컬럼명": [값1, 값2, ...]} 형태로 넘겨준다.
    # -> 샘플 단위 딕셔너리로 풀어서 format_data를 적용한 뒤, 다시 {"messages": [...]} 형태로 돌려준다.
    samples = [dict(zip(batch.keys(), values)) for values in zip(*batch.values())]
    return {"messages": [format_data(sample)["messages"] for sample in samples]}

formatted_dataset = dataset_add_label.with_transform(format_batch)
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
# 학습/테스트 데이터 분할

# test_size=0.2로 설정하여 전체 데이터의 20%를 테스트 세트로 분리
## [버그 수정] SFTTrainer가 Dataset 타입만 받으므로, sklearn으로는 "인덱스"만 나누고
##   Dataset.select로 해당 인덱스의 샘플을 골라 Dataset 타입을 유지한다. (with_transform도 그대로 유지됨)
##   random_state=42는 기존과 같으므로 학습/테스트 분할 결과도 기존 코드와 동일하다.
train_idx, test_idx = train_test_split(range(len(formatted_dataset)), test_size=0.2, random_state=42)
train_dataset = formatted_dataset.select(train_idx)
test_dataset = formatted_dataset.select(test_idx)
print('학습 데이터의 개수:', len(train_dataset)) # 35257
print('테스트 데이터의 개수:', len(test_dataset)) # 8815

print("\n=============================================")

######################################################################
# 모델 및 프로세서 로드
use_mps = torch.backends.mps.is_available()

model_id = "Qwen/Qwen3.5-0.8B" # 대화용(instruct) 모델. 사전학습만 된 모델은 "Qwen/Qwen3.5-0.8B-Base"

## 모델과 프로세서 로드
## - Qwen3.5-0.8B: 파라미터 약 8.5억 개 (비전 인코더 약 1억 개 포함) -> bf16(파라미터당 2바이트) 기준 가중치만 약 1.6GB
model = AutoModelForImageTextToText.from_pretrained(
    model_id,
    device_map="auto", # GPU 메모리에 자동 할당 (여러 GPU가 있는 경우 자동으로 분산 배치)
    torch_dtype=torch.float16 if use_mps else torch.bfloat16, # CPU로 떨어지면 기존대로 bfloat16
                                                              # (transformers 5.x에서는 torch_dtype 대신 dtype 권장. 지금은 경고만 뜨고 동작)
)

## 기존 LLM 학습할 때는 토크나이저를 로드하였지만, VLM에서는 프로세서를 로드
## - 프로세서는 모델에 특화된 전처리기를 로드하여 
##   토크나이저가 수행하던 텍스트 토크나이징 뿐만 아니라 이미지 전처리를 동시에 수행
processor = AutoProcessor.from_pretrained(model_id) # 텍스트/이미지 전처리기 로드

######################################################################
# 챗 탬플릿 적용 테스트
text = processor.apply_chat_template(train_dataset[0]["messages"], tokenize=False, add_generation_prompt=False)

## 여기서 <|vision_start|><|image_pad|><|vision_end|> 부분이 이미지가 삽입되는 위치이다.
## - 이 단계(tokenize=False)에서는 <|image_pad|>가 1개뿐이지만, 이후 processor가 이미지 크기에 맞춰
##   여러 개(이 데이터에서는 70개)로 늘린다. (아래 collate_fn 테스트의 디코딩 결과 참고)
## - assistant 응답 앞의 빈 <think></think> 블록은 Qwen3.5 템플릿이 "생각하지 않는 모드"로 붙이는 것이다.
##   추론할 때 add_generation_prompt=True로 템플릿을 적용하면 같은 빈 블록이 프롬프트 끝에 붙으므로 학습/추론 형식이 일치한다.
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
## [메모리 절약 ③] r=256 -> 16 (lora_alpha도 "alpha = 2 x r" 관례에 맞춰 128 -> 32)
##   - 학습 파라미터 수: 약 1억 6,400만 개 -> 약 1,020만 개
##   - 학습 파라미터 1개당 fp32 가중치(4) + 기울기(4) + AdamW 상태 2개(8) = 약 16바이트가 필요
##     -> 약 2.4GB -> 약 0.15GB
##   - 7개 속성을 고르는 단순한 태스크라 r=16으로도 충분하다. (성능이 부족하면 32~64로 올려서 실험)
peft_config = LoraConfig(
    lora_alpha=32, # 모델 가중치에 LoRA 업데이트를 적용하는 정도를 조절하는 스케일링 계수 (실제 배율 = lora_alpha / r = 2)
    lora_dropout=0.05, # 과적합을 방지하기 위한 드롭아웃 비율 설정
    r=16, # LoRA의 순위(rank) - 저차원 행렬의 차원을 결정
    bias="none", # 편향(bias) 업데이트 여부 - 'none'은 편향을 업데이트 하지않음 (대부분의 경우 편향 업데이트없어도 충분한 성능을 얻을 수 있다고 함)
    ## [Qwen3.5 구조] 24개 층 중 18개는 Gated DeltaNet(선형 어텐션), 나머지 6개(4층마다 1개)만 일반 어텐션이다.
    ##   q/k/v/o_proj는 일반 어텐션 6개 층에만 있으므로, 선형 어텐션 층의 투영 레이어도 지정해야 모든 층에 LoRA가 붙는다.
    ##   (모듈 이름은 model.named_modules()로 확인할 수 있다)
    target_modules=[ # LoRA를 적용할 대상 모듈들 - 트랜스포머 모델의 주요 투영 레이어들
        # 일반 어텐션 (6개 층)
        "q_proj", # 어텐션: Query 투영 레이어
        "k_proj", # 어텐션: Key 투영 레이어
        "v_proj", # 어텐션: Value 투영 레이어
        "o_proj", # 어텐션: Output 투영 레이어
        # 선형 어텐션 Gated DeltaNet (18개 층)
        "in_proj_qkv", # Query/Key/Value를 한 번에 만드는 투영 레이어
        "in_proj_z",   # 출력 게이트(z) 투영 레이어
        "out_proj",    # Output 투영 레이어
                       # (in_proj_a, in_proj_b는 출력 차원이 16밖에 안 되는 작은 게이트용 레이어라 제외)
        # FFN (24개 층 전부)
        "up_proj", # FFN: 상향 투영 레이어
        "gate_proj", # FFN: 게이트 투영 레이어
        "down_proj", # FFN: 하향 투영 레이어
        # 비전 인코더의 레이어는 이름이 달라서(attn.qkv, mlp.linear_fc1 등) 여기에 매칭되지 않는다 -> 비전 인코더는 학습하지 않음
    ],
    task_type="CAUSAL_LM", # 작업 유형 지정 - 인과적 언어 모델링(다음 토큰 예측)
)

## 모델 학습 파라미터 지정
args = SFTConfig(
    output_dir="Qwen3.5-0.8B-fashion-product-images-classification", # 저장될 디렉토리와 저장소 ID
    num_train_epochs=2, # 학습할 총 에포크 수
    ## [0.8B 배치 크기] 2B 파일의 4 x 32 -> 16 x 8
    ##   - 실제 배치 크기(배치 크기 x 누적 스텝수)는 128 그대로라서, 학습률 등 나머지 설정과 전체 스텝 수(552)도 그대로 둔다.
    ##   - 0.8B는 가중치가 2B보다 약 2.5GB 작아서, 남는 메모리로 한 번에 처리하는 샘플 수를 늘린다.
    ##     (GPU는 작은 일을 여러 번 나눠 받는 것보다 큰 일을 한 번에 받을 때 대체로 효율이 좋다)
    ##   - RTX 3080 Ti(12GB)에서 직접 잰 결과:
    ##     16 x 8 -> GPU 메모리 약 7.1GB
    ##     32 x 8 -> GPU 메모리 10GB 이상 (12GB의 80%를 넘어 ⑤의 메모리 반납 설정이 작동하는 구간), 학습 속도는 16일 때와 비슷
    ##     -> 배치 16에서 이미 GPU를 거의 다 쓰고 있어서, 배치를 더 키워도 빨라지지 않는다.
    ##   - 누적 스텝수는 학습 시간에 거의 영향이 없다. (처리하는 전체 샘플 수는 같고 업데이트 횟수만 바뀐다)
    ##   - 배치 크기를 바꿀 때는 두 값의 곱을 128로 유지할 것 (예: OOM이 나면 8 x 16 또는 4 x 32)
    ##   - 이 데이터는 샘플 길이가 거의 같아서(약 431~438토큰) 스텝마다 메모리 사용량도 거의 같다.
    ##     -> 처음 몇 스텝에서 OOM이 나지 않으면 끝까지 나지 않는다.
    per_device_train_batch_size=16, # GPU당 배치 크기
    gradient_accumulation_steps=8, # 그래디언트 누적 스텝수 (16개씩 8번 모아서 1번 업데이트 -> 실제 배치 크기 = 16 x 8 = 128)
    gradient_checkpointing=True, # 메모리 절약을 위한 체크 포인팅 (중간 계산값을 저장하지 않고 역전파 때 다시 계산)
    optim="adamw_torch" if use_mps else "adamw_torch_fused", # 옵티마이저
    logging_steps=10, # 로그 기록 주기
    save_strategy="steps", # 저장 전략
    save_steps=50, # 저장 주기
    bf16=not use_mps,   # CUDA와 CPU는 bf16, bf16 mixed precision을 사용 (RTX 30xx 이상에서 지원)
    fp16=use_mps,       # 맥북(MPS)만 fp16
    learning_rate=1e-4, # 학습률 (LoRA는 새로 추가한 작은 행렬만 학습하므로, 전체 파인튜닝(1e-5~2e-5)보다 높은 1e-4 정도를 흔히 사용)
    max_grad_norm=0.3, # 그래디언트 클리핑 (그레디언트 벡터의 노름(L2 norm)이 0.3을 초과하면 스케일링하여(0.3으로 줄여) 학습 안정성을 높인다.)
    warmup_ratio=0.03, # 워밍업 비율 (전체 학습 스텝의 3% 구간 동안 학습률을 0에서 설정한 학습률까지 점진적으로 증가, 학습 초기의 불안정성을 방지)
                       # !! 단, 아래 lr_scheduler_type="constant"는 워밍업을 무시하므로 지금 설정에서는 워밍업이 적용되지 않는다.
                       #    워밍업을 쓰려면 "constant_with_warmup"을 사용해야 한다. (warmup_ratio도 deprecated -> warmup_steps 권장)
    lr_scheduler_type="constant_with_warmup", # 고정 학습률
    push_to_hub=False, # 허브 업로드 안함
    ## 데이터셋에 있는 "messages" 컬럼을 지우지 않고 collate_fn까지 그대로 전달
    ## (True면 Trainer가 model.forward()의 인자 이름에 없는 컬럼을 지워버려서 collate_fn이 빈 데이터를 받는다)
    remove_unused_columns=False,
    ## TRL의 자동 전처리(챗 템플릿 적용 -> 토크나이징)를 건너뜀. 이 파일에서는 collate_fn이 직접 처리한다.
    dataset_kwargs={"skip_prepare_dataset": True},
    # [메모리 절약 ②] 로짓(logits)을 한꺼번에 만들지 않고 나눠서 loss를 계산 (계산 결과는 일반 loss와 동일)
    ## - 로짓 = 토큰마다 "다음 토큰이 사전(vocab) 248,320개 중 무엇일지"를 매긴 점수표
    ## - 배치 16 x 438토큰 x 248,320개 -> 로짓 + float32 복사본 + softmax 결과를 합치면 약 16GB (12GB GPU에 들어가지도 않음)
    ##   (사전 크기가 2B와 같아서, 토큰 1개당 로짓 크기는 모델이 작아져도 줄지 않는다)
    ## - chunked_nll은 256토큰씩 나눠서 계산하고, label이 -100인 토큰은 로짓 계산 자체를 건너뛴다.
    ##   -> 프롬프트 마스킹(collate_fn 7단계)과 함께 쓰면 로짓을 계산할 토큰이 배치 16 기준 약 5,800개 -> 약 910개로 줄어든다.
    ## - TRL 1.12 기준 옵션. (더 최신 TRL에서는 기본값 "nll"이 이 방식으로 바뀌어 "chunked_nll"은 같은 뜻의 별칭으로만 남음)
    loss_type="chunked_nll",
    ## [버그 수정] report_to=None -> "none"
    ##   None을 넣으면 [None]으로 바뀌어 Trainer 생성 시 "ValueError: None is not supported" 에러가 난다.
    ##   로깅 도구(wandb, tensorboard 등)를 쓰지 않으려면 문자열 "none"을 넣어야 한다.
    report_to="none"
)

######################################################################
# 학습시 데이터 전처리 함수 (SFTTrainer의 data_collator) 정의

## [참고] TRL 내장 기능으로 "모델 응답만 학습"하기
## 이 파일은 학습 과정을 직접 확인하기 위해 collate_fn에서 라벨을 손수 만들지만, TRL에는 같은 일을 해주는 기능이 있다.
## 데이터를 prompt(입력) / completion(정답)으로 나눠 담고 data_collator를 넘기지 않으면,
## SFTTrainer가 completion(모델 응답) 부분만 학습한다. (completion_only_loss가 prompt-completion 데이터에서는 기본값 True)
##
##   def format_data(sample):
##       return {
##           "images": [sample["image"]],   # 이미지는 별도 컬럼에 담는다
##           "prompt": [
##               {"role": "system", "content": [{"type": "text", "text": system_message}]},
##               {"role": "user", "content": [
##                   {"type": "text", "text": prompt.format(name=sample["productDisplayName"])},
##                   {"type": "image"},       # 메시지에는 이미지 "자리"만 표시
##               ]},
##           ],
##           "completion": [
##               {"role": "assistant", "content": [{"type": "text", "text": sample["label"]}]},
##           ],
##       }
##
##   train_dataset = dataset_add_label.map(format_data, remove_columns=dataset_add_label.column_names)
##   trainer = SFTTrainer(model=model, args=args, train_dataset=train_dataset,
##                        peft_config=peft_config, processing_class=processor)  # data_collator 생략
##
## - messages 형식 그대로 assistant 응답만 학습하는 assistant_only_loss=True 옵션도 있지만,
##   이미지가 있는 데이터에서는 transformers 5.18 이상이 필요하다. (현재 사용 중인 5.14에서는 에러)

## 프롬프트 마스킹(collate_fn 7단계)에서 "모델 응답이 시작되는 위치"를 찾을 때 쓰는 토큰 ID
## "<|im_start|>assistant\n" -> [248045, 74455, 198]
assistant_header_ids = processor.tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)

def collate_fn(examples):
    """
    텍스트와 이미지가 포함된 대화 데이터를 모델 학습에 적합한 형태로 변환하는 함수
    (라벨은 모델 응답(assistant) 부분만 남기고 나머지는 -100으로 처리)
    Args:
        examples: 각각 "messages" 키를 가진 딕셔너리들의 리스트
            messages는 role(system/user/assistant)과 content를 포함하는 대화 형태
    Returns:
        batch: 모델 학습에 사용할 수 있는 토큰화된 텍스트, 이미지, 라벨이 포함된 배치
    """
    # 1단계: 텍스트 전처리 - 채팅 템플릿 적용
    # 각 예제의 messages를 모델 고유의 채팅 형식으로 변환
    # 모델마다 다른 특수 토큰과 형식을 사용 (예: <|im_start|>, <|system|> 등)
    texts = [processor.apply_chat_template(example["messages"], tokenize=False) for example in examples]

    # 2단계: 이미지 데이터 추출 및 전처리
    # messages에서 이미지 정보를 추출하여 모델이 처리할 수 있는 형태로 변환
    # process_vision_info()는 messages에서 이미지나 비디오 정보를 찾아 적절한 형태로 변환해주는, Qwen 유틸리티 함수
    #   - 입력 처리: 로컬 파일 경로, HTTP URL, Base64 인코딩 데이터, PIL 이미지 등 다양한 형식의 이미지와 비디오 입력을 파싱
    #   - 전처리 연동: 크기 조정(smart_resize), 프레임 추출(fps 설정) 등을 거쳐 transformers의 AutoProcessor에 전달할 수 있는 image_inputs와 video_inputs 형태로 변환
    # =============================================
    # 예시
    # from qwen_vl_utils import process_vision_info
    #
    # 1. 메시지 구조에서 이미지/비디오 데이터 추출
    # image_inputs, video_inputs = process_vision_info(messages)
    #
    # 2. 프로세서에 전달
    # inputs = processor(
    #     text=[text],
    #     images=image_inputs,
    #     videos=video_inputs,
    #     padding=True,
    #     return_tensors="pt",
    # )
    # =============================================
    # !! 참고: process_vision_info의 image_patch_size 기본값은 14(Qwen2.5-VL 기준)인데 Qwen3.5는 16이다.
    #    그래서 60x80 이미지가 여기서 56x84로 한 번, 아래 processor에서 224x320으로 한 번 더 리사이즈된다.
    #    (최종 결과는 동일하게 동작하지만 리사이즈가 두 번 일어난다)
    image_inputs = [process_vision_info(example["messages"])[0] for example in examples]

    # 3단계: 텍스트 토크나이징 + 이미지 인코딩
    # 텍스트를 토큰으로 변환하고 이미지를 인코딩하여 하나의 배치로 묶음
    # return_tensors="pt": PyTorch 텐서 형태로 반환
    # padding=True: 배치 내 모든 시퀀스를 같은 길이로 맞춤 (짧은 것은 패딩 토큰으로 채움)
    # -> 처리 후 pixel_values에 process_vision_info가 추출하고 전처리한 이미지 픽셀 데이터(고차원 텐서)가 들어간다.
    # -> input_ids에는 순수한 텍스트 토큰 ID만 들어가고, 이미지가 위치했던 자리에는 <|image_pad|>와 같은 특수 토큰 ID만 있다.
    #    (이때 <|image_pad|> 1개가 이미지 크기에 맞는 개수(이 데이터에서는 70개)로 늘어난다)
    batch = processor(text=texts, images=image_inputs, return_tensors="pt", padding=True)

    # 4단계: 라벨 생성 (손실 계산용)
    # input_ids를 복사하여 라벨로 사용 (다음 토큰 예측 학습을 위함)
    labels = batch["input_ids"].clone()

    # 5단계: 패딩 토큰 손실 계산에서 제외
    # 패딩된 부분은 실제 데이터가 아니므로 손실 계산에서 제외
    # -100으로 설정하면 CrossEntropyLoss에서 자동으로 무시됨
    labels[labels == processor.tokenizer.pad_token_id] = -100

    # 6단계: 이미지 토큰 손실 계산에서 제외
    # 이미지 토큰은 예측 대상이 아니므로 손실 계산에서 제외
    # (이미지 토큰은 user 프롬프트 안에 있으므로 아래 7단계에서 함께 -100이 된다. 즉 지금은 중복이지만,
    #  7단계를 끄고 "프롬프트까지 학습"하는 실험을 할 때를 위해 남겨둔다)
    if isinstance(processor, Qwen3VLProcessor):
        # Qwen3.5 모델에서 사용하는 특수 이미지 토큰들의 ID (config.json의 vision_start_token_id 등과 동일)
        # - 248053: 이미지 시작 토큰, <|vision_start|>
        # - 248054: 이미지 종료 토큰, <|vision_end|>
        # - 248056: 이미지 패치 토큰, <|image_pad|>
        #   - 입력된 이미지의 시각적 특징(Visual Features)이 텍스트 모델 내부에서 자리 잡을 공간을 확보하기 위해 사용하는 특수 패딩 토큰(Place-holder Token)
        #   - 이미지 패치(Patch)가 들어갈 자리 표시
        # (참고: 원본 코드의 Qwen2-VL에서는 같은 토큰들의 ID가 151652, 151653, 151655 였다)
        image_tokens = [248053, 248054, 248056]
    else:
        # 다른 비전-언어 모델의 이미지 토큰 ID 추출
        image_tokens = [processor.tokenizer.convert_tokens_to_ids(processor.image_token)]

    # 이미지 토큰들을 손실 계산에서 제외 (-100으로 설정)
    for image_token_id in image_tokens:
        labels[labels == image_token_id] = -100

    # 7단계: [메모리 절약 ①] 프롬프트(system/user) 손실 계산에서 제외 -> 모델 응답(assistant)만 학습
    # - 추론할 때 모델은 system/user 프롬프트를 입력으로 "받기만" 하고 직접 생성하지 않는다.
    #   그런데 프롬프트까지 학습하면, 모든 샘플에서 거의 똑같은 템플릿 문장을 외우는 데 학습이 쓰인다.
    #   (이 데이터는 샘플 1개 약 434토큰 중 프롬프트가 약 378토큰, 정답 JSON은 약 57토큰뿐이다)
    # - 방법: 마지막 "<|im_start|>assistant\n"의 위치를 찾아, 그 앞쪽(헤더 포함)을 전부 -100으로 바꾼다.
    #   -> 학습 대상으로 남는 것: "<think>\n\n</think>\n\n{정답 JSON}<|im_end|>\n"
    #      (빈 think 블록은 Qwen3.5 템플릿이 붙이는 응답 형식의 일부, <|im_end|>는 "여기서 응답을 끝낸다"를 배우기 위해 필요)
    # - 학습 대상 토큰이 줄어드는 만큼 chunked_nll(SFTConfig)이 계산할 로짓도 줄어 메모리가 절약된다.
    header_len = len(assistant_header_ids)
    for i, ids in enumerate(batch["input_ids"].tolist()):
        # 뒤에서부터 찾는다 (멀티턴 대화라면 마지막 assistant 차례가 정답이므로)
        for start in range(len(ids) - header_len, -1, -1):
            if ids[start:start + header_len] == assistant_header_ids:
                labels[i, :start + header_len] = -100
                break

    # 8단계: 최종 배치에 라벨 추가
    # 모델 학습시 손실 계산에 사용될 라벨을 배치에 추가
    batch["labels"] = labels
    
    return batch

## 전처리 결과 확인
example = train_dataset[0] # 데이터셋의 첫번째 샘플
print("첫 번째 샘플 데이터:")
print(example)

## collate_fn 테스트 (배치 크기 1로)
batch = collate_fn([example])

# {'messages': [{'role': 'system', 'content': [{'type': 'text', 'text': '당신은 이미지와 제품명(name)으로부터 패션/스타일 정보를 추론하는 분류 모델입니다.'}]}, {'role': 'user', 'content': [{'type': 'text', 'text': '입력 정보:\n- name: Proline Men Blue Shorts\n- image: [image]\n\n위 정보를 바탕으로, 아래 7가지 key에 대한 값을 JSON 형태로 추론해 주세요:\n1) gender\n2) masterCategory\n3) subCategory\n4) season\n5) usage\n6) baseColour\n7) articleType\n\n출력시 **아래 JSON 예시 형태**를 반드시 지키세요:\n{\n  "gender": "예시값",\n  "masterCategory": "예시값",\n  "subCategory": "예시값",\n  "season": "예시값",\n  "usage": "예시값",\n  "baseColour": "예시값",\n  "articleType": "예시값"\n}\n\n# 예시\n{\n  "gender": "Men",\n  "masterCategory": "Accessories",\n  "subCategory": "Eyewear",\n  "season": "Winter",\n  "usage": "Casual",\n  "baseColour": "Blue",\n  "articleType": "Sunglasses"\n}\n\n# 주의\n- 7개 항목 이외의 정보(텍스트, 문장 등)는 절대 포함하지 마세요.\n'}, {'type': 'image', 'image': <PIL.PngImagePlugin.PngImageFile image mode=RGB size=60x80 at 0x3D9A24B90>}]}, {'role': 'assistant', 'content': [{'type': 'text', 'text': '{"gender": "Men", "masterCategory": "Apparel", "subCategory": "Bottomwear", "season": "Fall", "usage": "Casual", "baseColour": "Blue", "articleType": "Shorts"}'}]}]}
print("\n처리된 배치 데이터:")

## 모델 입력시
## 1. pixel_values가 비전 인코더를 통해 시각 임베딩 벡터로 변환
## 2. input_ids에 있는 정수들도 텍스트 임베딩 벡터로 변환
## 3. input_ids에 있던 특수 토큰(<|image_pad|>)의 임베딩 위치에 시각 임베딩 벡터를 끼워 넣은 후 LLM으로 전달
print("입력 ID 형태:", batch["input_ids"].shape) # torch.Size([1, 431]) # 이미지 자리(<|image_pad|> 70개)를 포함해 총 431 토큰
print("어텐션 마스크 형태:", batch["attention_mask"].shape) # torch.Size([1, 431])
## 이미지는 16x16 픽셀 크기의 패치(조각) 단위로 잘려서 들어간다.
## - 60x80 이미지가 224x320으로 확대된 뒤 -> 가로 14개 x 세로 20개 = 280개 패치
## - 패치 1개 = 1536개 값 (RGB 3채널 x 2프레임 x 16 x 16, 이미지는 같은 장면 2프레임짜리 영상처럼 처리됨)
## - 모델 안에서 2x2 패치를 1개 토큰으로 합치므로 280 / 4 = 70개 -> input_ids의 <|image_pad|> 70개와 일치
print("이미지 픽셀 형태:", batch["pixel_values"].shape) # torch.Size([280, 1536])
print("레이블 형태:", batch["labels"].shape) # torch.Size([1, 431])

# tensor([248045,   8678,    198,  62602,  79683,  32587, 185238,  77792, 151564,
#          76887,   3052,      8, 187094, 207028,     14, 160743,  31020, 176435,
#          55895, 150868,  41486, 184844, 160610, 151396,     13, 248046,    198,
#         248045,    846,    198,  42415,  27813,  58012,     25,    198,     12,
# ...
print('입력에 대한 정수 인코딩 결과:')
print(batch["input_ids"][0])

## 프롬프트(system/user, 이미지 토큰 포함) 부분은 전부 -100이고, 모델 응답 부분(431개 중 56개)만 원래 토큰 ID가 남는다.
# tensor([  -100,   -100,   -100, ...,   -100,   -100,
#         248068,    271, 248069,    271,   4754,  12596, ...,  12115,     82,   8934, 248046,    198])
# -> 248068='<think>', 271='\n\n', 248069='</think>', 4754='{"', 12596='gender', ...,
#    12115='Short', 82='s', 8934='"}', 248046='<|im_end|>', 198='\n'
print('레이블에 대한 정수 인코딩 결과:')
print(batch["labels"][0])

## 토큰 디코딩 예시 (입력 텍스트가 어떻게 변환 되었는지 확인)
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
# <|vision_start|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|vision_end|><|im_end|>
# <|im_start|>assistant
# <think>
# 
# </think>
# 
# {"gender": "Men", "masterCategory": "Apparel", "subCategory": "Bottomwear", "season": "Fall", "usage": "Casual", "baseColour": "Blue", "articleType": "Shorts"}<|im_end|>
decoded_text = processor.tokenizer.decode(batch["input_ids"][0])
print("\n디코딩된 텍스트:")
print(decoded_text)

## 레이블이 -100이 아닌 부분(= 실제로 학습하는 부분)만 골라서 디코딩
# <think>
#
# </think>
#
# {"gender": "Men", "masterCategory": "Apparel", "subCategory": "Bottomwear", "season": "Fall", "usage": "Casual", "baseColour": "Blue", "articleType": "Shorts"}<|im_end|>
label_ids = batch["labels"][0]
print("\n학습 대상(레이블이 -100이 아닌 부분) 디코딩:")
print(processor.tokenizer.decode(batch["input_ids"][0][label_ids != -100]))

print("\n=============================================")

######################################################################
# 모델 학습

trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset, # datasets.Dataset 타입이어야 한다 (위 [버그 수정] 참고)
    data_collator=collate_fn,
    peft_config=peft_config, # 원본 모델은 고정하고, target_modules에 LoRA 어댑터를 붙여 그 부분만 학습
    ## [버그 수정] tokenizer=processor.tokenizer -> processing_class=processor
    ##   - TRL 1.12의 SFTTrainer에는 tokenizer 인자가 없다 (TypeError). 이름이 processing_class로 바뀌었다.
    ##   - tokenizer가 아니라 processor를 넘겨야 TRL이 VLM(이미지+텍스트 모델)으로 인식하고,
    ##     save_model() 때 이미지 전처리 설정까지 함께 저장한다.
    processing_class=processor,
)

## 학습 시작
trainer.train() # save_steps(50) 마다 output_dir/checkpoint-50, checkpoint-100, ... 에 중간 결과가 저장됨 (push_to_hub=False라 허브 업로드는 안 함)
## 모델 저장
trainer.save_model() # 최종 결과를 output_dir에 저장. LoRA 학습이므로 원본 모델 전체가 아니라 학습한 LoRA 어댑터 가중치만 저장된다