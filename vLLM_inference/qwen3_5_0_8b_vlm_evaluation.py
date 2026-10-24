# Qwen3.5-0.8B VLM 파인튜닝 모델 평가 (vLLM 추론 + 속성별 F1)
# - 참고: 딥 러닝 파이토치 교과서 27-03 "vLLM을 이용한 파인튜닝 모델 평가"
# - 베이스 모델과 파인튜닝 모델을 같은 테스트 데이터로 추론해서 점수를 나란히 비교한다.
# - 실행: vLLM_inference 폴더에서 `uv run python qwen3_5_0_8b_vlm_evaluation.py`
#
# [주의] 실행 코드는 맨 아래 main()에 있고 `if __name__ == "__main__":`로 감싼다.
#   vLLM은 mac/Windows에서 엔진 프로세스를 spawn 방식으로 띄우는데, 이때 이 파일을 다시 import 한다.
#   가드가 없으면 자식 프로세스가 데이터 로드/모델 생성을 다시 실행하면서 에러가 난다.
#
# [mac] vllm-metal 플러그인의 기본 백엔드(MLX)로 추론한다.
#   kanana-2와 달리 Qwen3.5는 mlx-vlm에 구현(이미지 입력 포함)이 있어서 VLLM_METAL_BACKEND를 따로 설정하지 않는다.

import io
import json
import sys
from collections import defaultdict

import pandas as pd
from datasets import load_dataset
from PIL import Image
from qwen_vl_utils import process_vision_info
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from transformers import AutoProcessor
from vllm import LLM, SamplingParams

######################################################################
# 설정

BASE_MODEL_ID = "Qwen/Qwen3.5-0.8B"
TUNED_MODEL_ID = "raveas/Qwen3.5-0.8B-fashion-product-images-classification" # VLM_finetuning/qwen3_5_0_8b_vlm_lora_tuned_model_merge_and_upload_hf.py 로 업로드한 병합 모델

## 평가 샘플 수: 책과 같이 테스트 데이터 앞 50개
NUM_TEST_SAMPLES = 50

## vLLM 설정
## - 프롬프트는 이미지 자리(<|image_pad|> 70개)를 포함해 약 380토큰, 답변 JSON은 약 60토큰이라 max_model_len 2048이면 충분하다.
## - limit_mm_per_prompt: 프롬프트 하나에 들어가는 이미지/영상 최대 개수. 이 데이터는 이미지 1장뿐이라 책(각 10개)보다 줄인다.
VLLM_KWARGS = dict(max_model_len=2048, limit_mm_per_prompt={"image": 1, "video": 0})
if sys.platform == "darwin":
    ## gpu_memory_utilization: mac은 CPU와 GPU가 메모리를 같이 쓴다(통합 메모리).
    ## 기본값 0.92면 KV 캐시로 통합 메모리 대부분을 한 번에 잡으므로 낮춘다. (0.8B 모델에는 0.3도 충분)
    VLLM_KWARGS.update(gpu_memory_utilization=0.3)

######################################################################
# 1. 테스트 데이터 로드 및 전처리 (학습 코드 VLM_finetuning/qwen3_5_0_8b_vlm_finetuning.py와 동일)

system_message = (
    "당신은 이미지와 제품명(name)으로부터 패션/스타일 정보를 추론하는 분류 모델입니다."
)

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

def combine_cols_to_label(example):
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

def format_data(sample):
    # Image.Image를 PngImageFile로 변환
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
                    {"type": "text", "text": prompt.format(name=sample["productDisplayName"])},
                    {"type": "image", "image": image},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": sample["label"]}],
            },
        ],
    }

def load_test_data():
    """학습 코드와 같은 셔플(seed=4242)과 분할(test_size=0.2, random_state=42)로 테스트 데이터를 다시 만들고, 앞에서 몇 개를 고른다."""
    dataset = load_dataset("ashraq/fashion-product-images-small", split="train")
    dataset_add_label = dataset.map(combine_cols_to_label)
    dataset_add_label = dataset_add_label.shuffle(seed=4242)

    ## 학습 코드처럼 인덱스로 분할한다. (책은 test_size=0.9였지만 학습 코드는 0.2로 학습했다)
    ## 4.4만 장을 전부 format_data로 바꾸면 오래 걸리므로, 평가에 쓸 샘플만 골라서 변환한다.
    _, test_idx = train_test_split(range(len(dataset_add_label)), test_size=0.2, random_state=42)
    test_rows = dataset_add_label.select(test_idx[:NUM_TEST_SAMPLES])

    ## 파인튜닝 모델은 베이스 모델의 프로세서를 그대로 저장했으므로 프롬프트는 한 번만 만든다
    processor = AutoProcessor.from_pretrained(BASE_MODEL_ID)

    data = {"input": [], "label": [], "name": []}
    for row in test_rows:
        messages = format_data(row)["messages"]

        ## [책과 다른 점] 입력과 레이블 분리
        ## 책은 정답까지 넣고 템플릿을 적용한 뒤 '<|im_start|>assistant'에서 잘라 입력을 만든다.
        ## Qwen3.5 템플릿은 응답 앞에 빈 think 블록("\n<think>\n\n</think>\n\n")을 붙여서 학습했으므로
        ## 그렇게 자르면 학습 때와 입력이 달라진다.
        ## -> 정답(assistant)을 뺀 메시지에 add_generation_prompt=True로 적용하면 학습 때와 같은 헤더
        ##    "<|im_start|>assistant\n<think>\n\n</think>\n\n"까지 붙는다.
        input_part = processor.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
        label_part = messages[-1]["content"][0]["text"]

        ## 학습 때와 같은 이미지 전처리 (60x80 -> 56x84, 이후 vLLM 안의 프로세서가 224x320으로 한 번 더 리사이즈)
        image_inputs, video_inputs = process_vision_info(messages)

        mm_data = {}
        if image_inputs is not None:
            mm_data["image"] = image_inputs
        if video_inputs is not None:
            mm_data["video"] = video_inputs

        data["input"].append({"prompt": input_part, "multi_modal_data": mm_data})
        data["label"].append(label_part)
        data["name"].append(row["productDisplayName"])
    return data

######################################################################
# 2. 모델 추론

## 책과 같은 샘플링 설정
## top_p=0.001이면 확률이 가장 높은 토큰 하나만 남아서 사실상 매번 같은 결과가 나온다. (두 모델을 비교할 수 있음)
sampling_params = SamplingParams(
    temperature=0.1,
    top_p=0.001,
    repetition_penalty=1.05,
    max_tokens=256,
)

def generate_predictions(model_id, inputs):
    llm = LLM(model=model_id, **VLLM_KWARGS)
    outputs = llm.generate(inputs, sampling_params)
    ## 다음 모델을 올리기 전에 엔진 프로세스를 내려서 메모리를 돌려받는다
    llm.llm_engine.engine_core.shutdown()
    return [output.outputs[0].text.strip() for output in outputs]

######################################################################
# 3. 모델 평가: 속성(key)별 F1

def calculate_f1_scores(predictions, labels):
    """각 key 별로 F1 score를 계산하는 함수"""
    # 각 키별로 예측값과 실제값을 저장할 딕셔너리
    predictions_by_key = defaultdict(list)
    labels_by_key = defaultdict(list)
    num_parse_failures = 0

    for i, (pred_text, label_text) in enumerate(zip(predictions, labels)):
        # 모델 예측 결과 파싱
        ## [책과 다른 점] 베이스 모델은 JSON을 마크다운 코드 블록(```json ... ```)으로 감싸서 답한다.
        ## 책처럼 json.loads만 쓰면 베이스 모델은 전부 파싱 실패가 되어 비교가 안 되므로, 코드 블록 표시를 벗겨내고 파싱한다.
        pred_text = pred_text.removeprefix("```json").removeprefix("```").removesuffix("```")
        try:
            pred_json = json.loads(pred_text)
        except json.JSONDecodeError:
            print(f"샘플 {i+1}: 예측 결과 JSON 파싱 실패")
            num_parse_failures += 1
            continue

        # 실제 레이블 파싱 (format_data에서 json.dumps로 만든 문자열이라 실패하지 않는다)
        label_json = json.loads(label_text)

        # 각 키별로 예측값과 실제값 저장
        for key in label_json.keys():
            if key in pred_json:
                predictions_by_key[key].append(pred_json[key])
                labels_by_key[key].append(label_json[key])
            else:
                print(f"샘플 {i+1}: 키 '{key}' 예측 결과에 없음")

    # 각 키별로 F1 score 계산
    f1_scores = {}
    for key in labels_by_key.keys():
        if len(labels_by_key[key]) > 0:
            # 고유한 라벨들 확인
            unique_labels = list(set(labels_by_key[key] + predictions_by_key[key]))

            if len(unique_labels) == 1:
                # 모든 값이 동일한 경우
                f1_scores[key] = 1.0
            else:
                # F1 score 계산 (macro average)
                f1_scores[key] = f1_score(
                    labels_by_key[key],
                    predictions_by_key[key],
                    labels=unique_labels,
                    average="macro",
                    zero_division=0,
                )

    return f1_scores, predictions_by_key, labels_by_key, num_parse_failures

######################################################################
# 4. 베이스 모델 vs 파인튜닝 모델 비교

def main():
    data = load_test_data()
    print(f"평가 샘플 수: {len(data['input'])}")
    print("첫 번째 샘플 입력:")
    print(data["input"][0]["prompt"])

    print("\n=============================================")

    preds_by_model = {
        "base": generate_predictions(BASE_MODEL_ID, data["input"]),
        "tuned": generate_predictions(TUNED_MODEL_ID, data["input"]),
    }
    print(f"제품명: {data['name'][0]}")
    print(f"베이스 모델: {preds_by_model['base'][0]}")
    print(f"파인튜닝 모델: {preds_by_model['tuned'][0]}")
    print(f"레이블: {data['label'][0]}")

    print("\n=============================================")

    ## 책의 결과 표(Key / F1 Score / 샘플 수 + 키별 정확도)를 두 모델 나란히 놓고 비교한다
    ## - JSON 파싱에 실패했거나 key가 빠진 샘플은 책과 같이 계산에서 빠진다.
    ##   그래서 "샘플 수"가 50보다 작으면 그만큼 형식을 지키지 못한 것이고, 그 key의 F1/정확도는 나머지 샘플로만 계산한 값이다.
    results_by_model = {}
    summary = {}
    for model, preds in preds_by_model.items():
        print(f"\n[{model}]")
        f1_results, preds_by_key, labels_by_key, num_parse_failures = calculate_f1_scores(preds, data["label"])
        accuracy = {
            key: sum(1 for p, l in zip(preds_by_key[key], labels_by_key[key]) if p == l) / len(labels_by_key[key])
            for key in f1_results
        }
        results_by_model[model] = pd.DataFrame({
            "F1": f1_results,
            "정확도": accuracy,
            "샘플 수": {key: len(labels_by_key[key]) for key in f1_results},
        })
        summary[model] = {
            "평균 F1": results_by_model[model]["F1"].mean(),
            "평균 정확도": results_by_model[model]["정확도"].mean(),
            "JSON 파싱 실패 건수": num_parse_failures,
        }

    print("\n===== 속성별 결과 =====")
    print(pd.concat(results_by_model, axis=1).round(4).to_string())
    print("\n===== 베이스 모델 vs 파인튜닝 모델 =====")
    print(pd.DataFrame(summary).round(4).to_string())

    ## 샘플별 결과(제품명, 레이블, 두 모델의 예측)를 CSV로 저장해서 따로 살펴볼 수 있게 한다
    detail_df = pd.DataFrame({"name": data["name"], "label": data["label"]})
    for model, preds in preds_by_model.items():
        detail_df[f"{model}_pred"] = preds
    detail_df.to_csv("qwen3_5_0_8b_vlm_evaluation_results.csv", index=False, encoding="utf-8-sig")
    print("\n샘플별 결과 저장: qwen3_5_0_8b_vlm_evaluation_results.csv")


if __name__ == "__main__":
    main()
