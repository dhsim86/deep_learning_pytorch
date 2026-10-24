# kanana-2 RAG 파인튜닝 모델 평가 (vLLM 추론 + LLM 기반 평가 + 인용 문서 F1)
# - 참고: 딥 러닝 파이토치 교과서 26-08 "vLLM과 LLM 기반의 RAG 평가하기"
# - 베이스 모델과 파인튜닝 모델을 같은 테스트 데이터로 추론해서 점수를 나란히 비교한다.
# - 실행: vLLM_inference 폴더에서 `NAMC_API_KEY=<키> uv run python kanana2_rag_evaluation.py`
#
# [주의] 실행 코드는 맨 아래 main()에 있고 `if __name__ == "__main__":`로 감싼다.
#   vLLM은 mac/Windows에서 엔진 프로세스를 spawn 방식으로 띄우는데, 이때 이 파일을 다시 import 한다.
#   가드가 없으면 자식 프로세스가 데이터 로드/모델 생성을 다시 실행하면서 에러가 난다.

import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import List

## mac: vllm-metal 플러그인의 PyTorch MPS(Apple GPU) 백엔드로 추론한다. (vllm import 전에 설정)
## - 기본값인 MLX 백엔드는 mlx-lm에 kanana-2 구현이 없어서 "Model type kanana2_tiny not supported"로 실패한다.
## - MPS 백엔드는 vLLM의 Transformers 백엔드를 그대로 써서 kanana-2가 돈다. (CPU보다 약 7배 빠름)
## - PYTORCH_ENABLE_MPS_FALLBACK=0: MPS에 없는 연산을 몰래 CPU로 돌리지 않게 한다. (vllm-metal 권장 설정)
if sys.platform == "darwin":
    os.environ.setdefault("VLLM_METAL_BACKEND", "mps")
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "0")

import pandas as pd
from datasets import load_dataset
from openai import OpenAI
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

######################################################################
# 설정

BASE_MODEL_ID = "kakaocorp/kanana-2-1.3b-instruct"
TUNED_MODEL_ID = "raveas/kanana-2-1.3b-instruct-rag-ko" # RAG_finetuning/lora_tuned_model_merge_and_upload_hf.py 로 업로드한 병합 모델

## 평가 샘플 수: 테스트 데이터의 type 5가지에서 10개씩, 총 50개
## 책은 테스트 데이터 앞 50개를 쓰지만, 이 테스트 데이터는 type별로 이어 붙인 것이라 앞에서 자르면 한두 type에 몰린다.
NUM_SAMPLES_PER_TYPE = 10

## LLM 평가 모델: 사내 NAMC Open Models API (OpenAI 호환이라 openai 패키지를 그대로 쓴다)
## API 키는 코드에 남기지 않도록 환경 변수로 받는다. (키가 없으면 오래 걸리는 추론 전에 여기서 바로 실패)
client = OpenAI(
    base_url="https://namc-aigw.io.naver.com",
    api_key=os.environ["NAMC_API_KEY"],
)
JUDGE_MODEL = "gpt-6-luna"

## vLLM 설정
## - kanana-2(Kanana2TinyForCausalLM)는 vLLM에 전용 구현이 없어서, trust_remote_code=True로 허깅페이스 모델 코드를 받아
##   vLLM의 Transformers 백엔드로 실행한다. (sliding window 레이어도 이 백엔드가 처리)
## - 프롬프트(검색 결과 + 질문)는 최대 약 3,900 토큰, 답변은 최대 1,024 토큰이므로 max_model_len 8192면 충분하다.
VLLM_KWARGS = dict(dtype="bfloat16", trust_remote_code=True, max_model_len=8192)
if sys.platform == "darwin":
    ## gpu_memory_utilization: mac은 CPU와 GPU가 메모리를 같이 쓴다(통합 메모리).
    ## 기본값 0.92면 KV 캐시로 약 40GB를 한 번에 잡으려다 실패하므로 낮춘다. (0.3이면 KV 캐시 약 8만 토큰)
    VLLM_KWARGS.update(gpu_memory_utilization=0.3)

######################################################################
# 1. 테스트 데이터 로드 및 전처리

## 학습 코드(RAG_finetuning/kanana2_rag_finetuning.py)와 같은 시스템 프롬프트
system_message = """당신은 검색 결과를 바탕으로 질문에 답변해야 합니다.

다음의 지시사항을 따르십시오.
1. 질문과 검색 결과를 바탕으로 답변하십시오.
2. 검색 결과에 없는 내용을 답변하려고 하지 마십시오.
3. 질문에 대한 답이 검색 결과에 없다면 검색 결과에는 "해당 질문에 대한 내용이 없습니다."라고 답변하십시오.
4. 답변할 때 특정 문서를 참고하여 문장 또는 문단을 작성했다면 뒤에 출처는 이중 리스트로 해당 문서 번호를 남기십시오. 예를 들어서 특정 문장이나 문단을 1번 문서에서 인용했다면 뒤에 [[ref1]]이라고 기재하십시오.
5. 예를 들어서 특정 문장이나 문단을 1번 문서와 5번 문서에서 동시에 인용했다면 뒤에 [[ref1]], [[ref5]]이라고 기재하십시오.
6. 최대한 다수의 문서를 인용하여 답변하십시오.

검색 결과:
-----
{search_result}"""

def load_test_data():
    """학습 코드와 같은 분할 규칙(type별 앞 20%가 테스트)으로 테스트 데이터를 다시 만들고, type별로 앞에서 몇 개씩 고른다."""
    ## 학습 때 저장한 rag_finetuning_test_dataset 폴더가 없어도 같은 테스트 데이터를 만들 수 있다.
    dataset = load_dataset("iamjoon/klue-mrc-ko-rag-dataset", split="train")

    test_ratio = 0.2
    dataset_types = dataset["type"]
    test_indices = []
    ## set 순회 순서는 실행할 때마다 달라질 수 있어서 정렬해서 순서를 고정한다
    for type_name in sorted(set(dataset_types)):
        curr_type_data = [i for i in range(len(dataset)) if dataset_types[i] == type_name]
        test_size = int(len(curr_type_data) * test_ratio)
        test_indices.extend(curr_type_data[:test_size][:NUM_SAMPLES_PER_TYPE])

    ## 파인튜닝 모델은 베이스 모델의 토크나이저를 그대로 저장했으므로 프롬프트는 한 번만 만든다
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)

    data = {"prompt": [], "label": [], "question": [], "context": [], "type": []}
    for i in test_indices:
        sample = dataset[i]
        search_result = "\n-----\n".join([f"문서{idx + 1}: {result}" for idx, result in enumerate(sample["search_result"])])
        messages = [
            {"role": "system", "content": system_message.format(search_result=search_result)},
            {"role": "user", "content": sample["question"]},
        ]
        ## add_generation_prompt=True면 학습 때와 같은 assistant 헤더("<|im_start|>assistant\n<think>\n\n</think>\n\n")까지 붙는다
        data["prompt"].append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
        data["label"].append(sample["answer"])
        data["question"].append(sample["question"])
        data["context"].append(search_result)
        data["type"].append(sample["type"])
    return data

######################################################################
# 2. 모델 추론

## temperature=0: 매번 같은 결과가 나와야 두 모델을 비교할 수 있다
sampling_params = SamplingParams(
    temperature=0,
    max_tokens=1024,
)

def generate_answers(model_id, prompts):
    llm = LLM(model=model_id, **VLLM_KWARGS)
    ## [중요] add_special_tokens=False
    ## kanana-2 토크나이저는 기본값에서 맨 앞에 <|begin_of_text|>를 붙이는데, 학습 입력(apply_chat_template 결과)에는 없었다.
    ## (LLM_finetuning/kanana2_finance_news_analysis.py 추론 부분 참조)
    outputs = llm.generate(prompts, sampling_params, tokenization_kwargs={"add_special_tokens": False})
    ## 다음 모델을 올리기 전에 엔진 프로세스를 내려서 메모리를 돌려받는다
    llm.llm_engine.engine_core.shutdown()
    return [output.outputs[0].text for output in outputs]

######################################################################
# 3. LLM 기반의 평가

prompt_template = """
당신은 RAG(Retrieval-Augmented Generation) 시스템 평가 전문가입니다. 아래 정보를 바탕으로 생성된 답변의 품질을 철저히 평가해주세요.

질문: {question}

검색된 컨텍스트:
{context}

생성된 답변:
{prediction}

참조 답변(정답):
{label}

다음 4가지 평가 기준으로 1-5점 척도로 점수를 매겨주세요:

1. 응답 정확성 (Answer Correctness) [1-5]:
   * 생성된 답변이 참조 답변과 비교하여 정확하고 완전한 정보를 제공하는지 평가
   * 1점: 완전히 잘못된 정보
   * 2점: 부분적으로 관련된 정보를 담고 있으나 대부분 부정확함
   * 3점: 정확한 정보와 부정확한 정보가 혼재되어 있음
   * 4점: 대부분 정확하지만 일부 정보가 누락되거나 미미한 오류가 있음
   * 5점: 참조 답변과 비교했을 때 완전히 정확하고 포괄적인 정보를 제공함

2. 컨텍스트 관련성 (Context Relevance) [1-5]:
   * 검색된 컨텍스트가 질문에 대답하기 위해 관련성이 높은지 평가
   * 1점: 컨텍스트가 질문과 전혀 관련이 없음
   * 2점: 컨텍스트가 질문과 간접적으로만 관련됨
   * 3점: 컨텍스트 중 일부만 질문과 직접적으로 관련됨
   * 4점: 대부분의 컨텍스트가 질문과 직접적으로 관련됨
   * 5점: 모든 컨텍스트가 질문에 완벽하게 관련되어 있고 불필요한 정보가 없음

3. 컨텍스트 충실성 (Context Faithfulness) [1-5]:
   * 생성된 답변이 주어진 컨텍스트에만 기반하는지, 아니면 없는 정보를 추가했는지 평가
   * 1점: 답변이 컨텍스트에 없는 정보로만 구성됨 (심각한 환각)
   * 2점: 답변이 주로 컨텍스트에 없는 정보로 구성됨
   * 3점: 답변이 컨텍스트 정보와 없는 정보가 혼합되어 있음
   * 4점: 답변이 주로 컨텍스트에 기반하지만 약간의 추가 정보가 있음
   * 5점: 답변이 전적으로 컨텍스트에 있는 정보만을 사용함

4. 컨텍스트 충분성 (Context Recall) [1-5]:
   * 검색된 컨텍스트가 질문에 완전히 답변하기에 충분한 정보를 포함하는지 평가
   * 1점: 컨텍스트가 답변에 필요한 정보를 전혀 포함하지 않음
   * 2점: 컨텍스트가 필요한 정보의 일부만 포함함
   * 3점: 컨텍스트가 필요한 정보의 약 절반을 포함함
   * 4점: 컨텍스트가 필요한 정보의 대부분을 포함하지만 일부 누락됨
   * 5점: 컨텍스트가 질문에 완전히 답변하기 위한 모든 필요한 정보를 포함함

반드시 다음 JSON 형식으로만 응답하세요. 마크다운은 사용하지 않습니다.:
{
  "answer_correctness": 정수로 된 점수(1-5),
  "context_relevance": 정수로 된 점수(1-5),
  "context_faithfulness": 정수로 된 점수(1-5),
  "context_recall": 점수(1-5),
  "analysis": "종합적인 분석 의견"
}

다른 형식의 응답은 하지 마세요. 오직 마크다운이 아닌 JSON만 반환하세요.
"""

METRICS = ["answer_correctness", "context_relevance", "context_faithfulness", "context_recall"]

def evaluate_one(question, context, prediction, label):
    """한 샘플을 LLM으로 평가해서 4가지 점수와 분석 의견을 돌려준다."""
    ## 프롬프트 안에 JSON 예시의 중괄호가 있어서 format 대신 replace를 쓴다
    prompt = (prompt_template
              .replace("{question}", question)
              .replace("{context}", context)
              .replace("{prediction}", prediction)
              .replace("{label}", label))
    try:
        response = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[
                {"role": "system", "content": "당신은 RAG 평가 도구입니다. 반드시 유효한 JSON 형식으로만 응답하세요."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )
        result = json.loads(response.choices[0].message.content)
        row = {metric: result[metric] for metric in METRICS}
        row["total_score"] = sum(row.values()) # 총점: 4개 지표의 합 (20점 만점)
        row["analysis"] = result["analysis"]
        return row
    except Exception as e:
        print(f"평가 중 오류 발생: {e}")
        return {"error": str(e)}

def evaluate_rag_with_llm(questions, contexts, predictions, labels):
    """LLM 평가 메트릭으로 RAG 답변 전체를 평가해서 데이터프레임으로 돌려준다."""
    ## 책은 한 건씩 차례로 호출하지만, 판정 모델이 추론(reasoning) 모델이라 한 건이 오래 걸려서 8건씩 동시에 호출한다
    with ThreadPoolExecutor(max_workers=8) as executor:
        rows = list(tqdm(
            executor.map(evaluate_one, questions, contexts, predictions, labels),
            total=len(questions),
            desc="RAG 평가 진행중",
        ))
    return pd.DataFrame(rows)

######################################################################
# 4. 정량 평가: 인용한 문서 번호의 F1

## 학습 데이터의 답변은 근거 문서를 [[ref1]], [[ref2]] 형태로 남기므로, 정답과 예측이 인용한 문서 번호를 비교할 수 있다.
def extract_ref_numbers(text: str) -> List[int]:
    """
    텍스트에서 [[ref숫자]] 패턴의 숫자들을 추출하여 중복 없이 정렬된 리스트로 반환합니다.

    Example:
        >>> extract_ref_numbers("이것은 [[ref1]]과 [[ref2]], [[ref1]]입니다.")
        [1, 2]
    """
    pattern = r'\[\[ref(\d+)\]\]'
    numbers = [int(match.group(1)) for match in re.finditer(pattern, text)]
    return sorted(list(set(numbers)))

def calculate_f1_score(labels, predictions):
    """
    F1 score 계산 (multilabel 케이스)
    하나의 답변이 여러 문서를 인용할 수 있으므로, 모든 샘플의 TP/FP/FN을 누적한 뒤 정밀도와 재현율을 계산한다.

    Args:
        labels: 샘플별 정답 문서 번호 리스트의 리스트
        predictions: 샘플별 예측 문서 번호 리스트의 리스트
    """
    if len(labels) != len(predictions):
        raise ValueError("라벨과 예측의 길이가 일치하지 않습니다.")

    total_true_positives = 0
    total_false_positives = 0
    total_false_negatives = 0

    for sample_labels, sample_preds in zip(labels, predictions):
        sample_labels = set(sample_labels)
        sample_preds = set(sample_preds)
        total_true_positives += len(sample_labels & sample_preds)
        total_false_positives += len(sample_preds - sample_labels)
        total_false_negatives += len(sample_labels - sample_preds)

    ## 특수 케이스: 정답과 예측이 모두 빈 리스트면 "인용할 문서가 없음을 올바르게 인식"한 것으로 보고 1.0
    if sum(bool(l) for l in labels) == 0 and sum(bool(p) for p in predictions) == 0:
        return 1.0

    precision = total_true_positives / (total_true_positives + total_false_positives) if (total_true_positives + total_false_positives) > 0 else 0
    recall = total_true_positives / (total_true_positives + total_false_negatives) if (total_true_positives + total_false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return f1

######################################################################
# 5. 베이스 모델 vs 파인튜닝 모델 비교

def main():
    data = load_test_data()
    print(f"평가 샘플 수: {len(data['prompt'])}")
    print(pd.Series(data["type"]).value_counts().to_string())

    print("\n=============================================")

    preds_by_model = {
        "base": generate_answers(BASE_MODEL_ID, data["prompt"]),
        "tuned": generate_answers(TUNED_MODEL_ID, data["prompt"]),
    }
    print(f"질문: {data['question'][0]}")
    print(f"베이스 모델: {preds_by_model['base'][0]}")
    print(f"파인튜닝 모델: {preds_by_model['tuned'][0]}")
    print(f"정답: {data['label'][0]}")

    print("\n=============================================")

    llm_eval_by_model = {
        model: evaluate_rag_with_llm(data["question"], data["context"], preds, data["label"])
        for model, preds in preds_by_model.items()
    }

    label_ref_numbers = [extract_ref_numbers(label) for label in data["label"]]
    f1_by_model = {
        model: calculate_f1_score(label_ref_numbers, [extract_ref_numbers(pred) for pred in preds])
        for model, preds in preds_by_model.items()
    }

    print("\n=============================================")

    ## 참고: 컨텍스트 관련성/충분성은 질문과 검색 결과만 보는 지표라(검색기 품질) 두 모델이 비슷하게 나오는 것이 정상이다.
    ##       모델 차이는 응답 정확성, 컨텍스트 충실성, 인용 F1에서 드러난다.
    metric_names = {
        "answer_correctness": "응답 정확성",
        "context_relevance": "컨텍스트 관련성",
        "context_faithfulness": "컨텍스트 충실성",
        "context_recall": "컨텍스트 충분성",
        "total_score": "평균 총점(20점 만점)",
    }
    summary = {}
    for model, results_df in llm_eval_by_model.items():
        summary[model] = {name: results_df[metric].mean() for metric, name in metric_names.items()}
        summary[model]["평가 실패 건수"] = results_df["error"].notna().sum() if "error" in results_df else 0
        summary[model]["인용 문서 F1"] = f1_by_model[model]

    print("===== 베이스 모델 vs 파인튜닝 모델 =====")
    print(pd.DataFrame(summary).round(3).to_string())

    ## 샘플별 결과(질문, 정답, 두 모델의 답변과 점수)를 CSV로 저장해서 따로 살펴볼 수 있게 한다
    detail_df = pd.DataFrame({"type": data["type"], "question": data["question"], "label": data["label"]})
    for model, preds in preds_by_model.items():
        detail_df[f"{model}_pred"] = preds
        detail_df = detail_df.join(llm_eval_by_model[model].add_prefix(f"{model}_"))
    detail_df.to_csv("kanana2_rag_evaluation_results.csv", index=False, encoding="utf-8-sig")
    print("\n샘플별 결과 저장: kanana2_rag_evaluation_results.csv")


if __name__ == "__main__":
    main()
