import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from transformers import pipeline

######################################################################
# 파인튜닝된 모델 로드
username = "raveas"
model_name = 'kanana-2-1.3b-instruct-rag-ko'

uploaded_model_id = f"{username}/{model_name}"
model = AutoModelForCausalLM.from_pretrained(
    uploaded_model_id,
    torch_dtype=torch.bfloat16,
    attn_implementation="sdpa",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(uploaded_model_id, trust_remote_code=True)

######################################################################
# 데이터셋으로부터 입력, 정답 샘플 생성
dataset = load_dataset("iamjoon/klue-mrc-ko-rag-dataset", split="train")

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


def format_data(sample):
    # RAG 문서 검색 결과를 문서1, 문서2... 형태로 포매팅
    search_result = "\n-----\n".join([f"문서{idx + 1}: {result}" for idx, result in enumerate(sample["search_result"])])

    # OpenAI format으로 변환
    return {
        "messages": [
            {
                "role": "system",
                "content": system_message.format(search_result=search_result),
            },
            {
                "role": "user",
                "content": sample["question"],
            },
            {
                "role": "assistant",
                "content": sample["answer"]
            },
        ],
    }
formatted_dataset = format_data(dataset[0])

ASSISTANT_HEADER = "<|im_start|>assistant\n<think>\n\n</think>\n\n"
RESPONSE_END = "<|im_end|>" # kanana-2 의 턴 종료 토큰 (= tokenizer.eos_token, id 128010)

text = tokenizer.apply_chat_template(formatted_dataset["messages"], tokenize=False, add_generation_prompt=False)

## assistant 헤더를 경계로 두 조각으로 나눈다 (assistant 턴이 1개뿐이므로 항상 정확히 2조각)
before_assistant, after_assistant = text.split(ASSISTANT_HEADER)

## 입력: 시스템 + 유저 프롬프트 + 생성 프롬프트
##       split 하면 경계 문자열 자체는 사라지므로 다시 붙여줘야 한다
prompt = before_assistant + ASSISTANT_HEADER

## 정답: 모델 응답 본문만 (뒤에 붙은 <|im_end|> 와 줄바꿈은 잘라낸다)
label = after_assistant.split(RESPONSE_END)[0]

######################################################################
# 추론 및 결과 확인

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
eos_token = tokenizer(RESPONSE_END, add_special_tokens=False)["input_ids"][0]

## 추론 메서드 정의
def test_inference(pipe, prompt):
    outputs = pipe(
        prompt,
        max_new_tokens=8192,                 # 정답 응답이 길어도 잘리지 않을 만큼
        eos_token_id=eos_token,              # 이 토큰이 나오면 생성 중단
        pad_token_id=tokenizer.pad_token_id, # 지정 안 하면 경고가 뜬다 (배치 1이라 실제 패딩은 없음)
        do_sample=False,                     # 그리디 디코딩. 매번 같은 결과가 나와야 두 모델을 비교할 수 있다
        add_special_tokens=False,            # [중요] kanana2_finance_news_analysis.py의 추론 메서드 부분 참조
        return_full_text=False,              # 프롬프트를 뺀 "새로 생성된 텍스트"만 받는다
    )
    return outputs[0]["generated_text"].strip()

# response:
# 북태평양 기단과 오호츠크해 기단이 만나 국내에 머무르는 기간은 약 4~5주입니다.
# 이 기간 동안 '장마전선'이 형성되어 한반도를 통과하며 여름철 강우량을 발생시킵니다. 
# 특히, 7월 말부터 8월 초까지 다양한 규모의 폭풍우와 함께 폭우가 쏟아지면서 한반도를 통과하는 데 3주도 채 걸리지 않는 경향이 있습니다 [[ref1]].
# label:
# 북태평양 기단과 오호츠크해 기단이 만나 형성되는 장마전선은 한반도에 약 한 달가량 머무릅니다. 
# 문서2에 따르면, 장마전선은 한반도 중남부를 오르내리며 약 한 달 동안 비를 뿌린다고 합니다. 
# 이는 최근 30년간의 평균치와도 일치하며, 중부지방의 장마기간은 평균 32일로 나타났습니다 [[ref2]].
print(f" response:\n{test_inference(pipe, prompt)}")
print(f" label:\n{label}")
print("-"*50)