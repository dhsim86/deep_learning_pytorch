# =====================================================================================================
# [데모] Instruction tuned 모델 인퍼런스 과정 - kakaocorp/kanana-2-1.3b-instruct
#
# instruction tuned 모델은 베이스 모델을 "모델마다 고유한 챗 템플릿"으로 감싼 대화 데이터로 학습한 모델입니다.
# 그래서 추론할 때도 반드시 같은 챗 템플릿으로 변환한 뒤 모델에 넣어야 합니다.
#
#   messages(OpenAI 형식) --[apply_chat_template]--> 템플릿 문자열 --[tokenizer]--> input_ids --[generate]--> 응답
#
# 같은 입력을 hyperclovax_instruct_inference.py 에서도 사용하므로, 두 파일을 나란히 놓고
# "같은 대화가 모델별로 어떻게 다르게 변환되는지" 비교해 보세요.
# =====================================================================================================
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

######################################################################
# 모델 / 토크나이저 로드

model_id = "kakaocorp/kanana-2-1.3b-instruct"

## Kanana2TinyForCausalLM 은 transformers 내장 아키텍처가 아니라서 trust_remote_code=True 가 필요하다
## (허깅페이스 저장소의 modeling_kanana2_tiny.py 를 내려받아 실행)
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype=torch.bfloat16, # 모델 원본 정밀도
    attn_implementation="sdpa",
    trust_remote_code=True,
).to(device)

print("\n=============================================")

######################################################################
# 입력 메시지 (OpenAI 형식)

## 어떤 모델이든 "역할(role) + 내용(content)" 리스트로 대화를 표현하는 것은 같다.
## 이것을 모델 고유의 문자열로 바꿔주는 것이 챗 템플릿이다.
messages = [
    {"role": "system", "content": "당신은 친절한 AI 어시스턴트입니다. 질문에 간결하게 답변하세요."},
    {"role": "user", "content": "대한민국의 수도는 어디야?"},
]

######################################################################
# 1. 챗 템플릿 변환

## tokenize=False            : 정수 인코딩은 하지 않고 "템플릿이 적용된 문자열"만 받는다 (눈으로 확인용)
## add_generation_prompt=True : 맨 끝에 "이제 assistant 차례"라는 생성 프롬프트를 붙인다
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

## kanana-2 의 챗 템플릿 (ChatML 계열)
##   - 각 턴을 <|im_start|>{role}\n ... <|im_end|>\n 로 감싼다
##   - 생성 프롬프트로 <|im_start|>assistant\n 뒤에 "빈 추론 블록" <think>\n\n</think>\n\n 까지 붙인다
##     -> kanana-2 는 추론(thinking) 모델이라 템플릿에 thinking_mode 옵션이 있고, 기본값 "no_think" 이면
##        "생각은 비워두고 바로 답하라"는 표시로 빈 블록을 미리 채워 넣는다.
##        (apply_chat_template(..., thinking_mode="think") 로 주면 "<think>\n" 까지만 붙어 추론을 먼저 생성한다)
##   - HyperCLOVAX 에는 없는 kanana-2 만의 형식이다. 같은 ChatML 계열이라도 세부는 모델마다 다르다.
##
## [참고] add_generation_prompt=False 로 주면 마지막 "<|im_start|>assistant\n<think>\n\n</think>\n\n" 이 빠진다.
##        이 예시로 실측하면 kanana-2 는 assistant 헤더와 <think> 블록 없이 바로 답을 썼다.
##        (같은 실험에서 HyperCLOVAX 는 "<|im_start|>assistant\n" 헤더부터 스스로 생성했다)
##        즉 "어떤 형식으로 답할지"를 모델이 추측하게 되므로 출력 형식이 모델마다/입력마다 흔들린다.
##        학습 때 본 형식 그대로 생성 프롬프트까지 붙여주는 것이 정석이다.

# <|im_start|>system
# 당신은 친절한 AI 어시스턴트입니다. 질문에 간결하게 답변하세요.<|im_end|>
# <|im_start|>user
# 대한민국의 수도는 어디야?<|im_end|>
# <|im_start|>assistant
# <think>
#
# </think>
#
print("챗 템플릿 변환 결과:")
print(text)

print("\n=============================================")

######################################################################
# 2. 정수 인코딩

## [중요] add_special_tokens=False
##   템플릿 문자열에는 <|im_start|> 같은 제어 토큰이 이미 다 들어 있다.
##   그런데 kanana-2 토크나이저는 기본값(add_special_tokens=True)에서 맨 앞에 BOS(<|begin_of_text|>, 128000)를 하나 더 붙인다.
##   apply_chat_template 의 결과(=학습 때 모델이 본 형태)에는 BOS 가 없으므로, 추론 때 붙이면 학습 때와 입력이 달라진다.
##   (이 짧은 예시에서는 응답이 같았지만, 입력을 학습 때와 똑같이 맞추는 습관이 안전하다)
inputs = tokenizer(text, add_special_tokens=False, return_tensors="pt").to(device)

# 기본값        : [128000, 128009, 10594, 198, 51176, ...]  <- 맨 앞에 BOS 가 붙는다
# special=False : [128009, 10594, 198, 51176, 68168, ...]   <- 템플릿 그대로
print("기본값        :", tokenizer(text)["input_ids"][:5])
print("special=False :", inputs["input_ids"][0].tolist()[:5])

## 제어 토큰은 문자열이 아니라 "토큰 1개"로 인코딩된다
# {'<|im_start|>': 128009, '<|im_end|>': 128010, '<think>': 128032, '</think>': 128033}
print({t: tokenizer.convert_tokens_to_ids(t) for t in ["<|im_start|>", "<|im_end|>", "<think>", "</think>"]})

print("\n=============================================")

######################################################################
# 3. 추론

## 언제 생성을 멈출지(eos)도 모델마다 다르다.
## kanana-2 는 assistant 턴을 <|im_end|>(128010) 로 끝내도록 학습되었고, generation_config.json 의 eos_token_id 도
## 128010 으로 들어 있어서 따로 지정하지 않아도 멈춘다. (HyperCLOVAX 는 직접 지정해야 한다 -> 해당 파일 참고)
outputs = model.generate(
    **inputs,
    max_new_tokens=256,
    do_sample=False, # 그리디 디코딩. 매번 같은 결과가 나와야 두 모델을 비교할 수 있다
)

## 프롬프트 부분을 잘라내고 "새로 생성된 토큰"만 디코딩한다
generated_ids = outputs[0][inputs["input_ids"].shape[1]:]

# 대한민국의 수도는 서울입니다.<|im_end|>
print("모델 출력 (특수 토큰 포함):")
print(tokenizer.decode(generated_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False))

# 대한민국의 수도는 서울입니다.
print("\n모델 응답:")
print(tokenizer.decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False))
