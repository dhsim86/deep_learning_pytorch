# =====================================================================================================
# [데모] Instruction tuned 모델 인퍼런스 과정 - naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B
#
# instruction tuned 모델은 베이스 모델을 "모델마다 고유한 챗 템플릿"으로 감싼 대화 데이터로 학습한 모델입니다.
# 그래서 추론할 때도 반드시 같은 챗 템플릿으로 변환한 뒤 모델에 넣어야 합니다.
#
#   messages(OpenAI 형식) --[apply_chat_template]--> 템플릿 문자열 --[tokenizer]--> input_ids --[generate]--> 응답
#
# 같은 입력을 kanana2_instruct_inference.py 에서도 사용하므로, 두 파일을 나란히 놓고
# "같은 대화가 모델별로 어떻게 다르게 변환되는지" 비교해 보세요.
#
# [사전 준비] gated 모델이라 허깅페이스 로그인 + 라이선스 동의가 필요합니다.
#   1. https://huggingface.co/naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B 에서 동의
#   2. 터미널에서 .venv/bin/hf auth login
# =====================================================================================================
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

######################################################################
# 모델 / 토크나이저 로드

model_id = "naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B"

## LlamaForCausalLM 아키텍처라 transformers 에 내장되어 있다 (trust_remote_code 불필요)
tokenizer = AutoTokenizer.from_pretrained(model_id)

device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype=torch.bfloat16, # 모델 원본 정밀도
    attn_implementation="sdpa",
).to(device)

print("\n=============================================")

######################################################################
# 입력 메시지 (OpenAI 형식)

## system / user 는 kanana2_instruct_inference.py 와 동일하다.
## 다만 HyperCLOVAX 는 공식 사용 예제에서 맨 앞에 "tool_list" 역할을 넣는다. (도구 목록을 JSON 으로 넣는 자리이고, 도구가 없으면 빈 문자열)
## 템플릿이 자동으로 넣어주지 않으므로, 학습 때와 같은 형태가 되도록 공식 예제대로 직접 넣는다.
##   -> 같은 메시지를 kanana-2 템플릿에 넣으면 "TemplateError: Unexpected message role." 에러가 난다.
##      허용하는 role 부터가 모델마다 다르다는 뜻이다.
messages = [
    {"role": "tool_list", "content": ""},
    {"role": "system", "content": "당신은 친절한 AI 어시스턴트입니다. 질문에 간결하게 답변하세요."},
    {"role": "user", "content": "대한민국의 수도는 어디야?"},
]

######################################################################
# 1. 챗 템플릿 변환

## tokenize=False            : 정수 인코딩은 하지 않고 "템플릿이 적용된 문자열"만 받는다 (눈으로 확인용)
## add_generation_prompt=True : 맨 끝에 "이제 assistant 차례"라는 생성 프롬프트를 붙인다
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

## HyperCLOVAX 의 챗 템플릿 (ChatML 계열)
##   - 각 턴을 <|im_start|>{role}\n ... <|im_end|>\n 로 감싼다 (kanana-2 와 같은 뼈대)
##   - 생성 프롬프트는 <|im_start|>assistant\n 까지만 붙는다
##     -> kanana-2 처럼 빈 추론 블록(<think>\n\n</think>\n\n)을 붙이지 않는다. 애초에 <think> 토큰이 사전에 없다.
##   - 같은 ChatML 계열이라도 role 종류, 생성 프롬프트, 제어 토큰 id 가 모두 다르다.
##
## [참고] add_generation_prompt=False 로 주면 마지막 "<|im_start|>assistant\n" 이 빠진다.
##        이 예시로 실측하면 HyperCLOVAX 는 "<|im_start|>assistant\n" 헤더부터 스스로 생성했다.
##        (같은 실험에서 kanana-2 는 헤더 없이 바로 답을 썼다)
##        즉 "어떤 형식으로 답할지"를 모델이 추측하게 되므로 출력 형식이 모델마다/입력마다 흔들린다.
##        학습 때 본 형식 그대로 생성 프롬프트까지 붙여주는 것이 정석이다.

# <|im_start|>tool_list
# <|im_end|>
# <|im_start|>system
# 당신은 친절한 AI 어시스턴트입니다. 질문에 간결하게 답변하세요.<|im_end|>
# <|im_start|>user
# 대한민국의 수도는 어디야?<|im_end|>
# <|im_start|>assistant
print("챗 템플릿 변환 결과:")
print(text)

print("\n=============================================")

######################################################################
# 2. 정수 인코딩

## add_special_tokens=False
##   HyperCLOVAX 토크나이저는 기본값에서도 BOS 를 붙이지 않아 결과가 같다. (kanana-2 는 다르다 -> 해당 파일 참고)
##   그래도 템플릿 문자열을 토큰화할 때는 항상 False 로 두는 습관이 모델을 바꿔도 안전하다.
inputs = tokenizer(text, add_special_tokens=False, return_tensors="pt").to(device)

# 기본값        : [100272, 14506, 62, 1638, 198]  <- BOS 가 붙지 않는다
# special=False : [100272, 14506, 62, 1638, 198]
print("기본값        :", tokenizer(text)["input_ids"][:5])
print("special=False :", inputs["input_ids"][0].tolist()[:5])

## 제어 토큰은 문자열이 아니라 "토큰 1개"로 인코딩된다. 같은 <|im_start|> 라도 모델마다 id 가 다르다. (kanana-2 는 128009)
# {'<|im_start|>': 100272, '<|im_end|>': 100273, '<|endofturn|>': 100275, '<|stop|>': 100274}
print({t: tokenizer.convert_tokens_to_ids(t) for t in ["<|im_start|>", "<|im_end|>", "<|endofturn|>", "<|stop|>"]})

print("\n=============================================")

######################################################################
# 3. 추론

## [중요] 생성을 멈출 토큰을 직접 지정해야 한다
##   HyperCLOVAX 는 assistant 턴을 "응답<|im_end|><|endofturn|>" 형태로 끝내도록 학습되었다.
##   그런데 저장소에 generation_config.json 이 없어서 config.json 의 eos_token_id(100257, <|endoftext|>)가 쓰이고,
##   이 예시에서는 그 토큰이 끝까지 나오지 않았다. 실측하면 지정하지 않았을 때 아래처럼 max_new_tokens 까지 멈추지 않는다.
##     대한민국의 수도는 서울입니다.<|im_end|><|endofturn|><|endofturn|><|endofturn|>...
##   그래서 공식 예제(stop_strings=["<|endofturn|>", "<|stop|>"])와 같은 토큰에서 멈추도록 eos_token_id 를 준다.
stop_token_ids = tokenizer.convert_tokens_to_ids(["<|endofturn|>", "<|stop|>"])

outputs = model.generate(
    **inputs,
    max_new_tokens=256,
    do_sample=False, # 그리디 디코딩. 매번 같은 결과가 나와야 두 모델을 비교할 수 있다
    eos_token_id=stop_token_ids, # 이 토큰들 중 하나가 나오면 생성 중단
    pad_token_id=tokenizer.pad_token_id, # 지정 안 하면 경고가 뜬다 (배치 1이라 실제 패딩은 없음)
)

## 프롬프트 부분을 잘라내고 "새로 생성된 토큰"만 디코딩한다
generated_ids = outputs[0][inputs["input_ids"].shape[1]:]

# 대한민국의 수도는 서울입니다.<|im_end|><|endofturn|>
print("모델 출력 (특수 토큰 포함):")
print(tokenizer.decode(generated_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False))

# 대한민국의 수도는 서울입니다.
print("\n모델 응답:")
print(tokenizer.decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False))
