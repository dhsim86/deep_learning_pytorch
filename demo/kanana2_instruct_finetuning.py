# =====================================================================================================
# [데모] Instruction tuned 모델 LoRA 파인튜닝 - kakaocorp/kanana-2-1.3b-instruct  (챗 템플릿 사용)
#
# instruction tuned 모델은 베이스 모델을 "모델 고유의 챗 템플릿"으로 감싼 대화 데이터로 학습한 모델입니다.
# 그래서 이 모델을 추가로 파인튜닝할 때도 반드시 같은 챗 템플릿으로 데이터를 만들어야 하고,
# 추론할 때도 같은 템플릿을 써야 학습 효과가 그대로 나타납니다.
#
#   입력 : <|im_start|>user\n12시 땡!<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n하루가 또 가네요.<|im_end|>\n
#   정답 : assistant 응답 부분만 ("하루가 또 가네요.<|im_end|>\n"), 나머지는 -100 으로 loss 에서 제외
#
# [오해하기 쉬운 점] 이것도 결국 "다음 토큰 예측"입니다.
#   kanana2_base_finetuning.py 와 학습 방식(next token prediction) 자체는 같고, 달라지는 것은 두 가지뿐입니다.
#     (1) 입력 형식 : 사람이 정한 평문  vs  모델 고유의 챗 템플릿
#     (2) loss 범위 : 문장 전체         vs  assistant 응답 구간만
#
# kanana2_base_finetuning.py 와 같은 데이터 / 같은 뼈대로 작성했습니다.
# 모델 ID 를 빼면 두 파일의 코드 차이는 "2. 학습 데이터 포맷"과 "4. 추론 프롬프트" 두 곳뿐이니 나란히 놓고 비교해 보세요.
#
# [실행 환경] M1 Pro(32GB, MPS) 실측: 학습 125 step 191초 (스크립트 전체 약 3분 40초), MPS 메모리 6.0GB
# =====================================================================================================
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

model_id = "kakaocorp/kanana-2-1.3b-instruct"

######################################################################
# 1. 데이터 준비

## 한국어 챗봇 데이터 (11,823쌍, 짧은 일상 대화. GPT/chatbot_finetuning.py 에서 쓰던 것과 같은 데이터)
## 데모용이라 1,000개만 학습하고, 학습에 쓰지 않은 3개로 학습 전/후를 비교한다.
# {'Q': '12시 땡!', 'A': '하루가 또 가네요.', 'label': '0'}
dataset = load_dataset("songys/ChatbotData", split="train").shuffle(seed=42)
train_dataset = dataset.select(range(1000))
test_dataset = dataset.select(range(1000, 1003))

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

## instruct 모델에는 모델 고유의 챗 템플릿이 들어 있다 (chat_template.jinja)
print("chat_template 존재 여부:", tokenizer.chat_template is not None) # True

print("\n=============================================")

######################################################################
# 2. 학습 데이터 포맷 (챗 템플릿) + 정답(labels)

## [instruct 모델 학습의 핵심]
## 데이터를 OpenAI 형식(messages)으로 만들고, 문자열 변환은 반드시 모델의 apply_chat_template 에 맡긴다.
## 직접 "<|im_start|>user\n..." 문자열을 조립하면 <think> 블록처럼 모델마다 다른 세부 형식을 놓치기 쉽다.
##
## return_assistant_tokens_mask=True
##   kanana-2 의 템플릿에는 assistant 응답을 {% generation %} ... {% endgeneration %} 으로 표시해 두어서,
##   "어느 토큰이 assistant 응답인지"를 1/0 마스크로 바로 받을 수 있다. 1 인 위치만 정답으로 남기고 나머지는 -100.
##   -> 생성 프롬프트에 해당하는 "<|im_start|>assistant\n<think>\n\n</think>\n\n" 은 마스크 0 이라 loss 에서 빠진다.
##      즉 추론 때 템플릿이 붙여주는 부분은 맞힐 필요가 없고, 그 뒤의 응답만 학습한다.
##   -> 템플릿에 {% generation %} 표시가 없는 모델(예: HyperCLOVAX)은 마스크가 전부 0 으로 나온다.
##      그때는 RAG_finetuning/kanana2_rag_finetuning.py 의 tokenize_with_assistant_labels 처럼
##      "assistant 를 뺀 프롬프트의 길이"로 응답 시작 위치를 구하면 된다.
def format_chat(sample):
    messages = [
        {"role": "user", "content": sample["Q"]},
        {"role": "assistant", "content": sample["A"]},
    ]
    encoded = tokenizer.apply_chat_template(
        messages,
        return_dict=True,
        return_assistant_tokens_mask=True,
    )
    input_ids = encoded["input_ids"]
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": [token_id if is_assistant else -100 for token_id, is_assistant in zip(input_ids, encoded["assistant_masks"])],
    }

train_dataset = train_dataset.map(format_chat, remove_columns=train_dataset.column_names)

## 학습 샘플 확인 (BOS 없이 <|im_start|> 로 시작한다. 추론 때 add_special_tokens=False 를 주는 이유)
# <|im_start|>user
# 딱 내 취향인데<|im_end|>
# <|im_start|>assistant
# <think>
#
# </think>
#
# 용기를 내보세요.<|im_end|>
#
print("input_ids 디코딩:")
print(tokenizer.decode(train_dataset[0]["input_ids"]))

## loss 에 들어가는 구간 = assistant 응답만 (같은 샘플인 kanana2_base_finetuning.py 는 문장 전체였다)
# 용기를 내보세요.<|im_end|>
#
print("\nlabels 디코딩 (-100 제외):")
print(tokenizer.decode([i for i in train_dataset[0]["labels"] if i != -100]))

## [템플릿 없이 평문으로 학습하면?]  (이 데모에서는 실행하지 않음)
##   kanana2_base_finetuning.py 처럼 "질문: ...\n답변: ..." 평문으로 학습하면 모델이 원래 알던 대화 형식과 다른 형식을 새로 배우게 된다.
##   그러면 추론 때 (a) 챗 템플릿을 쓰면 학습한 형식과 달라 효과가 제대로 나타나지 않고,
##   (b) 평문을 쓰면 system 프롬프트 / 멀티턴 등 템플릿 기반으로 익힌 대화 능력을 활용하기 어렵다.
##   instruction tuning 에 들인 학습을 그대로 살리려면 같은 템플릿으로 이어서 학습하는 것이 정석이다.

print("\n=============================================")

######################################################################
# 3. 모델 로드

## M1 Pro GPU 에는 bfloat16 연산 유닛이 없어서 fp16 이 약 2배 빠르다 (LLM_finetuning/kanana2_finance_news_analysis.py 참고)
## CUDA 에서는 bf16 이 정석이다.
use_mps = torch.backends.mps.is_available()

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype=torch.float16 if use_mps else torch.bfloat16,
    attn_implementation="sdpa",
    trust_remote_code=True, # Kanana2TinyForCausalLM 은 transformers 내장 아키텍처가 아니다
)
device = "cuda" if torch.cuda.is_available() else "mps" if use_mps else "cpu"
model.to(device)

######################################################################
# 4. 학습 전 추론

## 추론할 때도 "학습 때와 같은 챗 템플릿"으로 프롬프트를 만들고, 생성 프롬프트(assistant 차례)까지 붙인다
def build_prompt(question):
    messages = [{"role": "user", "content": question}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def generate(model, question):
    inputs = tokenizer(build_prompt(question), add_special_tokens=False, return_tensors="pt").to(model.device) # 템플릿에 없는 BOS 가 붙지 않도록
    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=False, # 그리디 디코딩. 학습 전/후를 같은 조건에서 비교하기 위함
        pad_token_id=tokenizer.pad_token_id,
    )
    return tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=False, clean_up_tokenization_spaces=False)

before = [generate(model, q) for q in test_dataset["Q"]]

print("\n=============================================")

######################################################################
# 5. LoRA 학습

peft_config = LoraConfig(
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    bias="none",
    target_modules=["q_proj", "v_proj"], # 학습 대상 파라미터 1,966,080개 (전체의 0.152%)
    task_type="CAUSAL_LM",
)

args = SFTConfig(
    output_dir="kanana2-1.3b-instruct-chatbot-lora", # save_strategy="no" 라 빈 디렉터리만 생기고 체크포인트는 저장되지 않는다
    num_train_epochs=1,
    per_device_train_batch_size=8,
    learning_rate=2e-4,
    lr_scheduler_type="constant",
    logging_steps=10,

    ## [중요] fused 옵티마이저를 쓰지 않는다
    ## MPS + fp16 + adamw_torch_fused 조합은 fp16 오버플로가 난 스텝을 건너뛰지 못하고 그대로 반영해서
    ## LoRA 가중치가 NaN 이 된다 (torch 2.12.1 에서 실측. 진단 실행에서는 4번째 스텝에서 발생했고, 이후 출력이 "!!!!..." 로 망가졌다).
    ## 비-fused 인 adamw_torch 는 정상적으로 건너뛴다.
    optim="adamw_torch",
    fp16=use_mps,     # 맥북(MPS)은 fp16
    bf16=not use_mps, # CUDA / CPU 는 bf16

    save_strategy="no", # 데모라 저장하지 않는다. 어댑터를 저장하려면 학습 후 trainer.save_model() 호출
    report_to="none",
    remove_unused_columns=False,                   # 직접 만든 input_ids / labels 컬럼 유지
    dataset_kwargs={"skip_prepare_dataset": True}, # 이미 토큰화했으므로 TRL 의 자동 전처리 OFF
)

## [메모리] 이 데이터는 템플릿을 씌워도 샘플당 최대 52토큰이라 배치 8 이어도 메모리 부담이 거의 없다
trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100), # 패딩 자리만 labels -100
    peft_config=peft_config,
)
trainer.train()

print("\n=============================================")

######################################################################
# 6. 학습 후 추론 및 비교

trainer.model.eval()
after = [generate(trainer.model, q) for q in test_dataset["Q"]]

## 실측 결과 (loss 3.66 -> 2.27)
##   학습 전 : 이미 대화 형식은 알고 있어서 자연스럽게 답하지만, 길고 설명적인 범용 어시스턴트 말투다 (50토큰에서 잘림)
##            -> 베이스 모델(형식도, 끝내는 법도 모름)과 출발점이 다르다. 형식은 instruction tuning 때 이미 배웠다
##   학습 후 : 데이터처럼 짧은 한 문장으로 답하고 <|im_end|> 에서 멈춘다
##            -> 같은 템플릿으로 학습했기 때문에, 원래 알던 대화 형식 위에 "말투"만 새로 얹혔다
#
# 질문     : 오래 연애하고 싶어.
# 정답     : 진짜 사랑하면 가능합니다.
# 학습 전  : '정말 멋진 마음이야. 오래 연애하고 싶다는 건, 그만큼 서로를 진심으로 아끼고, 함께 성장하고 싶다는 뜻이지. ...'
# 학습 후  : '오래 연애하는 것도 좋지만, 지금 연애하는 것도 좋아요.<|im_end|>'
# --------------------------------------------------
# 질문     : 올해 왜 이러지
# 정답     : 내년이라고 별 수 있겠어요.
# 학습 전  : '"올해 왜 이러지?"라는 말은 여러 가지 의미로 해석될 수 있습니다.  \n예를 들어:\n\n1. **날씨, 경제, 사회, 건강 등 ...'
# 학습 후  : '올해가 그런 것 같아요.<|im_end|>'
# --------------------------------------------------
# 질문     : 요즘 자꾸 깜빡깜빡해
# 정답     : 메모하는 습관을 들여보세요.
# 학습 전  : '요즘 자꾸 깜빡깜빡하시는 것 같아요. 이런 증상은 스트레스, 수면 부족, 피로, 나이, 혹은 일시적인 집중력 저하 등 ...'
# 학습 후  : '기억력이 감퇴하는 시기인가봐요.<|im_end|>'
for question, answer, b, a in zip(test_dataset["Q"], test_dataset["A"], before, after):
    print(f"질문     : {question}")
    print(f"정답     : {answer}")
    print(f"학습 전  : {b!r}")
    print(f"학습 후  : {a!r}")
    print("-" * 50)
