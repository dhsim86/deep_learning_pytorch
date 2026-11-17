# =====================================================================================================
# [데모] 베이스 모델 LoRA 파인튜닝 - kakaocorp/kanana-2-1.3b-base  (챗 템플릿 없음)
#
# 베이스 모델은 사전학습만 끝난 모델이라 "대화 형식"을 모릅니다. 챗 템플릿도 없습니다.
# 그래서 원하는 형식의 "평문"을 만들어 그대로 다음 토큰 예측(next token prediction)으로 학습합니다.
#
#   입력 : <|begin_of_text|>질문: 12시 땡!\n답변: 하루가 또 가네요.<|end_of_text|>
#   정답 : 입력 전체 (모든 토큰이 loss 계산에 들어감)
#
# kanana2_instruct_finetuning.py 와 같은 데이터 / 같은 뼈대로 작성했습니다.
# 모델 ID 를 빼면 두 파일의 코드 차이는 "2. 학습 데이터 포맷"과 "4. 추론 프롬프트" 두 곳뿐이니 나란히 놓고 비교해 보세요.
#
# [실행 환경] M1 Pro(32GB, MPS) 실측: 학습 125 step 167초 (스크립트 전체 약 3분 20초), MPS 메모리 5.9GB
# =====================================================================================================
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

model_id = "kakaocorp/kanana-2-1.3b-base"

######################################################################
# 1. 데이터 준비

## 한국어 챗봇 데이터 (11,823쌍, 짧은 일상 대화. GPT/chatbot_finetuning.py 에서 쓰던 것과 같은 데이터)
## 데모용이라 1,000개만 학습하고, 학습에 쓰지 않은 3개로 학습 전/후를 비교한다.
# {'Q': '12시 땡!', 'A': '하루가 또 가네요.', 'label': '0'}
dataset = load_dataset("songys/ChatbotData", split="train").shuffle(seed=42)
train_dataset = dataset.select(range(1000))
test_dataset = dataset.select(range(1000, 1003))

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

## 베이스 모델에는 챗 템플릿이 없다 -> apply_chat_template 을 호출하면 에러가 난다
print("chat_template 존재 여부:", tokenizer.chat_template is not None) # False

print("\n=============================================")

######################################################################
# 2. 학습 데이터 포맷 (평문) + 정답(labels)

## [베이스 모델 학습의 핵심]
## 사람이 정한 평문 형식("질문: ...\n답변: ...")을 그대로 이어 붙이고, 문장 전체를 다음 토큰 예측으로 학습한다.
##   - labels = input_ids : 입력 자체가 정답이다 (한 칸 미는 shift 는 모델 내부에서 처리, GPT/chatbot_finetuning.py 참고)
##   - 끝에 eos(<|end_of_text|>)를 직접 붙인다 : 토크나이저가 BOS 는 자동으로 붙여주지만 EOS 는 붙여주지 않는다.
##     EOS 를 학습시키지 않으면 모델이 "답변을 언제 끝낼지" 배우지 못해 멈추지 않고 계속 생성한다.
def format_base(sample):
    text = f"질문: {sample['Q']}\n답변: {sample['A']}"
    input_ids = tokenizer(text)["input_ids"] + [tokenizer.eos_token_id]
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": list(input_ids), # 전체 시퀀스가 정답
    }

train_dataset = train_dataset.map(format_base, remove_columns=train_dataset.column_names)

## 학습 샘플 확인
# <|begin_of_text|>질문: 딱 내 취향인데
# 답변: 용기를 내보세요.<|end_of_text|>
print("input_ids 디코딩:")
print(tokenizer.decode(train_dataset[0]["input_ids"]))

## loss 에 들어가는 구간 = 입력 전체
# <|begin_of_text|>질문: 딱 내 취향인데
# 답변: 용기를 내보세요.<|end_of_text|>
print("\nlabels 디코딩 (-100 제외):")
print(tokenizer.decode([i for i in train_dataset[0]["labels"] if i != -100]))

## [!! 함정] kanana-2 base 는 pad_token 과 eos_token 이 같은 토큰(<|end_of_text|>, 128001)이다.
##   GPT/chatbot_finetuning.py 처럼 labels[labels == pad_token_id] = -100 으로 패딩을 지우면
##   정답 끝의 EOS 까지 함께 지워져서, 모델이 끝내는 법을 배우지 못한다.
##   그래서 아래 DataCollatorForSeq2Seq 처럼 "패딩으로 늘어난 자리만" -100 으로 채우는 방식을 쓴다.
print("\npad_token_id:", tokenizer.pad_token_id, "/ eos_token_id:", tokenizer.eos_token_id) # 128001 / 128001

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

## 추론할 때도 "학습 때와 같은 평문 형식"으로 프롬프트를 만들고, 답변 앞까지만 넣는다
def build_prompt(question):
    return f"질문: {question}\n답변:"

def generate(model, question):
    inputs = tokenizer(build_prompt(question), return_tensors="pt").to(model.device) # 베이스는 평문이라 BOS 자동 추가가 학습과 동일
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
    output_dir="kanana2-1.3b-base-chatbot-lora", # save_strategy="no" 라 빈 디렉터리만 생기고 체크포인트는 저장되지 않는다
    num_train_epochs=1,
    per_device_train_batch_size=8,
    learning_rate=2e-4,
    lr_scheduler_type="constant",
    logging_steps=10,

    ## [중요] fused 옵티마이저를 쓰지 않는다
    ## MPS + fp16 + adamw_torch_fused 조합은 fp16 오버플로가 난 스텝을 건너뛰지 못하고 그대로 반영해서
    ## LoRA 가중치가 NaN 이 된다 (torch 2.12.1 에서 실측). 비-fused 인 adamw_torch 는 정상적으로 건너뛴다.
    optim="adamw_torch",
    fp16=use_mps,     # 맥북(MPS)은 fp16
    bf16=not use_mps, # CUDA / CPU 는 bf16

    save_strategy="no", # 데모라 저장하지 않는다. 어댑터를 저장하려면 학습 후 trainer.save_model() 호출
    report_to="none",
    remove_unused_columns=False,                   # 직접 만든 input_ids / labels 컬럼 유지
    dataset_kwargs={"skip_prepare_dataset": True}, # 이미 토큰화했으므로 TRL 의 자동 전처리 OFF
)

## [메모리] 이 데이터는 샘플당 최대 44토큰이라 배치 8 이어도 메모리 부담이 거의 없다
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

## 실측 결과 (loss 3.72 -> 2.51)
##   학습 전 : 질문을 반복하거나 엉뚱한 글을 이어 쓰며 max_new_tokens 까지 멈추지 않는다
##            -> 베이스 모델은 "질문/답변" 형식도, 어디서 끝내야 하는지도 모른다
##   학습 후 : 데이터와 비슷한 짧은 답을 하고 <|end_of_text|> 에서 스스로 멈춘다
##            -> 형식과 EOS 를 학습했다. 챗 템플릿 없이도 "평문 형식"만으로 학습된다
#
# 질문     : 오래 연애하고 싶어.
# 정답     : 진짜 사랑하면 가능합니다.
# 학습 전  : ' 오래 연애하고 싶어. 1. 오래 연애하고 싶어. 2. 오래 연애하고 싶어. 3. 오래 연애하고 싶어. ...'
# 학습 후  : ' 오래 연애할 수 있을 만큼 사랑하세요.<|end_of_text|>'
# --------------------------------------------------
# 질문     : 올해 왜 이러지
# 정답     : 내년이라고 별 수 있겠어요.
# 학습 전  : ' 1. 2020년 1월 1일, 12:00:00\n안녕하세요, 2020년 1월 1일, 12:00:00에 질문이 올라왔습니다. ...'
# 학습 후  : ' 올해는 더 좋은 일이 생길 거예요.<|end_of_text|>'
# --------------------------------------------------
# 질문     : 요즘 자꾸 깜빡깜빡해
# 정답     : 메모하는 습관을 들여보세요.
# 학습 전  : ' 요즘 자꾸 깜빡깜빡해. 1. 요즘 자꾸 깜빡깜빡해. 요즘 자꾸 깜빡깜빡해. ...'
# 학습 후  : ' 자꾸 잊어버리는 건 그만큼 중요한 일이 있다는 거겠죠.<|end_of_text|>'
for question, answer, b, a in zip(test_dataset["Q"], test_dataset["A"], before, after):
    print(f"질문     : {question}")
    print(f"정답     : {answer}")
    print(f"학습 전  : {b!r}")
    print(f"학습 후  : {a!r}")
    print("-" * 50)
