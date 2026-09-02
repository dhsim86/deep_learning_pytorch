# GPT-2를 이용한 챗봇 파인튜닝

from transformers import AutoTokenizer, PreTrainedTokenizerFast, GPT2LMHeadModel

######################################################################
# 모델과 토크나이저 로드 및 특수 토큰 확인
## 한국어 GPT 중 하나인 'skt/kogpt2-base-v2'를 사용.

## GPT2LMHeadModel = GPT2Model(Transformer 본체) + LMHead(맨 마지막 Linear 층)
model = GPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2')

## 토크나이저 로드시 텍스트 생성 시작 토큰(bos_token), 텍스트 생성 종료 토큰(eos_token) 등 지정 가능
tokenizer = PreTrainedTokenizerFast.from_pretrained(
    'skt/kogpt2-base-v2',
    bos_token='</s>', eos_token='</s>', unk_token='<unk>',
    pad_token='<pad>', mask_token='<mask>')


## 특수 토큰 확인

## <usr> 및 <sys> 특수 토큰은 GPT로 대화 모델을 만들 경우 시스템 및 유저 프롬프트를 구분하기 위한 용도
print("bos token ID: ", tokenizer.bos_token_id)
print("eos token ID: ", tokenizer.eos_token_id)
print("pad token ID: ", tokenizer.pad_token_id)
print('-' * 10)
print("1번 토큰: ", tokenizer.decode(1))
print("2번 토큰: ", tokenizer.decode(2)) # <usr>, 유저 프롬프트
print("3번 토큰: ", tokenizer.decode(3))
print("4번 토큰: ", tokenizer.decode(4)) # <sys>, 시스템 프롬프트

print("\n=============================================")

######################################################################
#