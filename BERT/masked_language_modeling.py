from transformers import BertForMaskedLM
from transformers import AutoTokenizer

# 모델과 토크나이저 로드
## 모델과 토크나이저는 같은 모델을 사용해야 한다.
## 다른 BERT끼리 쓰면 모델이 텍스트를 제대로 이해하지 못한다. (정수 인코딩 결과가 서로 다를 수 있음)

## BertForMaskedLM: [MASK]로 가려진 원래 단어를 맞추기 위한 
##  마스크드 언어 모델링을 위한 구조로 BERT를 로드
model = BertForMaskedLM.from_pretrained('bert-large-uncased')
tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased")

## 토크나이저로 정수인코딩 결과 확인
inputs = tokenizer('Soccer is a really fun [MASK].')

## input_ids: 토큰별 정수 인코딩 결과
## token_type_ids: 문장을 구분하는 세그먼트 인코딩 결과
## attention_mask: 패딩 여부를 구분하는 어텐션 마스크
# {
#   'input_ids': [101, 4715, 2003, 1037, 2428, 4569, 103, 1012, 102], 
#   'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0], 
#   'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1]
# }
print(inputs)

# [MASK] 토큰 예측
## FillMaskPipeline: 마스크드 언어 모델링의 예측 결과를 정리해서 보여준다.
from transformers import FillMaskPipeline
pip = FillMaskPipeline(model=model, tokenizer=tokenizer)

## [MASK]로 가려진 원래 단어를 예측한 결과를 보여준다.
print(pip('Soccer is a really fun [MASK].'))
print(pip('The Avengers is a really fun [MASK].'))
print(pip('I went to [MASK] this morning.'))
