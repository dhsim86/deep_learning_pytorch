import torch

from transformers import BertForNextSentencePrediction
from transformers import AutoTokenizer

# 모델과 토크나이저 로드
## BertForNextSentencePrediction: 두 개의 문장이 이어지는 문장 관계인지 여부를 판단하는 BERT 구조를 로드
model = BertForNextSentencePrediction.from_pretrained('bert-base-uncased')
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# 입력 텍스트 준비
prompt = "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."
next_sentence = "pizza is eaten with the use of a knife and fork. In casual settings, however, it is cut into wedges to be eaten while held in the hand."

print(tokenizer.cls_token, ':', tokenizer.cls_token_id)
print(tokenizer.sep_token, ':' , tokenizer.sep_token_id)

encoding = tokenizer(prompt, next_sentence, return_tensors='pt')

## [CLS] 토큰(101)이 맨 앞에 추가됨
## [SEP] 토큰(102)이 첫 번째 문장 및 두 번째 뭍장 끝에 각각 추가됨
## token_type_ids를 보면 세그먼트 임베딩을 통해 문장을 구분하는 것을 확인할 수 있다.
# {
#   'input_ids': tensor([[  101,  1999,  3304,  1010, 10733,  2366,  1999,  5337, 10906,  1010,
#          2107,  2004,  2012,  1037,  4825,  1010,  2003,  3591,  4895, 14540,
#          6610,  2094,  1012,   102, 10733,  2003,  8828,  2007,  1996,  2224,
#          1997,  1037,  5442,  1998,  9292,  1012,  1999, 10017, 10906,  1010,
#          2174,  1010,  2009,  2003,  3013,  2046, 17632,  2015,  2000,  2022,
#          8828,  2096,  2218,  1999,  1996,  2192,  1012,   102]]), 
#   'token_type_ids': tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#         1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]),
#   'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#         1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
# }
print(encoding)

## 다시 디코딩 결과 확인
# [CLS] in italy, pizza served in formal settings, such as at a restaurant, 
# is presented unsliced. [SEP] pizza is eaten with the use of a knife and fork. 
# in casual settings, however, it is cut into wedges to be eaten while held in the hand. [SEP]
print(tokenizer.decode(encoding['input_ids'][0]))

# 다음 문장 예측
## 서로 이어지는 문장일 경우 0번 인덱스, 아니면 1번 인덱스로 레이블을 지정하여 학습됨

## 모델에 입력시 소프트맥스 함수를 지나기 전의 값을 리턴 (logits)
pred = model(encoding['input_ids'], token_type_ids=encoding['token_type_ids'])
probs = torch.nn.functional.softmax(pred.logits, dim=1)  # Softmax 적용하여 확률 얻기

## tensor([[1.0000e+00, 2.8382e-06]], grad_fn=<SoftmaxBackward0>)
print(probs)

next_sentence_label = torch.argmax(probs, dim=1).item()  # 예측된 라벨 얻기
print('최종 예측 레이블 :', next_sentence_label) # 0


## 상관없는 두 개의 문장
prompt = "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."
next_sentence = "The sky is blue due to the shorter wavelength of blue light."
encoding = tokenizer(prompt, next_sentence, return_tensors='pt')

pred = model(encoding['input_ids'], token_type_ids=encoding['token_type_ids'])
probs = torch.nn.functional.softmax(pred.logits, dim=1)  # Softmax 적용하여 확률 얻기

## tensor([[1.2606e-04, 9.9987e-01]], grad_fn=<SoftmaxBackward0>)
print(probs)

next_sentence_label = torch.argmax(probs, dim=1).item()  # 예측된 라벨 얻기
print('최종 예측 레이블 :', next_sentence_label) # 1