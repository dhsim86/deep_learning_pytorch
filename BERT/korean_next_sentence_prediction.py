import torch
from transformers import BertForNextSentencePrediction
from transformers import AutoTokenizer

model = BertForNextSentencePrediction.from_pretrained('klue/bert-base')
tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")

# 이어지는 두 개의 문장 확인
prompt = "2002년 월드컵 축구대회는 일본과 공동으로 개최되었던 세계적인 큰 잔치입니다."
next_sentence = "여행을 가보니 한국의 2002년 월드컵 축구대회의 준비는 완벽했습니다."
encoding = tokenizer(prompt, next_sentence, return_tensors='pt')

pred = model(encoding['input_ids'], token_type_ids=encoding['token_type_ids'])
probs = torch.nn.functional.softmax(pred.logits, dim=1)  # Softmax 적용하여 확률 얻기

# tensor([[9.9989e-01, 1.1219e-04]], grad_fn=<SoftmaxBackward0>)
print(probs)

next_sentence_label = torch.argmax(probs, dim=1).item()  # 예측된 라벨 얻기
print('최종 예측 레이블 :', next_sentence_label) # 0

# 상관없는 두 개의 문장
prompt = "2002년 월드컵 축구대회는 일본과 공동으로 개최되었던 세계적인 큰 잔치입니다."
next_sentence = "극장가서 로맨스 영화를 보고싶어요"
encoding = tokenizer(prompt, next_sentence, return_tensors='pt')

pred = model(encoding['input_ids'], token_type_ids=encoding['token_type_ids'])
probs = torch.nn.functional.softmax(pred.logits, dim=1)  # Softmax 적용하여 확률 얻기

# tensor([[3.5087e-04, 9.9965e-01]], grad_fn=<SoftmaxBackward0>)
print(probs)

next_sentence_label = torch.argmax(probs, dim=1).item()  # 예측된 라벨 얻기
print('최종 예측 레이블 :', next_sentence_label) # 1
