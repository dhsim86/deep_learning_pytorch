import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# 모델과 토크나이저 로드
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# 텍스트 인코딩 및 생성
input_ids = tokenizer.encode("Hello, what's your name?", return_tensors='pt')
output = model.generate(input_ids, max_length=30)

# output:  tensor([[15496,    11,   644,   338,   534,  1438,    30,   198,   198,    40,
#          1101,   257,  3516,   508,   338,   587,  2877,   287,   262,  1748,
#           329,   257,   981,    13,   314,  1101,   257,  3516,   508,   338]])
print("output: ", output)

# decoded output:  Hello, what's your name?
#
# I'm a guy who's been living in the city for a while. I'm a guy who's
print("decoded output: ", tokenizer.decode(output[0], skip_special_tokens=True))

print("\n=============================================")
print("한 forward pass에서 로짓 예측")

# 로짓 계산
output = model(input_ids)

# tensor([[15496,    11,   644,   338,   534,  1438,    30]])
print("input_ids: ", input_ids)
# tensor([[[ -35.2361,  -35.3263,  -38.9752,  ...,  -44.4643,  -43.9973,
#           -36.4578],
#         ...,
#         [-108.7915, -107.8531, -110.9524,  ..., -118.9040, -119.1501,
#          -106.8247]]], grad_fn=<UnsafeViewBackward0>)
print("output: ", output.logits)
# torch.Size([1, 7, 50257])
print("logit shape: ", output.logits.shape)

# 입력 토큰을 하나씩 디코딩 (input_ids[0]: 배치의 첫 번째 문장)
# input_ids 를 그대로 순회하면 문장 전체가 한 덩어리로 디코딩되므로 [0] 이 필요하다.
# ['Hello', ',', ' what', "'s", ' your', ' name', '?']
input_tokens = [tokenizer.decode(token_id) for token_id in input_ids[0].tolist()]
print(input_tokens)

# forward 한 번으로 "모든 위치"에서 다음 토큰을 동시에 예측한다.
# logits[i] = i번째 토큰까지 본 상태에서 다음에 올 토큰 후보들의 점수
logits = output.logits[0]
top1 = torch.topk(logits, k=1)
print("top1: ", top1)

# top1.indices: (7, 1) -> 위치별 1등 토큰 ID
predicted_ids = top1.indices[:, 0].tolist()

# 위치별로 "입력 토큰 -> 예측한 다음 토큰"을 나란히 출력
# pos | input token  | predicted next  | actual next
# 0   | 'Hello'      | ','             | ','
# 1   | ','          | ' I'            | ' what'
# 2   | ' what'      | "'s"            | "'s"
# 3   | "'s"         | ' up'           | ' your'
# 4   | ' your'      | ' name'         | ' name'
# 5   | ' name'      | '?"'            | '?'
# 6   | '?'          | '\n'            | -
#
# 앞 6개는 정답(actual next)이 이미 입력에 있어서 문장 생성에는 쓰지 않고,
# 마지막 위치(6)의 예측만 다음 토큰으로 선택된다. 이 '\n'(198)은 위 generate() 결과의
# 8번째 토큰(198)과 같다.
print(f"{'pos':<4}| {'input token':<13}| {'predicted next':<16}| actual next")
for i, input_token in enumerate(input_tokens):
    predicted = repr(tokenizer.decode(predicted_ids[i]))
    actual = repr(input_tokens[i + 1]) if i + 1 < len(input_tokens) else '-'
    print(f"{i:<4}| {input_token!r:<13}| {predicted:<16}| {actual}")