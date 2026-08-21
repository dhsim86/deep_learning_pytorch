from transformers import GPT2LMHeadModel, GPT2Tokenizer

# 모델과 토크나이저 로드
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# 텍스트 인코딩 및 생성
input_ids = tokenizer.encode("Hello, what's your name?", return_tensors='pt')

# I'm a guy who's been living in the city for a while. I'm a guy who's
output = model.generate(input_ids, max_length=30)

print(output)
print(tokenizer.decode(output[0], skip_special_tokens=True))
