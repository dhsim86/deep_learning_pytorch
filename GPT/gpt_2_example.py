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
