from transformers import AutoTokenizer

# 모든 LLM들은 고유한 토크나이저가 존재

## LLama 모델을 한국어로 추가학습한 모델
model_id = "allganize/Llama-3-Alpha-Ko-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)

## 입력 텍스트를 토큰화 및 정수 인코딩
input_ids = tokenizer.encode('안녕하세요. 반갑습니다.')
print(input_ids) # [128000, 101193, 124409, 13, 64857, 115072, 39331, 13]

## 다시 원래 입력으로 복원
print(tokenizer.decode(input_ids)) # <|begin_of_text|>안녕하세요. 반갑습니다.
