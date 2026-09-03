from transformers import AutoTokenizer

# OpenAI API 형태로 시스템 / 유저 프롬프트 정의
messages = [
    {"role": "system", "content": "당신은 인공지능 어시스턴트입니다. 묻는 말에 친절하고 정확하게 답변하세요."},
    {"role": "user", "content": "은행의 기준 금리에 대해서 설명해줘"}
]

model_id = "allganize/Llama-3-Alpha-Ko-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)

# 모델의 고유한 챗 탬플릿 형태로 변환
## tokenizer=False는 인코딩은 안 하고 챗 템플릿만 적용
template_messages = tokenizer.apply_chat_template(messages, tokenize=False)

## <|begin_of_text|> 이후로
## <|start_header_id|>system<|end_header_id|> 시스템 프롬프트 <|eot_id|>
## <|start_header_id|>user<|end_header_id|> 유저 프롬프트 <|eot_id|>
## 로 변환된다. (LLama-3의 챗 탬플릿 형태)

# <|begin_of_text|><|start_header_id|>system<|end_header_id|>
# 
# 당신은 인공지능 어시스턴트입니다. 묻는 말에 친절하고 정확하게 답변하세요.<|eot_id|><|start_header_id|>user<|end_header_id|>
# 
# 은행의 기준 금리에 대해서 설명해줘<|eot_id|>
print(template_messages)

# 챗 템플릿 형태로 변환 (정수 인코딩까지)
encodeds = tokenizer.apply_chat_template(messages)

# {
#   'input_ids': [128000, 128006, 9125, 128007, 271, 65895, 83628, 34804, 59777, 103896, 67119, 101139, 30426, 25941, 95252, 29726, 80052, 13, 100570, 119, 16969, 101264, 19954, 108280, 104834, 101360, 127923, 102893, 111964, 92245, 13, 128009, 128006, 882, 128007, 271, 34804, 101066, 21028, 111902, 104193, 111833, 122115, 114942, 34983, 59269, 246, 128009], 
#   'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
# }
print('템플릿 적용 및 정수 인코딩 후:\n', encodeds)

# <|begin_of_text|><|start_header_id|>system<|end_header_id|>
# 
# 당신은 인공지능 어시스턴트입니다. 묻는 말에 친절하고 정확하게 답변하세요.<|eot_id|><|start_header_id|>user<|end_header_id|>
# 
# 은행의 기준 금리에 대해서 설명해줘<|eot_id|>
print('템플릿 적용 및 정수 인코딩 결과를 복원:\n',tokenizer.decode((encodeds)))

print("\n=============================================")

######################################################################
# 생성 프롬프트 확인
encodeds = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
print(encodeds)

## <|start_header_id|>assistant<|end_header_id|>가 추가로 붙는다. (생성 프롬프트)

# <|begin_of_text|><|start_header_id|>system<|end_header_id|>
#
# 당신은 인공지능 어시스턴트입니다. 묻는 말에 친절하고 정확하게 답변하세요.<|eot_id|><|start_header_id|>user<|end_header_id|>
# 
# 은행의 기준 금리에 대해서 설명해줘<|eot_id|><|start_header_id|>assistant<|end_header_id|>
print('템플릿 적용 및 정수 인코딩 결과를 복원:\n',tokenizer.decode((encodeds)))
