from transformers import BertForMaskedLM
from transformers import AutoTokenizer

# 한국어 BERT 모델 및 토크나이저 로드
model = BertForMaskedLM.from_pretrained('klue/bert-base')
tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")

inputs = tokenizer('축구는 정말 재미있는 [MASK]다.')

# {
#   'input_ids': [2, 4713, 2259, 3944, 6001, 2259, 4, 809, 18, 3], 
#   'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
#   'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
# }
print(inputs)

from transformers import FillMaskPipeline
pip = FillMaskPipeline(model=model, tokenizer=tokenizer)

print(pip('축구는 정말 재미있는 [MASK]다.'))
print(pip('어벤져스는 정말 재미있는 [MASK]다.'))
print(pip('나는 오늘 아침에 [MASK]에 출근을 했다.'))
