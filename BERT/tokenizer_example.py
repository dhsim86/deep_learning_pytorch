import pandas as pd
from transformers import BertTokenizer

# Bert-base의 토크나이저
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# 토크나이저로 토큰화
result = tokenizer.tokenize('Here is the sentence I want embeddings for.')

# 토큰화 결과
## embeddings는 단어 집합에 존재하지 않아, 'em', '##bed', '##ding', '##s'로 분리
## ['here', 'is', 'the', 'sentence', 'i', 'want', 'em', '##bed', '##ding', '##s', 'for', '.']
print(result)

# BERT의 단어 집합에 특정 단어가 있는지 조회
## 존재함 (단어 here는 정수 인코딩을 위해 내부적으로 2182로 매핑되어 있음)
print(tokenizer.vocab['here']) # 2182

## embeddings는 존재하지 않음
## Traceback (most recent call last):
 ## File "/Users/user/workspace/repo/personal/deep_learning_pytorch/BERT/tokenizer_example.py", line 20, in <module>
 ##   print(tokenizer.vocab['embeddings'])
## KeyError: 'embeddings'
# print(tokenizer.vocab['embeddings'])

## 서브워드 em, ##bed, ##ing, ##s 존재함
print(tokenizer.vocab['em']) # 7861
print(tokenizer.vocab['##bed']) # 8270
print(tokenizer.vocab['##ding']) # 4667
print(tokenizer.vocab['##s']) # 2015

# BERT의 단어 집합 살펴보기
## BERT의 단어 집합을 vocabulary.txt에 저장
## 단어가 매핑된 정수 값(items()) 기준으로 정렬해 저장
with open('vocabulary.txt', 'w') as f:
  for token, index in sorted(tokenizer.vocab.items(), key=lambda x: x[1]):
    f.write(token + '\n')

## pandas로 데이터 프레임 형태로 읽어 확인

##                0
## 0          [PAD]
## 1      [unused0]
## 2      [unused1]
## 3      [unused2]
## 4      [unused3]
## ...          ...
## 30517        ##．
## 30518        ##／
## 30519        ##：
## 30520        ##？
## 30521        ##～
df = pd.read_fwf('vocabulary.txt', header=None)
print(df)

print('단어 집합의 크기 :',len(df)) # 30522

print(df.loc[4667].values[0]) ## ding, 정수 4667과 매핑됨

# BERT의 특수 토큰들
## PAD, UNK, CLS, SEP, MASK
print(df.loc[0].values[0])   # [PAD]
print(df.loc[100].values[0]) # [UNK]
print(df.loc[101].values[0]) # [CLS]
print(df.loc[102].values[0]) # [SEP]
print(df.loc[103].values[0]) # [MASK]