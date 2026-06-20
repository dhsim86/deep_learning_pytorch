import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import urllib.request
from konlpy.tag import Mecab
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from collections import Counter

dataset_path = 'lstm_cinema_review_dataset.pkl'

with open(dataset_path, 'rb') as f:
    loaded_dataset = pickle.load(f)

X_train = loaded_dataset['X_train']
X_valid = loaded_dataset['X_valid']
X_test = loaded_dataset['X_test']
y_train = loaded_dataset['y_train']
y_valid = loaded_dataset['y_valid']
y_test = loaded_dataset['y_test']

############################################################################
## 단어 집합 생성

### 단어 갯수 카운트
word_list = []
for sent in X_train:
    for word in sent:
      word_list.append(word)

word_counts = Counter(word_list)
print('총 단어수 :', len(word_counts))

print('훈련 데이터에서의 단어 영화의 등장 횟수 :', word_counts['영화'])
print('훈련 데이터에서의 단어 공감의 등장 횟수 :', word_counts['공감'])

### 등장 빈도수가 높은 순으로 단어를 정렬
vocab = sorted(word_counts, key=word_counts.get, reverse=True)
print('등장 빈도수 상위 10개 단어')
# ['영화', '너무', '정말', '만', '적', '진짜', '으로', '로', '점', '에서']
print(vocab[:10])

### 빈도수가 낮은 단어들은 자연어 처리에서 배제 (등장 빈도수가 3회 미만인 단어들을 제거)
threshold = 3
total_cnt = len(word_counts) # 단어의 수
rare_cnt = 0 # 등장 빈도수가 threshold보다 작은 단어의 개수를 카운트
total_freq = 0 # 훈련 데이터의 전체 단어 빈도수 총 합
rare_freq = 0 # 등장 빈도수가 threshold보다 작은 단어의 등장 빈도수의 총 합

# 단어와 빈도수의 쌍(pair)을 key와 value로 받는다.
for key, value in word_counts.items():
    total_freq = total_freq + value

    # 단어의 등장 빈도수가 threshold보다 작으면
    if(value < threshold):
        rare_cnt = rare_cnt + 1
        rare_freq = rare_freq + value

# 단어 집합(vocabulary)의 크기 : 88276
# 등장 빈도가 2번 이하인 희귀 단어의 수: 60078
# 단어 집합에서 희귀 단어의 비율: 68.05700303593277
# 전체 등장 빈도에서 희귀 단어 등장 빈도 비율: 5.644509532507002
print('단어 집합(vocabulary)의 크기 :',total_cnt)
print('등장 빈도가 %s번 이하인 희귀 단어의 수: %s'%(threshold - 1, rare_cnt))
print("단어 집합에서 희귀 단어의 비율:", (rare_cnt / total_cnt)*100)
print("전체 등장 빈도에서 희귀 단어 등장 빈도 비율:", (rare_freq / total_freq)*100)

############################################################################
## 정수 인코딩

### 등장 빈도수가 3회 미만인 단어는 제외하고 정수 인코딩

# 전체 단어 개수 중 빈도수 2이하인 단어는 제거.
vocab_size = total_cnt - rare_cnt
vocab = vocab[:vocab_size]
print('단어 집합의 크기 :', len(vocab))

# 스페셜 토큰 추가 (패딩, UNK)
word_to_index = {}
word_to_index['<PAD>'] = 0
word_to_index['<UNK>'] = 1

# 단어마다 고유 번호 부여
for index, word in enumerate(vocab) :
  word_to_index[word] = index + 2

print('단어 <PAD>와 맵핑되는 정수 :', word_to_index['<PAD>'])
print('단어 <UNK>와 맵핑되는 정수 :', word_to_index['<UNK>'])
print('단어 영화와 맵핑되는 정수 :', word_to_index['영화'])

# 정수 인코딩
## 단어 집합에 속하지 않는 단어들은 UNK로 매핑
def texts_to_sequences(tokenized_X_data, word_to_index):
  encoded_X_data = []
  for sent in tokenized_X_data:
    index_sequences = []
    for word in sent:
      try:
          index_sequences.append(word_to_index[word])
      except KeyError:
          index_sequences.append(word_to_index['<UNK>'])
    encoded_X_data.append(index_sequences)
  return encoded_X_data

encoded_X_train = texts_to_sequences(X_train, word_to_index)
encoded_X_valid = texts_to_sequences(X_valid, word_to_index)
encoded_X_test = texts_to_sequences(X_test, word_to_index)

## 상위 샘플 2개 출력
## [373, 2422, 7023, 16429]
## [9560, 3697, 75, 2346, 544, 24, 1, 615]
for sent in encoded_X_train[:2]:
  print(sent)

## 디코딩 테스트
index_to_word = {}
for key, value in word_to_index.items():
    index_to_word[value] = key

decoded_sample = [index_to_word[word] for word in encoded_X_train[0]]
print('기존의 첫번째 샘플 :', X_train[0]) # ['이야', '어쩜', '이렇게나', '지루할수가']
print('복원된 첫번째 샘플 :', decoded_sample) # ['이야', '어쩜', '이렇게나', '지루할수가']

############################################################################
## 패딩

print('리뷰의 최대 길이 :',max(len(review) for review in encoded_X_train))
print('리뷰의 평균 길이 :',sum(map(len, encoded_X_train))/len(encoded_X_train))
plt.hist([len(review) for review in encoded_X_train], bins=50)
plt.xlabel('length of samples')
plt.ylabel('number of samples')
plt.show()

### 모델이 처리할 수 있도록 모든 샘플의 길이를 특정 길이로 동일하게 맞춘다.

# 전체 샘플 중 길이가 max_len 이하인 샘플의 비율을 구하는 함수
def below_threshold_len(max_len, nested_list):
  count = 0
  for sentence in nested_list:
    if(len(sentence) <= max_len):
        count = count + 1
  print('전체 샘플 중 길이가 %s 이하인 샘플의 비율: %s'%(max_len, (count / len(nested_list))*100))

max_len = 30
below_threshold_len(max_len, X_train) # 94.11%

# 모든 샘플의 길이를 30으로 맞춤
def pad_sequences(sentences, max_len):
  features = np.zeros((len(sentences), max_len), dtype=int)
  for index, sentence in enumerate(sentences):
    if len(sentence) != 0:
      features[index, :len(sentence)] = np.array(sentence)[:max_len]
  return features

padded_X_train = pad_sequences(encoded_X_train, max_len=max_len)
padded_X_valid = pad_sequences(encoded_X_valid, max_len=max_len)
padded_X_test = pad_sequences(encoded_X_test, max_len=max_len)

# 훈련 데이터의 크기 : (116314, 30)
# 검증 데이터의 크기 : (29079, 30)
# 테스트 데이터의 크기 : (48852, 30)
print('훈련 데이터의 크기 :', padded_X_train.shape)
print('검증 데이터의 크기 :', padded_X_valid.shape)
print('테스트 데이터의 크기 :', padded_X_test.shape)

print('첫번째 샘플의 길이 :', len(padded_X_train[0]))

# [  373  2422  7023 16429     0     0     0     0     0     0     0     0
#     0     0     0     0     0     0     0     0     0     0     0     0
#     0     0     0     0     0     0]
print('첫번째 샘플 :', padded_X_train[0])