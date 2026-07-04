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


############################################################################
# LSTM을 이용한 네이버 영화 리뷰 분류 모델

import torch
import torch.nn as nn
import torch.nn.functional as F

USE_CUDA = torch.cuda.is_available()
device = torch.device(
    "cuda" if USE_CUDA
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
print("cuda/mps/cpu 중 다음 기기로 학습함:", device)

## LSTM 모델 구현

### 각 층에서의 텐서 크기 변화
# 입력: (배치크기, 문장 길이)
# 임베딩 층 이후: (배치크기, 문장 길이, 임베딩 벡터의 차원)

# LSTM의 출력 차원: (배치크기, 은닉 상태의 차원)
## -> 다대일 문제이므로 최종 시점의 은닉 상태만 사용 (모든 시점의 은닉 상태를 사용하지 않음)

# 최종 출력: (배치크기, 분류하고자하는 카테고리 수)
## -> 소프트맥스 회귀 사용

### 모델 정의
class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(TextClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x: (batch_size, seq_length)
        embedded = self.embedding(x)  # (batch_size, seq_length, embedding_dim)

        # LSTM은 (hidden state, cell state)의 튜플을 반환합니다
        lstm_out, (hidden, cell) = self.lstm(embedded)  # lstm_out: (batch_size, seq_length, hidden_dim), hidden: (층의 갯수(1), batch_size, hidden_dim)

        last_hidden = hidden.squeeze(0)  # (batch_size, hidden_dim)
        logits = self.fc(last_hidden)  # (batch_size, output_dim)
        return logits
    

# 훈련 데이터 및 검증, 테스트 데이터를 파이토치 텐선로 변환 후 데이터로더 사용

## 정답 데이터를 파이토치 텐서로 변환
train_label_tensor = torch.tensor(np.array(y_train))
valid_label_tensor = torch.tensor(np.array(y_valid))
test_label_tensor = torch.tensor(np.array(y_test))
print(train_label_tensor[:5])

encoded_train = torch.tensor(padded_X_train).to(torch.int64)
train_dataset = torch.utils.data.TensorDataset(encoded_train, train_label_tensor)
train_dataloader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=32)

encoded_test = torch.tensor(padded_X_test).to(torch.int64)
test_dataset = torch.utils.data.TensorDataset(encoded_test, test_label_tensor)
test_dataloader = torch.utils.data.DataLoader(test_dataset, shuffle=True, batch_size=1)

encoded_valid = torch.tensor(padded_X_valid).to(torch.int64)
valid_dataset = torch.utils.data.TensorDataset(encoded_valid, valid_label_tensor)
valid_dataloader = torch.utils.data.DataLoader(valid_dataset, shuffle=True, batch_size=1)

total_batch = len(train_dataloader)
print('학습 데이터의 총 배치의 수 : {}'.format(total_batch))

## 모델 객체 생성
### 하이퍼파라미터 정의
embedding_dim = 100 # 임베딩 벡터 차원
hidden_dim = 128    # LSTM 모델의 은닉 차원
output_dim = 2      # 출력 차원
learning_rate = 0.01
num_epochs = 10

# vocab_size + 2: PAD, UNK 포함
model = TextClassifier(vocab_size + 2, embedding_dim, hidden_dim, output_dim)
model.to(device)

### 손실 함수 및 옵티마이저 정의 (소프트맥스 회귀이므로, BCE 사용)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

############################################################################
# 모델 학습

## 모델 학습 준비
### 모델 정확도 측정 함수 정의
def calculate_accuracy(logits, labels):
    # _, predicted = torch.max(logits, 1)
    predicted = torch.argmax(logits, dim=1)
    correct = (predicted == labels).sum().item()
    total = labels.size(0)
    accuracy = correct / total
    return accuracy

### 검증 및 테스트 데이터에 대한 성능을 측정하는 함수 정의
def evaluate(model, valid_dataloader, criterion, device):
    val_loss = 0
    val_correct = 0
    val_total = 0

    # 모델을 평가 모드로 설정 (반드시 사용해야 함)
    ## 모델 내부의 모든 레이어에 대해 평가 모드 활성화
    ## 드롭아웃이나 배치 정규화는 학습과 평가시 다르게 동작
    ## -> 평가시에는 드롭아웃 기능이 비활성화됨
    model.eval()

    # 파이토치의 autograd, 자동 미분 엔진에서 기울기 계산을 비활성화
    ## 평가 중에는 기울기를 계산할 필요없고, 메모리 절약 및 속도를 증대시킬 수 있다.
    ## 만약 하지 않으면 평가 중인데도 기울기 계산하고, 메모리를 차지한다.
    with torch.no_grad():
        # 데이터로더로부터 배치 크기만큼의 데이터를 연속으로 로드
        for batch_X, batch_y in valid_dataloader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            # 모델의 예측값
            logits = model(batch_X)

            # 손실을 계산
            loss = criterion(logits, batch_y)

            # 정확도와 손실을 계산함
            val_loss += loss.item()
            val_correct += calculate_accuracy(logits, batch_y) * batch_y.size(0)
            val_total += batch_y.size(0)

    val_accuracy = val_correct / val_total
    val_loss /= len(valid_dataloader)

    return val_loss, val_accuracy

## 모델 학습
num_epochs = 5

# Training loop
best_val_loss = float('inf')

for epoch in range(num_epochs):
    # Training
    train_loss = 0
    train_correct = 0
    train_total = 0

    # 학습 모드로 설정
    model.train()

    # 배치로 모델 학습
    for batch_X, batch_y in train_dataloader:
        # Forward pass
        batch_X, batch_y = batch_X.to(device), batch_y.to(device) # 배치 데이터를 GPU로 로드

        # batch_X.shape == (batch_size, max_len)
        logits = model(batch_X) # 예측값(logits) 계산

        # Compute loss
        loss = criterion(logits, batch_y)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Calculate training accuracy and loss
        train_loss += loss.item()
        train_correct += calculate_accuracy(logits, batch_y) * batch_y.size(0)
        train_total += batch_y.size(0)

    train_accuracy = train_correct / train_total
    train_loss /= len(train_dataloader)

    # Validation
    ## 각 epoch마다 검증 데이터셋으로 평가를 진행
    val_loss, val_accuracy = evaluate(model, valid_dataloader, criterion, device)

    print(f'Epoch {epoch+1}/{num_epochs}:')
    print(f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}')
    print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')

    # 검증 손실이 최소일 때 모델의 체크포인트 저장
    if val_loss < best_val_loss:
        print(f'Validation loss improved from {best_val_loss:.4f} to {val_loss:.4f}. 체크포인트를 저장합니다.')
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_model_checkpoint.pth')

# Epoch 1/5:
# Train Loss: 0.5141, Train Accuracy: 0.7151
# Validation Loss: 0.3867, Validation Accuracy: 0.8242
# Validation loss improved from inf to 0.3867. 체크포인트를 저장합니다.

# Epoch 2/5:
# Train Loss: 0.3323, Train Accuracy: 0.8547
# Validation Loss: 0.3544, Validation Accuracy: 0.8399
# Validation loss improved from 0.3867 to 0.3544. 체크포인트를 저장합니다.

# Epoch 3/5:
# Train Loss: 0.2594, Train Accuracy: 0.8919
# Validation Loss: 0.3630, Validation Accuracy: 0.8422

# Epoch 4/5:
# Train Loss: 0.1972, Train Accuracy: 0.9215
# Validation Loss: 0.4045, Validation Accuracy: 0.8415

# Epoch 5/5:
# Train Loss: 0.1431, Train Accuracy: 0.9454
# Validation Loss: 0.4512, Validation Accuracy: 0.8375

############################################################################
# 모델 평가

## 모델 로드
model.load_state_dict(torch.load('best_model_checkpoint.pth'))

## 모델을 device로 이동
model.to(device)

## 검증 데이터에 대한 정확도와 손실 계산
val_loss, val_accuracy = evaluate(model, valid_dataloader, criterion, device)

print(f'Best model validation loss: {val_loss:.4f}')
print(f'Best model validation accuracy: {val_accuracy:.4f}')

## 테스트 데이터에 대한 정확도와 손실 계산
test_loss, test_accuracy = evaluate(model, test_dataloader, criterion, device)

print(f'Best model test loss: {test_loss:.4f}')
print(f'Best model test accuracy: {test_accuracy:.4f}')

############################################################################
# 모델 직접 테스트
from konlpy.tag import Okt
okt = Okt()
stopwords = ['도', '는', '다', '의', '가', '이', '은', '한', '에', '하', '고', '을', '를', '인', '듯', '과', '와', '네', '들', '듯', '지', '임', '게']

index_to_tag = {0 : '부정', 1 : '긍정'}

## 임의의 입력에 대해 모델 예측하는 함수 정의
def predict(text, model, word_to_index, index_to_tag):
    # 평가 모드로 전환
    model.eval()

    # 입력 텍스트 토큰화
    tokens = okt.morphs(text) # 토큰화
    tokens = [word for word in tokens if not word in stopwords] # 불용어 제거
    # 정수 인코딩, 사전에 없는 단어는 UNK로 부여
    token_indices = [word_to_index.get(token, 1) for token in tokens]

    # 정수 인코딩 후 텐서로 변환
    input_tensor = torch.tensor([token_indices], dtype=torch.long).to(device)  # (1, seq_length)

    # 인퍼런스
    with torch.no_grad():
        logits = model(input_tensor)  # (1, output_dim)

    # 예측한 분류 클래스 확인
    predicted_index = torch.argmax(logits, dim=1)
    predicted_tag = index_to_tag[predicted_index.item()]

    return predicted_tag


## 임의의 텍스트로 예측
test_input = "이 영화 개꿀잼 ㅋㅋㅋ"
print(predict(test_input, model, word_to_index, index_to_tag))

test_input = "이딴게 영화냐 ㅉㅉ"
print(predict(test_input, model, word_to_index, index_to_tag))

test_input = "감독 뭐하는 놈이냐?"
print(predict(test_input, model, word_to_index, index_to_tag))

test_input = "와 개쩐다 정말 세계관 최강자들의 영화다"
print(predict(test_input, model, word_to_index, index_to_tag))

test_input = "감독 누구냐? 뭐 이따위 만듬?"
print(predict(test_input, model, word_to_index, index_to_tag))