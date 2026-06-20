import numpy as np
from collections import Counter
import gensim

# 문장의 긍, 부정을 판단하는 감성 분류 모델 테스트
## 문장과 레이블 데이터 준비
sentences = [
   'nice great best amazing',
   'stop lies',
   'pitiful nerd',
   'excellent work',
   'supreme quality',
   'bad',
   'highly respectable']
y_train = [
   1,
   0,
   0,
   1,
   1,
   0,
   1] # 긍정인 문장은 1, 부정이면 0

# 단어 토큰화
tokenized_sentences = [sent.split() for sent in sentences]
print('단어 토큰화 된 결과 :', tokenized_sentences)

# 단어 집합 생성
word_list = []
for sent in tokenized_sentences:
    for word in sent:
      word_list.append(word)

word_counts = Counter(word_list)
print(word_counts)
print('총 단어수 :', len(word_counts)) # 단어의 등장 빈도수 기록

# 등장 빈도순으로 정렬
vocab = sorted(word_counts, key=word_counts.get, reverse=True)
print(vocab)

word_to_index = {}
word_to_index['<PAD>'] = 0 # 패딩 토큰
word_to_index['<UNK>'] = 1 # OOV 문제발생시 할당

for index, word in enumerate(vocab) :
  word_to_index[word] = index + 2

vocab_size = len(word_to_index)
print('패딩 토큰, UNK 토큰을 고려한 단어 집합의 크기 :', vocab_size)

# 정수 인코딩
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

X_encoded = texts_to_sequences(tokenized_sentences, word_to_index)
print(X_encoded)

# 패딩
max_len = max(len(l) for l in X_encoded)
print('최대 길이 :',max_len)

def pad_sequences(sentences, max_len):
  features = np.zeros((len(sentences), max_len), dtype=int)
  for index, sentence in enumerate(sentences):
    if len(sentence) != 0:
      features[index, :len(sentence)] = np.array(sentence)[:max_len]
  return features

X_train = pad_sequences(X_encoded, max_len=max_len)
y_train = np.array(y_train)
print('패딩 결과 :')
print(X_train)

# 벡터화
## nn.Embedding을 통해 모델 설계
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

## 모델 객체 정의
class SimpleModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(SimpleModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(embedding_dim * max_len, 1) # (400, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # embedded.shape == (배치 크기, 문장의 길이, 임베딩 벡터의 차원)
        embedded = self.embedding(x) # (2 x 4 x 100)

        # flattend.shape == (배치 크기, 문장의 길이 × 임베딩 벡터의 차원)
        flattened = self.flatten(embedded) # (2 x 400)

        # output.shape == (배치 크기, 1)
        output = self.fc(flattened) # (2 x 400) x (400 x 1) => (2, 1)
        return self.sigmoid(output)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

## 임베딩 벡터의 크기는 100으로 지정
embedding_dim = 100
simple_model = SimpleModel(vocab_size, embedding_dim).to(device)

## 긍정,부정을 분류하는 로지스틱 회귀 모델을 쓰므로, 손실 함수로 BCE 사용
criterion = nn.BCELoss()
optimizer = Adam(simple_model.parameters())

# 배치 크기가 2인 데이터로더로 변환
train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.long), torch.tensor(y_train, dtype=torch.float32))
train_dataloader = DataLoader(train_dataset, batch_size=2)

# 학습 전 임베딩 출력
print(simple_model.embedding(torch.tensor(X_train, dtype=torch.long).to(device)))

# 학습
for epoch in range(500):
    for inputs, targets in train_dataloader:
        # inputs.shape == (배치 크기, 문장 길이)
        # targets.shape == (배치 크기)
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()

        # outputs.shape == (배치 크기)
        outputs = simple_model(inputs).view(-1) 

        loss = criterion(outputs, targets)
        loss.backward()

        optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

# 학습 후 임베딩 출력 (학습이 됨)
print(simple_model.embedding(torch.tensor(X_train, dtype=torch.long).to(device)))
