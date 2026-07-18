import re
import os
import unicodedata
import urllib3
import zipfile
import shutil
import numpy as np
import pandas as pd
import torch
from collections import Counter
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset

# 19만개의 데이터 중 33,000개의 샘플만 사용
num_samples = 33000

############################################################
# 전처리 함수 구현

## 프랑스어 정규화
def unicode_to_ascii(s):
  # 프랑스어 악센트(accent) 삭제
  # 예시 : 'déjà diné' -> deja dine
  return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

## 문장 정규화
def preprocess_sentence(sent):
  # 악센트 삭제 함수 호출
  sent = unicode_to_ascii(sent.lower())

  # 단어와 구두점 사이에 공백을 만듭니다.
  # Ex) "he is a boy." => "he is a boy ."
  sent = re.sub(r"([?.!,¿])", r" \1", sent)

  # (a-z, A-Z, ".", "?", "!", ",") 이들을 제외하고는 전부 공백으로 변환합니다.
  sent = re.sub(r"[^a-zA-Z!.?]+", r" ", sent)

  # 다수 개의 공백을 하나의 공백으로 치환
  sent = re.sub(r"\s+", " ", sent)
  return sent

## 영어 및 프랑스어 문장을 전처리 후, 인코더 / 디코더 입력용 데이터 준비
def load_preprocessed_data():
  encoder_input, decoder_input, decoder_target = [], [], []

  with open("fra.txt", "r", encoding="utf-8") as lines:
    for i, line in enumerate(lines):
      
      # 영어 및 프랑스어 문장 로드 (source 데이터와 target 데이터 분리)
      src_line, tar_line, _ = line.strip().split('\t')

      # 영어, source 데이터 전처리
      src_line = [w for w in preprocess_sentence(src_line).split()]

      # 프랑스어, target 데이터 전처리
      tar_line = preprocess_sentence(tar_line)

      # tearcher forcing을 위해 훈련시 사용할 디코더의 입력 시퀀스와
      # 실제값, 레이블에 해당하는 출력 시퀀스를 분리하여 저장
      tar_line_in = [w for w in ("<sos> " + tar_line).split()] # 디코더 입력 시퀀스, <sos>를 앞에 더해준다.
      tar_line_out = [w for w in (tar_line + " <eos>").split()] # 디코더 출력 시퀀스, <eos>를 뒤에 더해준다.


      encoder_input.append(src_line)
      decoder_input.append(tar_line_in)
      decoder_target.append(tar_line_out)

      if i == num_samples - 1:
        break

  return encoder_input, decoder_input, decoder_target

## 전처리 테스트
en_sent = u"Have you had dinner?"
fr_sent = u"Avez-vous déjà diné?"

## 전처리 전 영어 문장 : Have you had dinner?
## 전처리 후 영어 문장 : have you had dinner ?
## 전처리 전 프랑스어 문장 : Avez-vous déjà diné?
## 전처리 후 프랑스어 문장 : avez vous deja dine ?
print('전처리 전 영어 문장 :', en_sent)
print('전처리 후 영어 문장 :',preprocess_sentence(en_sent))
print('전처리 전 프랑스어 문장 :', fr_sent)
print('전처리 후 프랑스어 문장 :', preprocess_sentence(fr_sent))

# 코퍼스 로드 후, 데이터셋 인코더의 입력, 디코더의 입력 및 레이블 상위 5개 샘플 출력
## 인코더의 입력 : [['go', '.'], ['go', '.'], ['go', '.'], ['go', '.'], ['hi', '.']]
## 디코더의 입력 : [['<sos>', 'va', '!'], ['<sos>', 'marche', '.'], ['<sos>', 'en', 'route', '!'], ['<sos>', 'bouge', '!'], ['<sos>', 'salut', '!']]
## 디코더의 레이블 : [['va', '!', '<eos>'], ['marche', '.', '<eos>'], ['en', 'route', '!', '<eos>'], ['bouge', '!', '<eos>'], ['salut', '!', '<eos>']]
sents_en_in, sents_fra_in, sents_fra_out = load_preprocessed_data()
print('인코더의 입력 :',sents_en_in[:5])
print('디코더의 입력 :',sents_fra_in[:5])
print('디코더의 레이블 :',sents_fra_out[:5])

############################################################
# 단어 집합 생성

## 단어 집합 생성 함수
## PAD -> 0 / UNK -> 1
## 입력된 데이터로부터 빈도 순으로 단어를 정렬 후 정수를 할당
def build_vocab(sents):
  word_list = []

  for sent in sents:
      for word in sent:
        word_list.append(word)

  # 각 단어별 등장 빈도를 계산하여 등장 빈도가 높은 순서로 정렬
  word_counts = Counter(word_list)
  vocab = sorted(word_counts, key=word_counts.get, reverse=True)

  word_to_index = {}
  word_to_index['<PAD>'] = 0
  word_to_index['<UNK>'] = 1

  # 등장 빈도가 높은 단어일수록 낮은 정수를 부여
  for index, word in enumerate(vocab) :
    word_to_index[word] = index + 2

  return word_to_index

## 영어 및 프랑스어를 위한 단어 집합 생성
src_vocab = build_vocab(sents_en_in)
tar_vocab = build_vocab(sents_fra_in + sents_fra_out)

src_vocab_size = len(src_vocab)
tar_vocab_size = len(tar_vocab)
## 영어 단어 집합의 크기 : 4520, 프랑스어 단어 집합의 크기 : 7913
print("영어 단어 집합의 크기 : {:d}, 프랑스어 단어 집합의 크기 : {:d}".format(src_vocab_size, tar_vocab_size))

## 정수에서 단어를 얻는 딕셔너리 생성 (훈련 후 예측값/실제값 비교하는 단계에서 사용)
index_to_src = {v: k for k, v in src_vocab.items()}
index_to_tar = {v: k for k, v in tar_vocab.items()}

############################################################
# 데이터로부터 정수 인코딩
def texts_to_sequences(sents, word_to_index):
  encoded_X_data = []
  for sent in tqdm(sents):
    index_sequences = []
    for word in sent:
      try:
          index_sequences.append(word_to_index[word])
      except KeyError:
          index_sequences.append(word_to_index['<UNK>'])
    encoded_X_data.append(index_sequences)
  return encoded_X_data

encoder_input = texts_to_sequences(sents_en_in, src_vocab)
decoder_input = texts_to_sequences(sents_fra_in, tar_vocab)
decoder_target = texts_to_sequences(sents_fra_out, tar_vocab)

## 상위 5개의 샘플에 대해서 정수 인코딩 전, 후 문장 출력
## 인코더 입력이므로 <sos>나 <eos>가 없음
for i, (item1, item2) in zip(range(5), zip(sents_en_in, encoder_input)):
    # Index: 0, 정수 인코딩 전: ['go', '.'], 정수 인코딩 후: [27, 2]
    # Index: 1, 정수 인코딩 전: ['go', '.'], 정수 인코딩 후: [27, 2]
    # Index: 2, 정수 인코딩 전: ['go', '.'], 정수 인코딩 후: [27, 2]
    # Index: 3, 정수 인코딩 전: ['go', '.'], 정수 인코딩 후: [27, 2]
    # Index: 4, 정수 인코딩 전: ['hi', '.'], 정수 인코딩 후: [696, 2]
    print(f"Index: {i}, 정수 인코딩 전: {item1}, 정수 인코딩 후: {item2}")

############################################################
# 패딩
def pad_sequences(sentences, max_len=None):
    # 최대 길이 값이 주어지지 않을 경우 데이터 내 최대 길이로 패딩
    if max_len is None:
        max_len = max([len(sentence) for sentence in sentences])

    features = np.zeros((len(sentences), max_len), dtype=int)
    for index, sentence in enumerate(sentences):
        if len(sentence) != 0:
            features[index, :len(sentence)] = np.array(sentence)[:max_len]
    return features

encoder_input = pad_sequences(encoder_input)
decoder_input = pad_sequences(decoder_input)
decoder_target = pad_sequences(decoder_target)

# 인코더의 입력의 크기(shape) : (33000, 7)
# 디코더의 입력의 크기(shape) : (33000, 16)
# 디코더의 레이블의 크기(shape) : (33000, 16)
print('인코더의 입력의 크기(shape) :',encoder_input.shape)
print('디코더의 입력의 크기(shape) :',decoder_input.shape)
print('디코더의 레이블의 크기(shape) :',decoder_target.shape)

############################################################
# 학습을 위한 데이터셋 준비

# 데이터 셔플
## 랜덤한 인덱스 생성
indices = np.arange(encoder_input.shape[0])
np.random.shuffle(indices)
print('랜덤 시퀀스 :',indices)

## 랜덤 인덱스로 랜덤으로 섞인 데이터 샘플을 얻는다.
encoder_input = encoder_input[indices]
decoder_input = decoder_input[indices]
decoder_target = decoder_target[indices]

## 임의 샘플 출력, decoder_input과 decoder_target는 <sos> / <eos> 제외하고 동일한 시퀀스여야 함
## ['i', 'm', 'not', 'in', '.', '<PAD>', '<PAD>']
## ['<sos>', 'je', 'passe', '.', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>']
## ['je', 'passe', '.', '<eos>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>', '<PAD>']
print([index_to_src[word] for word in encoder_input[30997]])
print([index_to_tar[word] for word in decoder_input[30997]])
print([index_to_tar[word] for word in decoder_target[30997]])

# 테스트 데이터셋 분리 (10% 사용)
n_of_val = int(33000*0.1)
print('검증 데이터의 개수 :',n_of_val) # 검증 데이터의 개수 : 3300

encoder_input_train = encoder_input[:-n_of_val]
decoder_input_train = decoder_input[:-n_of_val]
decoder_target_train = decoder_target[:-n_of_val]

encoder_input_test = encoder_input[-n_of_val:]
decoder_input_test = decoder_input[-n_of_val:]
decoder_target_test = decoder_target[-n_of_val:]

## 훈련 source 데이터의 크기 : (29700, 7)
## 훈련 target 데이터의 크기 : (29700, 16)
## 훈련 target 레이블의 크기 : (29700, 16)
## 테스트 source 데이터의 크기 : (3300, 7)
## 테스트 target 데이터의 크기 : (3300, 16)
## 테스트 target 레이블의 크기 : (3300, 16)
print('훈련 source 데이터의 크기 :',encoder_input_train.shape)
print('훈련 target 데이터의 크기 :',decoder_input_train.shape)
print('훈련 target 레이블의 크기 :',decoder_target_train.shape)
print('테스트 source 데이터의 크기 :',encoder_input_test.shape)
print('테스트 target 데이터의 크기 :',decoder_input_test.shape)
print('테스트 target 레이블의 크기 :',decoder_target_test.shape)

############################################################
# 기계 번역기 모델 정의
import torch
import torch.nn as nn
import torch.optim as optim

embedding_dim = 256
hidden_dim = 256

class Encoder(nn.Module):
    def __init__(self, src_vocab_size, embedding_dim, hidden_dim):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(src_vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)

    def forward(self, x):
        # x.shape == (batch_size, seq_len, embedding_dim)
        x = self.embedding(x)
        
        # hidden.shape == (1, batch_size, hidden_dim)
        # cell.shape == (1, batch_size, hidden_dim)
        _, (hidden, cell) = self.lstm(x)
        
        # 인코더의 출력은 hidden state, cell state
        # 참고 (lstm_out, (hidden, cell) = self.lstm(embedded))
        #   lstm_out: 모든 시점의 마지막 층 은닉 상태를 모아놓은 텐서
        #     - 크기: (batch_size, seq_length, hidden_dim)
        #     - 즉 문장의 각 토큰 위치마다 하나씩 은닉 상태가 들어 있다. seq_length개의 은닉 상태가 순서대로 쌓여 있다.
        #   hidden: 가장 마지막 시점의 은닉 상태만 담는 텐서 
        #     - 크기: (num_layers(층 갯수) * num_directions(단방향 or 양방향), batch_size, hidden_dim)
        #     - 첫 번째 차원은 LSTM의 은닉층 갯수, 단방향/양방향 여부에 따라 다르다.
        #     - lstm_out[:, -1, :] == hidden.squeeze(0): lstm_out의 마지막 시점의 슬라이스와 같은 값
        #   cell: 가장 마지막 시점의 셀 상태 
        #     - 크기: (num_layers(층 갯수) * num_directions(단방향 or 양방향), batch_size, hidden_dim)
        #     - 셀 상태는 LSTM 내부의 장기 기억을 담는 통로이며, 다음 예측을 위해 넘겨줄 때 사용.
        #     - LSTM 내부 게이트를 통해 일부만 지우고 일부만 더하는 식으로 조금씩 갱신. 장기 의존성을 위해 존재
        # c_t = (망각게이트 ⊙ c_{t-1})  +  (입력게이트 ⊙ 새 후보값)   # 셀 상태 갱신
        # h_t = 출력게이트 ⊙ tanh(c_t)                                # 셀에서 걸러낸 출력



        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, tar_vocab_size, embedding_dim, hidden_dim):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(tar_vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, tar_vocab_size)

    def forward(self, x, hidden, cell):

        # x.shape == (batch_size, seq_len, embedding_dim)
        x = self.embedding(x)

        # 디코더의 LSTM으로 인코더의 hidden state, cell state를 전달.
        # output.shape == (batch_size, seq_len, hidden_dim)
        # hidden.shape == (1, batch_size, hidden_dim)
        # cell.shape == (1, batch_size, hidden_dim)
        output, (hidden, cell) = self.lstm(x, (hidden, cell))

        # output.shape: (batch_size, seq_len, tar_vocab_size)
        output = self.fc(output)

        # 디코더의 출력은 예측값, hidden state, cell state
        return output, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, trg):
        hidden, cell = self.encoder(src)

        # 훈련 중에는 디코더의 출력 중 오직 output만 사용한다.
        output, _, _ = self.decoder(trg, hidden, cell)
        return output

encoder = Encoder(src_vocab_size, embedding_dim, hidden_dim)
decoder = Decoder(tar_vocab_size, embedding_dim, hidden_dim)
model = Seq2Seq(encoder, decoder)

loss_function = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.Adam(model.parameters())

print(model)

# 평가 함수정의
def evaluation(model, dataloader, loss_function, device):
    
    # 모델을 평가 모드로 설정
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_count = 0

    # 자동 미분/기울기 계산을 비활성화 후 평가
    with torch.no_grad():
        for encoder_inputs, decoder_inputs, decoder_targets in dataloader:
            encoder_inputs = encoder_inputs.to(device)
            decoder_inputs = decoder_inputs.to(device)
            decoder_targets = decoder_targets.to(device)

            # 순방향 전파
            # outputs.shape == (batch_size, seq_len, tar_vocab_size)
            outputs = model(encoder_inputs, decoder_inputs)

            # 손실 계산
            # outputs.view(-1, outputs.size(-1))의 shape는 (batch_size * seq_len, tar_vocab_size)
            # decoder_targets.view(-1)의 shape는 (batch_size * seq_len)
            loss = loss_function(outputs.view(-1, outputs.size(-1)), decoder_targets.view(-1))
            total_loss += loss.item()

            # 정확도 계산 (패딩 토큰 제외)
            mask = decoder_targets != 0
            total_correct += ((outputs.argmax(dim=-1) == decoder_targets) * mask).sum().item()
            total_count += mask.sum().item()

    return total_loss / len(dataloader), total_correct / total_count

# 데이터셋을 torch 텐서로 변환
encoder_input_train_tensor = torch.tensor(encoder_input_train, dtype=torch.long)
decoder_input_train_tensor = torch.tensor(decoder_input_train, dtype=torch.long)
decoder_target_train_tensor = torch.tensor(decoder_target_train, dtype=torch.long)

encoder_input_test_tensor = torch.tensor(encoder_input_test, dtype=torch.long)
decoder_input_test_tensor = torch.tensor(decoder_input_test, dtype=torch.long)
decoder_target_test_tensor = torch.tensor(decoder_target_test, dtype=torch.long)

## 데이터셋 및 데이터로더 생성
batch_size = 128

### 학습 데이터로더는 shuffle=True로 설정하여 데이터를 에포크마다 랜덤하게 섞어준다.
train_dataset = TensorDataset(encoder_input_train_tensor, decoder_input_train_tensor, decoder_target_train_tensor)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

### 테스트 데이터셋 로더는 shuffle=False로 설정하여 데이터의 순서를 유지
valid_dataset = TensorDataset(encoder_input_test_tensor, decoder_input_test_tensor, decoder_target_test_tensor)
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

# 훈련
USE_CUDA = torch.cuda.is_available()
device = torch.device(
    "cuda" if USE_CUDA
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
print("cuda/mps/cpu 중 다음 기기로 학습함:", device)

model.to(device)

# Training loop
num_epochs = 30
best_val_loss = float('inf')

for epoch in range(num_epochs):
    # 훈련 모드
    model.train()

    for encoder_inputs, decoder_inputs, decoder_targets in train_dataloader:
        encoder_inputs = encoder_inputs.to(device)
        decoder_inputs = decoder_inputs.to(device)
        decoder_targets = decoder_targets.to(device)

        # 기울기 초기화
        optimizer.zero_grad()

        # 순방향 전파
        # outputs.shape == (batch_size, seq_len, tar_vocab_size)

        # tensor([[[-0.0648,  0.0359, -0.0735,  ..., -0.0241,  0.0819, -0.0039],
        #  [ 0.0483, -0.1251, -0.0108,  ...,  0.0081, -0.0155,  0.0736],
        #  [-0.0068, -0.0655, -0.1068,  ..., -0.0072, -0.0464,  0.0725],
        #  ...,
        #  [-0.0620,  0.0508, -0.0284,  ...,  0.0174,  0.0639,  0.0128],
        #  [-0.0620,  0.0507, -0.0279,  ...,  0.0186,  0.0637,  0.0130],
        #  [-0.0620,  0.0506, -0.0277,  ...,  0.0193,  0.0636,  0.0131]],
        # ...
        outputs = model(encoder_inputs, decoder_inputs)

        # 손실 계산 및 역방향 전파
        # outputs.view(-1, outputs.size(-1))의 shape는 (batch_size * seq_len, tar_vocab_size)
        # decoder_targets.view(-1)의 shape는 (batch_size * seq_len)
        loss = loss_function(outputs.view(-1, outputs.size(-1)), decoder_targets.view(-1))
        loss.backward()

        # 가중치 업데이트
        optimizer.step()

    train_loss, train_acc = evaluation(model, train_dataloader, loss_function, device)
    valid_loss, valid_acc = evaluation(model, valid_dataloader, loss_function, device)

    print(f'Epoch: {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Valid Loss: {valid_loss:.4f} | Valid Acc: {valid_acc:.4f}')

    # 검증 손실이 최소일 때 체크포인트 저장
    if valid_loss < best_val_loss:
        print(f'Validation loss improved from {best_val_loss:.4f} to {valid_loss:.4f}. 체크포인트를 저장합니다.')
        best_val_loss = valid_loss
        torch.save(model.state_dict(), 'best_model_checkpoint.pth')


# 모델 로드
model.load_state_dict(torch.load('best_model_checkpoint.pth'))

# 모델을 device에 올립니다.
model.to(device)

# 검증 데이터에 대한 정확도와 손실 계산
val_loss, val_accuracy = evaluation(model, valid_dataloader, loss_function, device)

print(f'Best model validation loss: {val_loss:.4f}')
print(f'Best model validation accuracy: {val_accuracy:.4f}')

print(tar_vocab['<sos>']) # 3
print(tar_vocab['<eos>']) # 4

# 추론 단계, 기계 번역기로 동작시키기

## 정수에서 단어를 얻는 딕셔너리 생성 (훈련 후 예측값/실제값 비교하는 단계에서 사용)
index_to_src = {v: k for k, v in src_vocab.items()}
index_to_tar = {v: k for k, v in tar_vocab.items()}

## 원문의 정수 시퀀스를 텍스트 시퀀스로 변환
def seq_to_src(input_seq):
  sentence = ''
  for encoded_word in input_seq:
    if(encoded_word != 0):
      sentence = sentence + index_to_src[encoded_word] + ' '
  return sentence

## 번역문의 정수 시퀀스를 텍스트 시퀀스로 변환
def seq_to_tar(input_seq):
  sentence = ''
  for encoded_word in input_seq:
    # <sos> / <eos> 제외
    if(encoded_word != 0 and encoded_word != tar_vocab['<sos>'] and encoded_word != tar_vocab['<eos>']):
      sentence = sentence + index_to_tar[encoded_word] + ' '
  return sentence

## 데이터셋(정수 인코딩된 것) 샘플 출력
print(encoder_input_test[25])
print(decoder_input_test[25])
print(decoder_target_test[25])

## 추론 진행 함수
def decode_sequence(input_seq, model, src_vocab_size, tar_vocab_size, max_output_len, int_to_src_token, int_to_tar_token):
    
    encoder_inputs = torch.tensor(input_seq, dtype=torch.long).unsqueeze(0).to(device) # (seq_length) -> (1, seq_length)

    # 인코더의 초기 상태 설정
    hidden, cell = model.encoder(encoder_inputs)

    # 시작 토큰 <sos> (정수인코딩: 3)을 디코더의 첫 입력으로 설정
    # unsqueeze(0)는 배치 차원을 추가하기 위함.
    decoder_input = torch.tensor([3], dtype=torch.long).unsqueeze(0).to(device)

    decoded_tokens = []

    # for문을 도는 것 == 디코더의 각 시점
    for _ in range(max_output_len):
        # decoder_input의 seq_length: 1
        # output.shape: (1, 1, tar_vocab_size)
        
        # 루프마다 이전 시점의 hidden, cell 상태를 넘겨준다.
        output, hidden, cell = model.decoder(decoder_input, hidden, cell)

        # 소프트맥스 회귀를 수행. 예측 단어의 인덱스
        # 각 단어의 확률 분포로부터 가장 높은 값을 가진 단어 인덱스를 선택
        output_token = output.argmax(dim=-1).item()

        # 종료 토큰 <eos>
        if output_token == 4:
            break

        # 각 시점의 단어(정수)는 decoded_tokens에 누적하였다가 최종 번역 시퀀스로 리턴합니다.
        decoded_tokens.append(output_token)

        # 현재 시점의 예측. 다음 시점의 입력으로 사용된다.
        decoder_input = torch.tensor([output_token], dtype=torch.long).unsqueeze(0).to(device)

    return ' '.join(int_to_tar_token[token] for token in decoded_tokens)

## 추론 테스트
for seq_index in [3, 50, 100, 300, 1001]:
  input_seq = encoder_input_train[seq_index]
  translated_text = decode_sequence(input_seq, model, src_vocab_size, tar_vocab_size, 20, index_to_src, index_to_tar)

  print("입력문장 :",seq_to_src(encoder_input_train[seq_index]))
  print("정답문장 :",seq_to_tar(decoder_input_train[seq_index]))
  print("번역문장 :",translated_text)
  print("-"*50)


for seq_index in [3, 50, 100, 300, 1001]:
  input_seq = encoder_input_test[seq_index]
  translated_text = decode_sequence(input_seq, model, src_vocab_size, tar_vocab_size, 20, index_to_src, index_to_tar)

  print("입력문장 :",seq_to_src(encoder_input_test[seq_index]))
  print("정답문장 :",seq_to_tar(decoder_input_test[seq_index]))
  print("번역문장 :",translated_text)
  print("-"*50)

  