import urllib.request
import numpy as np
from tqdm import tqdm
import re
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

urllib.request.urlretrieve("https://raw.githubusercontent.com/ukairia777/tensorflow-nlp-tutorial/main/12.%20RNN%20Sequence%20Labeling/dataset/train.txt", filename="train.txt")

f = open('train.txt', 'r')
tagged_sentences = []
sentence = []

for line in f:
    if len(line)==0 or line.startswith('-DOCSTART') or line[0]=="\n":
        if len(sentence) > 0:
            tagged_sentences.append(sentence)
            sentence = []
        continue
    splits = line.split(' ') # 공백을 기준으로 속성을 구분한다.
    splits[-1] = re.sub(r'\n', '', splits[-1]) # 줄바꿈 표시 \n을 제거한다.
    word = splits[0].lower() # 단어들은 소문자로 바꿔서 저장한다.
    sentence.append([word, splits[-1]]) # 단어와 개체명 태깅만 기록한다.

print("전체 샘플 개수: ", len(tagged_sentences)) # 전체 샘플의 개수 출력

# [['eu', 'B-ORG'], ['rejects', 'O'], ['german', 'B-MISC'], ['call', 'O'], ['to', 'O'], ['boycott', 'O'], ['british', 'B-MISC'], ['lamb', 'O'], ['.', 'O']]
print(tagged_sentences[0]) # 첫번째 샘플 출력

############################################################################
# 입력(X)와 레이블(y)를 준비

## 각 샘플에 대해 단어는 sentences에, 태깅 정보는 pos_tags에 저장
sentences, ner_tags = [], [] 
for tagged_sentence in tagged_sentences: # 14,041개의 문장 샘플을 1개씩 불러온다.
    sentence, tag_info = zip(*tagged_sentence) # 각 샘플에서 단어들은 sentence에 개체명 태깅 정보들은 tag_info에 저장.
    sentences.append(list(sentence)) # 각 샘플에서 단어 정보만 저장한다.
    ner_tags.append(list(tag_info)) # 각 샘플에서 개체명 태깅 정보만 저장한다.

# ['eu', 'rejects', 'german', 'call', 'to', 'boycott', 'british', 'lamb', '.']
print(sentences[0])
# ['B-ORG', 'O', 'B-MISC', 'O', 'O', 'O', 'B-MISC', 'O', 'O']
print(ner_tags[0])

## 훈련 / 검증 / 테스트 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(sentences, ner_tags, test_size=.2, random_state=777)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=.2, random_state=777)

# 훈련 데이터의 개수 : 8985
# 검증 데이터의 개수 : 2247
# 테스트 데이터의 개수 : 2809
# 훈련 데이터 레이블의 개수 : 8985
# 검증 데이터 레이블의 개수 : 2247
# 테스트 데이터 레이블의 개수 : 2809
print('훈련 데이터의 개수 :', len(X_train))
print('검증 데이터의 개수 :', len(X_valid))
print('테스트 데이터의 개수 :', len(X_test))
print('훈련 데이터 레이블의 개수 :', len(X_train))
print('검증 데이터 레이블의 개수 :', len(X_valid))
print('테스트 데이터 레이블의 개수 :', len(X_test))

############################################################################
# 단어 집합 생성 (입력)

## Counter를 통해 각 단어의 등장 빈도를 카운트
word_list = []
for sent in X_train:
    for word in sent:
      word_list.append(word)

word_counts = Counter(word_list)
print('총 단어수 :', len(word_counts))
# 훈련 데이터에서의 단어 the의 등장 횟수 : 5410
# 훈련 데이터에서의 단어 love의 등장 횟수 : 7
print('훈련 데이터에서의 단어 the의 등장 횟수 :', word_counts['the'])
print('훈련 데이터에서의 단어 love의 등장 횟수 :', word_counts['love'])

## 단어 빈도수 기반으로 내림차순으로 정렬 후 등장 빈도 상위 10개 단어 출력
vocab = sorted(word_counts, key=word_counts.get, reverse=True)
print('등장 빈도수 상위 10개 단어')
# ['the', ',', '.', 'of', 'in', 'to', 'a', ')', '(', 'and']
print(vocab[:10])

## 패딩 및 UNK 토큰 할당 및 단어 집합 생성
word_to_index = {}
word_to_index['<PAD>'] = 0
word_to_index['<UNK>'] = 1

for index, word in enumerate(vocab) :
  word_to_index[word] = index + 2

vocab_size = len(word_to_index)
print('패딩 토큰과 UNK 토큰을 고려한 단어 집합의 크기 :', vocab_size)

############################################################################
# 정수 인코딩 (입력)
def texts_to_sequences(tokenized_X_data, word_to_index):
  # 텍스트를 정수로 변환, OOV가 발생하면 UNK 토큰으로 매핑
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
for sent in encoded_X_train[:2]:
  # [1260, 3215, 117, 17, 21, 123, 56, 539, 23]
  # [5456, 10, 8229, 9, 8230, 186, 84, 1815, 11, 8, 1073, 5, 421, 6, 8231, 35, 2043, 291, 790, 957, 267, 4]
  print(sent)

## 디코딩 테스트
index_to_word = {}
for key, value in word_to_index.items():
    index_to_word[value] = key
decoded_sample = [index_to_word[word] for word in encoded_X_train[0]]
# 기존의 첫번째 샘플 : ['young', 'boys', '9', '1', '0', '8', '6', '19', '3']
# 복원된 첫번째 샘플 : ['young', 'boys', '9', '1', '0', '8', '6', '19', '3']
print('기존의 첫번째 샘플 :', X_train[0])
print('복원된 첫번째 샘플 :', decoded_sample)

############################################################################
# 단어 집합 생성 (레이블)

## y_train으로부터 존재하는 모든 태그들의 집합 구하기
flatten_tags = [tag for sent in y_train for tag in sent]
tag_vocab = list(set(flatten_tags))
# 태그 집합 : ['B-PER', 'I-MISC', 'B-ORG', 'I-PER', 'B-LOC', 'I-LOC', 'I-ORG', 'O', 'B-MISC']
# 태그 집합의 크기 : 9
print('태그 집합 :', tag_vocab)
print('태그 집합의 크기 :', len(tag_vocab))

tag_to_index = {}
tag_to_index['<PAD>'] = 0

for index, word in enumerate(tag_vocab) :
  tag_to_index[word] = index + 1

tag_vocab_size = len(tag_to_index)
# 태그 집합 : {'<PAD>': 0, 'O': 1, 'I-PER': 2, 'B-MISC': 3, 'I-LOC': 4, 'B-ORG': 5, 'I-ORG': 6, 'B-LOC': 7, 'B-PER': 8, 'I-MISC': 9}
print('태그 집합 :', tag_to_index)

############################################################################
# 정수 인코딩 (레이블)
## many to many 문제인 경우, 레이블도 시퀀스 데이터이므로 각 레이블을 정수 시퀀스로 변경
def encoding_label(sequence, tag_to_index):
  label_sequence = []
  for seq in sequence:
    label_sequence.append([tag_to_index[tag] for tag in seq])
  return label_sequence

encoded_y_train = texts_to_sequences(y_train, tag_to_index)
encoded_y_valid = texts_to_sequences(y_valid, tag_to_index)
encoded_y_test = texts_to_sequences(y_test, tag_to_index)

print('X 데이터 상위 2개')
# [[1260, 3215, 117, 17, 21, 123, 56, 539, 23], [5456, 10, 8229, 9, 8230, 186, 84, 1815, 11, 8, 1073, 5, 421, 6, 8231, 35, 2043, 291, 790, 957, 267, 4]]
print(encoded_X_train[:2])
print('-' * 50)
print('y 데이터 상위 2개')
# [[5, 8, 4, 4, 4, 4, 4, 4, 4], [1, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]]
print(encoded_y_train[:2])
print('-' * 50)
print('첫번째 샘플과 레이블의 길이')
print(len(encoded_X_train[0])) # 9
print(len(encoded_y_train[0])) # 9

############################################################################
# 패딩

# 샘플의 최대 길이 : 78
print('샘플의 최대 길이 : %d' % max(len(l) for l in encoded_X_train))
# 샘플의 평균 길이 : 14.518420
print('샘플의 평균 길이 : %f' % (sum(map(len, encoded_X_train))/len(encoded_X_train)))
plt.hist([len(s) for s in encoded_X_train], bins=50)
plt.xlabel('length of samples')
plt.ylabel('number of samples')
plt.show()

# 모델이 처리할 수 있도록 모든 샘플의 길이를 특정 길이로 맞추어야 함
max_len = 80

# 패딩 함수
def pad_sequences(sentences, max_len):
    features = np.zeros((len(sentences), max_len), dtype=int)
    for index, sentence in enumerate(sentences):
        if len(sentence) != 0:
            features[index, :len(sentence)] = np.array(sentence)[:max_len]
    return features

padded_X_train = pad_sequences(encoded_X_train, max_len=max_len)
padded_X_valid = pad_sequences(encoded_X_valid, max_len=max_len)
padded_X_test = pad_sequences(encoded_X_test, max_len=max_len)

## many to many 문제를 푸는 경우, 레이블도 패딩해야 한다.
padded_y_train = pad_sequences(encoded_y_train, max_len=max_len)
padded_y_valid = pad_sequences(encoded_y_valid, max_len=max_len)
padded_y_test = pad_sequences(encoded_y_test, max_len=max_len)

# 훈련 데이터의 크기 : (8985, 80)
# 검증 데이터의 크기 : (2247, 80)
# 테스트 데이터의 크기 : (2809, 80)
print('훈련 데이터의 크기 :', padded_X_train.shape)
print('검증 데이터의 크기 :', padded_X_valid.shape)
print('테스트 데이터의 크기 :', padded_X_test.shape)
print('-' * 30)

# 훈련 데이터의 레이블 : (8985, 80)
# 검증 데이터의 레이블 : (2247, 80)
# 테스트 데이터의 레이블 : (2809, 80)
print('훈련 데이터의 레이블 :', padded_y_train.shape)
print('검증 데이터의 레이블 :', padded_y_valid.shape)
print('테스트 데이터의 레이블 :', padded_y_test.shape)

print('훈련 데이터의 상위 샘플 2개')
# [[1260 3215  117   17   21  123   56  539   23    0    0    0    0    0
#      0    0    0    0    0    0    0    0    0    0    0    0    0    0
#      0    0    0    0    0    0    0    0    0    0    0    0    0    0
#      0    0    0    0    0    0    0    0    0    0    0    0    0    0
#      0    0    0    0    0    0    0    0    0    0    0    0    0    0
#      0    0    0    0    0    0    0    0    0    0]
#  [5456   10 8229    9 8230  186   84 1815   11    8 1073    5  421    6
#   8231   35 2043  291  790  957  267    4    0    0    0    0    0    0
#      0    0    0    0    0    0    0    0    0    0    0    0    0    0
#      0    0    0    0    0    0    0    0    0    0    0    0    0    0
#      0    0    0    0    0    0    0    0    0    0    0    0    0    0
#      0    0    0    0    0    0    0    0    0    0]]
print(padded_X_train[:2])
print('-' * 5 + '레이블' + '-' * 5)
# [[3 8 6 6 6 6 6 6 6 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#   0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#   0 0 0 0 0 0 0 0]
#  [5 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#   0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#   0 0 0 0 0 0 0 0]]
print(padded_y_train[:2])

############################################################################
# 모델 학습 준비
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

## LSTM을 이용한 개체명 인식 모델, 양방향 LSTM 모델 구현
class NERTagger(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, num_layers=2):
        super(NERTagger, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim) # 양방향 LSTM 출력을 위하 hidden_dim * 2를 입력받음

    def forward(self, x):
        # x: (batch_size, seq_length)
        embedded = self.embedding(x)  # (batch_size, seq_length, embedding_dim)
        lstm_out, _ = self.lstm(embedded)  # (batch_size, seq_length, hidden_dim * 2)
        logits = self.fc(lstm_out)  # (batch_size, seq_length, output_dim)
        return logits
    
## 데이터셋을 텐서로 변환 후 배치 처리를 위한 데이터로더로 변환
X_train_tensor = torch.tensor(padded_X_train, dtype=torch.long)
y_train_tensor = torch.tensor(padded_y_train, dtype=torch.long)
X_valid_tensor = torch.tensor(padded_X_valid, dtype=torch.long)
y_valid_tensor = torch.tensor(padded_y_valid, dtype=torch.long)
X_test_tensor = torch.tensor(padded_X_test, dtype=torch.long)
y_test_tensor = torch.tensor(padded_y_test, dtype=torch.long)

train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
train_dataloader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=32)
valid_dataset = torch.utils.data.TensorDataset(X_valid_tensor, y_valid_tensor)
valid_dataloader = torch.utils.data.DataLoader(valid_dataset, shuffle=False, batch_size=32)
test_dataset = torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor)
test_dataloader = torch.utils.data.DataLoader(test_dataset, shuffle=False, batch_size=32)

## 모델 생성
### 하이퍼파라미터 정의
embedding_dim = 100
hidden_dim = 256
output_dim = tag_vocab_size # 10
learning_rate = 0.01
num_epochs = 10
num_layers = 2

model = NERTagger(vocab_size, embedding_dim, hidden_dim, output_dim, num_layers)
model.to(device)

## 비용 함수 및 옵티마이저 정의
criterion = nn.CrossEntropyLoss(ignore_index=0) # 특정 인덱스(0)에 대한 loss를 구하지 않음 (패딩에 대해서 loss를 구하지 않는다.)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

## 평가 코드 작성
### 정확도를 구하는 함수
def calculate_accuracy(logits, labels, ignore_index=0):
    # 예측 레이블을 구합니다.
    predicted = torch.argmax(logits, dim=1)

    # 패딩 토큰은 무시합니다.
    mask = (labels != ignore_index)

    # 정답을 맞춘 경우를 집계합니다.
    correct = (predicted == labels).masked_select(mask).sum().item()
    total = mask.sum().item()

    accuracy = correct / total
    return accuracy

### 검증 및 테스트 데이터에 대한 모델 성능을 측정하는 함수 정의
def evaluate(model, valid_dataloader, criterion, device):
    val_loss = 0
    val_correct = 0
    val_total = 0

    model.eval()
    with torch.no_grad():
        for batch_X, batch_y in valid_dataloader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            # Forward pass
            logits = model(batch_X)

            # Compute loss
            loss = criterion(logits.view(-1, output_dim), batch_y.view(-1))

            # Calculate validation accuracy and loss
            val_loss += loss.item()
            val_correct += calculate_accuracy(logits.view(-1, output_dim), batch_y.view(-1)) * batch_y.size(0)
            val_total += batch_y.size(0)

    val_accuracy = val_correct / val_total
    val_loss /= len(valid_dataloader)

    return val_loss, val_accuracy

############################################################################
# 모델 학습
best_val_loss = float('inf')

for epoch in range(num_epochs):
    # Training
    train_loss = 0
    train_correct = 0
    train_total = 0

    model.train()

    for batch_X, batch_y in train_dataloader:
        # Forward pass
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        logits = model(batch_X)

        # Compute loss
        loss = criterion(logits.view(-1, output_dim), batch_y.view(-1))

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Calculate training accuracy and loss
        train_loss += loss.item()
        train_correct += calculate_accuracy(logits.view(-1, output_dim), batch_y.view(-1)) * batch_y.size(0)
        train_total += batch_y.size(0)

    train_accuracy = train_correct / train_total
    train_loss /= len(train_dataloader)

    # Validation
    val_loss, val_accuracy = evaluate(model, valid_dataloader, criterion, device)

    print(f'Epoch {epoch+1}/{num_epochs}:')
    print(f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}')
    print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')

    # 검증 손실이 최소일 때 체크포인트 저장
    if val_loss < best_val_loss:
        print(f'Validation loss improved from {best_val_loss:.4f} to {val_loss:.4f}. 체크포인트를 저장합니다.')
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_model_checkpoint.pth')

# Epoch 1/10:
# Train Loss: 0.4538, Train Accuracy: 0.8739
# Validation Loss: 0.2452, Validation Accuracy: 0.9294
# Validation loss improved from inf to 0.2452. 체크포인트를 저장합니다.

# Epoch 2/10:
# Train Loss: 0.1430, Train Accuracy: 0.9579
# Validation Loss: 0.1660, Validation Accuracy: 0.9549
# Validation loss improved from 0.2452 to 0.1660. 체크포인트를 저장합니다.

# Epoch 3/10:
# Train Loss: 0.0654, Train Accuracy: 0.9806
# Validation Loss: 0.1880, Validation Accuracy: 0.9567

# Epoch 4/10:
# Train Loss: 0.0378, Train Accuracy: 0.9885
# Validation Loss: 0.1701, Validation Accuracy: 0.9590
# ...

############################################################################
# 모델 로드 및 평가

## 모델 로드
model.load_state_dict(torch.load('best_model_checkpoint.pth'))
model.to(device)

## 검증 데이터에 대한 정확도(accuracy)와 손실(loss) 계산
val_loss, val_accuracy = evaluate(model, valid_dataloader, criterion, device)

# Best model validation loss: 0.1660
# Best model validation accuracy: 0.9549
print(f'Best model validation loss: {val_loss:.4f}')
print(f'Best model validation accuracy: {val_accuracy:.4f}')

## 테스트 데이터에 대한 정확도와 손실 계산
test_loss, test_accuracy = evaluate(model, test_dataloader, criterion, device)

# Best model test loss: 0.1577
# Best model test accuracy: 0.9561
print(f'Best model test loss: {test_loss:.4f}')
print(f'Best model test accuracy: {test_accuracy:.4f}')

############################################################################
# 인퍼런스 및 테스트

## 임의의 텍스트 입력에 대해 레이블을 예측하는 함수
index_to_tag = {}
for key, value in tag_to_index.items():
    index_to_tag[value] = key

def predict_labels(text, model, word_to_ix, index_to_tag, max_len=150):
    # 단어 토큰화
    tokens = text.split()

    # 정수 인코딩, OOV가 발생하면 1(UNK) 할당
    token_indices = [word_to_ix.get(token, 1) for token in tokens]

    # 패딩
    token_indices_padded = np.zeros(max_len, dtype=int)
    token_indices_padded[:len(token_indices)] = token_indices[:max_len]

    # 텐서로 변환
    input_tensor = torch.tensor(token_indices_padded, dtype=torch.long).unsqueeze(0).to(device)

    # 모델의 입력으로 사용하고 예측값 리턴
    model.eval()
    with torch.no_grad():
        logits = model(input_tensor)

    # 가장 값이 높은 인덱스를 예측값으로 선택
    predicted_indices = torch.argmax(logits, dim=-1).squeeze(0).tolist()

    # 패딩 토큰 제거
    predicted_indices_no_pad = predicted_indices[:len(tokens)]

    # 패딩 토큰을 제외하고 정수 시퀀스를 예측 시퀀스로 변환
    predicted_tags = [index_to_tag[index] for index in predicted_indices_no_pad]

    return predicted_tags

# ['feyenoord', 'rotterdam', 'suffered', 'an', 'early', 'shock', 'when', 'they', 'went', '1-0', 'down', 'after', 'four', 'minutes', 'against', 'de', 'graafschap', 'doetinchem', '.']
print(X_test[0])

# feyenoord rotterdam suffered an early shock when they went 1-0 down after four minutes against de graafschap doetinchem .
sample = ' '.join(X_test[0])
print(sample)

predicted_tags = predict_labels(sample, model, word_to_index, index_to_tag)

# 예측 : ['B-ORG', 'I-ORG', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-ORG', 'I-ORG', 'I-ORG', 'O']
# 실제값 : ['B-ORG', 'I-ORG', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-ORG', 'I-ORG', 'O']
print('예측 :', predicted_tags)
print('실제값 :', y_test[0])
