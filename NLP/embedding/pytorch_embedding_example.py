import torch
import torch.nn as nn

# 임의의 문장으로부터 단어 집합을 만든 후 정수 인코딩
train_data = 'you need to know how to code'

# 중복을 제거한 단어들의 집합인 단어 집합 생성.
word_set = set(train_data.split())

# 단어 집합의 각 단어에 고유한 정수 맵핑.
vocab = {word: i+2 for i, word in enumerate(word_set)}
vocab['<unk>'] = 0
vocab['<pad>'] = 1
print(vocab)

# 단어 집합의 크기를 행 크기로 하는 임베딩 테이블 구현 (각 임베딩 벡터 차원 크기는 3)
embedding_table = torch.FloatTensor([
                               [ 0.0,  0.0,  0.0],
                               [ 0.0,  0.0,  0.0],
                               [ 0.2,  0.9,  0.3],
                               [ 0.1,  0.5,  0.7],
                               [ 0.2,  0.1,  0.8],
                               [ 0.4,  0.1,  0.1],
                               [ 0.1,  0.8,  0.9],
                               [ 0.6,  0.1,  0.1]])

# 임베딩 벡터 가져와보기
sample = 'you need to run'.split()
idxes = []

# 각 단어를 정수로 변환
for word in sample:
  try:
    idxes.append(vocab[word])
  # 단어 집합에 없는 단어일 경우 <unk>로 대체된다.
  except KeyError:
    idxes.append(vocab['<unk>'])
idxes = torch.LongTensor(idxes)

# 각 정수를 인덱스로 임베딩 테이블에서 값을 가져온다.
lookup_result = embedding_table[idxes, :]
print(lookup_result)


# 파이토치의 nn.embedding 사용해보기
train_data = 'you need to know how to code'

# 중복을 제거한 단어들의 집합인 단어 집합 생성.
word_set = set(train_data.split())

# 단어 집합의 각 단어에 고유한 정수 맵핑.
vocab = {tkn: i+2 for i, tkn in enumerate(word_set)}
vocab['<unk>'] = 0
vocab['<pad>'] = 1

# nn.Embedding()을 사용하여 학습가능한 임베딩 테이블을 생성
## num_embeddings: 임베딩을 할 단어들의 개수. 단어 집합의 크기
## embedding_dim: 임베딩 할 벡터의 차원. 사용자가 정해주는 하이퍼파라미터
## padding_idx: 선택적으로 사용하는 인자. 패딩을 위한 토큰의 인덱스를 지정
embedding_layer = nn.Embedding(num_embeddings=len(vocab), 
                               embedding_dim=3,
                               padding_idx=1)

## 임베딩 테이블 출력
print(embedding_layer.weight)