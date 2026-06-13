from konlpy.tag import Okt

okt = Okt()

# 형태소 추출
token = okt.morphs("나는 자연어 처리를 배운다")
# ['나', '는', '자연어', '처리', '를', '배운다']
print(token)

# 토큰화
word2Index = {}
for voca in token:
    if voca not in word2Index.keys():
        word2Index[voca] = len(word2Index)

# {'나': 0, '는': 1, '자연어': 2, '처리': 3, '를': 4, '배운다': 5}
print(word2Index)


def one_hot_encoding(word, word2Index):
    one_hot_vector = [0] * (len(word2Index))

    index = word2Index[word]
    one_hot_vector[index] = 1

    return one_hot_vector

# [0, 0, 1, 0, 0, 0]
print(one_hot_encoding("자연어", word2Index))


import torch

# 원-핫 벡터 생성
dog = torch.FloatTensor([1, 0, 0, 0, 0])
cat = torch.FloatTensor([0, 1, 0, 0, 0])
computer = torch.FloatTensor([0, 0, 1, 0, 0])
netbook = torch.FloatTensor([0, 0, 0, 1, 0])
book = torch.FloatTensor([0, 0, 0, 0, 1])

print(torch.cosine_similarity(dog, cat, dim=0))
print(torch.cosine_similarity(cat, computer, dim=0))
print(torch.cosine_similarity(computer, netbook, dim=0))
print(torch.cosine_similarity(netbook, book, dim=0))