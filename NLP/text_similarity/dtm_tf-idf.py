import pandas as pd  # 데이터프레임 사용을 위해
from math import log  # IDF 계산을 위해

docs = [
    "먹고 싶은 사과",
    "먹고 싶은 바나나",
    "길고 노란 바나나 바나나",
    "저는 과일이 좋아요",
]
vocab = list(set(w for doc in docs for w in doc.split()))
vocab.sort()

# ['과일이', '길고', '노란', '먹고', '바나나', '사과', '싶은', '저는', '좋아요']
print(vocab)

# TF-IDF 수식을 계산하는 메서드 정의
## 총 문서의 수
N = len(docs)


def tf(t, d):
    return d.count(t)


def idf(t):
    df = 0
    for doc in docs:
        df += t in doc
    return log(N / (df + 1))


def tfidf(t, d):
    return tf(t, d) * idf(t)


result = []

# 각 문서에 대해서 아래 연산을 반복
for i in range(N):
    result.append([])
    d = docs[i]
    for j in range(len(vocab)):
        t = vocab[j]
        result[-1].append(tf(t, d))

# DTM 생성
tf_ = pd.DataFrame(result, columns=vocab)

#   과일이  길고  노란  먹고  바나나  사과  싶은  저는  좋아요
# 0    0   0   0   1    0   1   1   0    0
# 1    0   0   0   1    1   0   1   0    0
# 2    0   1   1   0    2   0   0   0    0
# 3    1   0   0   0    0   0   0   1    1
print(tf_)

# IDF 계산
result = []
for j in range(len(vocab)):
    t = vocab[j]
    result.append(idf(t))

idf_ = pd.DataFrame(result, index=vocab, columns=["IDF"])

#          IDF
# 과일이  0.693147
# 길고   0.693147
# 노란   0.693147
# 먹고   0.287682
# 바나나  0.287682
# 사과   0.693147
# 싶은   0.287682
# 저는   0.693147
# 좋아요  0.693147
print(idf_)

# TF-IDF 행렬 생성
result = []
for i in range(N):
    result.append([])
    d = docs[i]
    for j in range(len(vocab)):
        t = vocab[j]
        result[-1].append(tfidf(t, d))

tfidf_ = pd.DataFrame(result, columns=vocab)

#        과일이        길고        노란        먹고       바나나        사과        싶은        저는       좋아요
# 0  0.000000  0.000000  0.000000  0.287682  0.000000  0.693147  0.287682  0.000000  0.000000
# 1  0.000000  0.000000  0.000000  0.287682  0.287682  0.000000  0.287682  0.000000  0.000000
# 2  0.000000  0.693147  0.693147  0.000000  0.575364  0.000000  0.000000  0.000000  0.000000
# 3  0.693147  0.000000  0.000000  0.000000  0.000000  0.000000  0.000000  0.693147  0.693147
print(tfidf_)

# 사이킷런을 이용한 DTM과 TF-IDF 실습
## CountVectorizer를 사용하면 DTM 생성 가능
from sklearn.feature_extraction.text import CountVectorizer

corpus = [
    'you know I want your love',
    'I like you',
    'what should I do ',    
]

vector = CountVectorizer()

# 코퍼스로부터 각 단어의 빈도수를 기록
# [[0 1 0 1 0 1 0 1 1]
#  [0 0 1 0 0 0 0 1 0]
#  [1 0 0 0 1 0 1 0 0]]
print(vector.fit_transform(corpus).toarray())

# 각 단어와 맵핑된 인덱스 출력
## {'you': 7, 'know': 1, 'want': 5, 'your': 8, 'love': 3, 'like': 2, 'what': 6, 'should': 4, 'do': 0}
print(vector.vocabulary_)

# 사이킷런은 TF-IDF를 자동 계산해주는 TfidfVectorizer를 제공
from sklearn.feature_extraction.text import TfidfVectorizer

corpus = [
    'you know I want your love',
    'I like you',
    'what should I do ',    
]

tfidfv = TfidfVectorizer().fit(corpus)

# [[0.         0.46735098 0.         0.46735098 0.         0.46735098 0.         0.35543247 0.46735098]
#  [0.         0.         0.79596054 0.         0.         0.         0.         0.60534851 0.        ]
#  [0.57735027 0.         0.         0.         0.57735027 0.         0.57735027 0.         0.        ]]
print(tfidfv.transform(corpus).toarray())

## {'you': 7, 'know': 1, 'want': 5, 'your': 8, 'love': 3, 'like': 2, 'what': 6, 'should': 4, 'do': 0}
print(tfidfv.vocabulary_)