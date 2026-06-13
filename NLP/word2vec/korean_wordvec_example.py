import pandas as pd
import matplotlib.pyplot as plt
import urllib.request
from gensim.models.word2vec import Word2Vec
from konlpy.tag import Okt
from tqdm import tqdm

# 네이버 영화 리뷰 데이터 다운로드
urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings.txt", filename="ratings.txt")

train_data = pd.read_table('ratings.txt')
print(train_data[:5]) # 상위 5개 출력
print(f"리뷰 갯수: {len(train_data)}") # 리뷰 개수 출력

# 전처리
## 결측값 제거
train_data = train_data.dropna(how = 'any') # Null 값이 존재하는 행 제거
print(train_data.isnull().values.any()) # Null 값이 존재하는지 확인
print(f"결측값 제거 후 리뷰 갯수: {len(train_data)}") # 리뷰 개수 출력

## 정규 표현식을 통한 한글 외 문자 제거
train_data['document'] = train_data['document'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","", regex=True)
print(train_data[:5]) # 상위 5개 출력

## 불용어 제거 및 토큰화
### 불용어 정의
stopwords = ['의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다']

### 형태소 분석기 OKT를 사용한 토큰화 작업 (다소 시간 소요)
okt = Okt()
tokenized_data = []
for sentence in tqdm(train_data['document']):
    tokenized_sentence = okt.morphs(sentence, stem=True) # 토큰화
    stopwords_removed_sentence = [word for word in tokenized_sentence if not word in stopwords] # 불용어 제거
    tokenized_data.append(stopwords_removed_sentence)

### 리뷰 길이 분포 확인
print('리뷰의 최대 길이 :',max(len(review) for review in tokenized_data))
print('리뷰의 평균 길이 :',sum(map(len, tokenized_data))/len(tokenized_data))
plt.hist([len(review) for review in tokenized_data], bins=50)
plt.xlabel('length of samples')
plt.ylabel('number of samples')
plt.show()

# Word2Vec 훈련
## 파이썬의 gensim 패키지에는 Word2Vec을 지원
## gensim 패키지를 이용하면 손쉽게 단어를 임베딩 벡터로 변환가능

from gensim.models import Word2Vec

print("----------------------------")
print("학습")
model = Word2Vec(sentences = tokenized_data, vector_size = 100, window = 5, min_count = 5, workers = 4, sg = 0)

print(f"완성된 임베딩 매트릭스의 크기 확인: {model.wv.vectors.shape}")

print("----------------------------")
# 학습 후 입력 단어에 대해 가장 유사한 단어를 출력
print(f"최민식과 유사한 단어: {model.wv.most_similar("최민식")}")
print(f"히어로와 유사한 단어: {model.wv.most_similar("히어로")}")
