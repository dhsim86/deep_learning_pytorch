import gensim
import urllib.request

# 구글의 사전 훈련된 Word2Vec 모델을 로드.
#urllib.request.urlretrieve("https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz", \
#                            filename="GoogleNews-vectors-negative300.bin.gz")

word2vec_model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)

# (3000000, 300): 3백만의 단어가 있고, 각 단어는 300차원 밀집 임베딩으로 표현
print(f"임베딩 매트릭스의 크기 확인: {word2vec_model.vectors.shape}")

# 두 단어 유사도 평가
print(word2vec_model.similarity('this', 'is'))
print(word2vec_model.similarity('post', 'book'))

# book의 벡터 출력
print(word2vec_model['book'])