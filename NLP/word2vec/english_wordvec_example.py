import re
import urllib.request
import zipfile
from lxml import etree
from nltk.tokenize import word_tokenize, sent_tokenize

# 훈련 데이터 다운로드
## 훈련 데이터는 XML이므로 전처리가 필요
## 전처리: XML 문법을 제거하고, <content> 안에 있는 내용을 가져오되, 배경음을 나타내는 단어도 제거
urllib.request.urlretrieve("https://raw.githubusercontent.com/ukairia777/tensorflow-nlp-tutorial/main/09.%20Word%20Embedding/dataset/ted_en-20160408.xml", filename="ted_en-20160408.xml")

targetXML = open('ted_en-20160408.xml', 'r', encoding='UTF8')
target_text = etree.parse(targetXML)

# xml 파일로부터 <content>와 </content> 사이의 내용만 가져온다.
parse_text = '\n'.join(target_text.xpath('//content/text()'))

# 정규 표현식의 sub 모듈을 통해 content 중간에 등장하는 (Audio), (Laughter) 등의 배경음 부분을 제거.
# 해당 코드는 괄호로 구성된 내용을 제거.
content_text = re.sub(r'\([^)]*\)', '', parse_text)

# 입력 코퍼스에 대해서 NLTK를 이용하여 문장 토큰화를 수행.
sent_text = sent_tokenize(content_text)

# 각 문장에 대해서 구두점을 제거하고, 대문자를 소문자로 변환.
normalized_text = []
for string in sent_text:
     tokens = re.sub(r"[^a-z0-9]+", " ", string.lower())
     normalized_text.append(tokens)

# 각 문장에 대해서 NLTK를 이용하여 단어 토큰화를 수행.
result = [word_tokenize(sentence) for sentence in normalized_text]

print('총 샘플의 개수 : {}'.format(len(result)))

# 샘플 3개만 출력
for line in result[:3]:
    print(line)

# Word2Vec 훈련
## 파이썬의 gensim 패키지에는 Word2Vec을 지원
## gensim 패키지를 이용하면 손쉽게 단어를 임베딩 벡터로 변환가능
from gensim.models import Word2Vec
from gensim.models import KeyedVectors

# vector_size: 임베딩 벡터의 차원
# window: 컨텍스트 윈도우 크기
# min_count: 단어 최소 빈도수 제한 (빈도가 적은 단어들은 학습에서 제외)
# workers: 학습을 위한 프로세스 수
# sg: 0은 CBOW, 1은 Skip-gram

print("----------------------------")
print("학습")
model = Word2Vec(sentences=result, vector_size=100, window=5, min_count=5, workers=4, sg=0)

# (21613, 100): 21,613개의 단어가 있고, 각 단어는 100차원 밀집 임베딩으로 표현
print(f"완성된 임베딩 매트릭스의 크기 확인: {model.wv.vectors.shape}")

print("----------------------------")
# 학습 후 입력 단어에 대해 가장 유사한 단어를 출력
model_result = model.wv.most_similar("man")
print(f"man과 유사한 단어: {model_result}")

# 학습된 모델 저장
model.wv.save_word2vec_format('eng_w2v') # 모델 저장
loaded_model = KeyedVectors.load_word2vec_format("eng_w2v") # 모델 로드

print("----------------------------")
# 로드된 모델에 대해 다시 유사한 단어 출력 테스트
model_result = loaded_model.most_similar("man")
print(f"man과 유사한 단어: {model_result}")