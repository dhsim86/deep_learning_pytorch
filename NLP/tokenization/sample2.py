# 토큰화

en_text = "A Dog Run back corner near spare bedrooms"

## 영어의 경우 대표적인 토큰화 도구로 sapCy와 NLTK가 있음
### spacy
### 실행 전 python -m spacy download en_core_web_sm 로 영어 모델 다운로드 필요
import spacy
spacy_en = spacy.load('en_core_web_sm')
def tokenize(en_text):
    return [tok.text for tok in spacy_en.tokenizer(en_text)]

# ['A', 'Dog', 'Run', 'back', 'corner', 'near', 'spare', 'bedrooms']
print(tokenize(en_text))

### NLTK
import nltk
nltk.download('punkt')

from nltk.tokenize import word_tokenize
# ['A', 'Dog', 'Run', 'back', 'corner', 'near', 'spare', 'bedrooms']
print(word_tokenize(en_text))

### 띄어쓰기로 토큰화
# ['A', 'Dog', 'Run', 'back', 'corner', 'near', 'spare', 'bedrooms']
print(en_text.split())

## 한국어 
kor_text = "사과의 놀라운 효능이라는 글을 봤어. 그래서 오늘 사과를 먹으려고 했는데 사과가 썩어서 슈퍼에 가서 사과랑 오렌지 사왔어"

### 띄어쓰기로 토큰화
# ['사과의', '놀라운', '효능이라는', '글을', '봤어.', '그래서', '오늘', '사과를', '먹으려고', '했는데', '사과가', '썩어서', '슈퍼에', '가서', '사과랑', '오렌지', '사왔어']
print(kor_text.split())

### 형태소 토큰화
#from konlpy.tag import Mecab
#tokenizer = Mecab()
# ['사과', '의', '놀라운', '효능', '이', '라는', '글', '을', '봤', '어', '.', '그래서', '오늘', '사과', '를', '먹', '으려고', '했', '는데', '사과', '가', '썩', '어서', '슈퍼', '에', '가', '서', '사과', '랑', '오렌지', '사', '왔', '어']
# print(tokenizer.morphs(kor_text))

#### OKT 사용
from konlpy.tag import Okt
okt = Okt()
# ['사과', '의', '놀라운', '효능', '이라는', '글', '을', '봤어', '.', '그래서', '오늘', '사과', '를', '먹으려고', '했는데', '사과', '가', '썩어서', '슈퍼', '에', '가서', '사과', '랑', '오렌지', ' 사왔어']
print(okt.morphs(kor_text))