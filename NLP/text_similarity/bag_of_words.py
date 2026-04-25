from konlpy.tag import Okt

okt = Okt()


def build_bag_of_words(document):
    # 온점 제거 및 형태소 분석
    document = document.replace(".", "")
    tokenized_document = okt.morphs(document)

    word_to_index = {}
    bow = []

    for word in tokenized_document:
        if word not in word_to_index.keys():
            word_to_index[word] = len(word_to_index)  # 인덱스 부여
            # BoW에 전부 기본값 1을 넣는다.
            bow.insert(len(word_to_index) - 1, 1)
        else:
            # 재등장하는 단어의 인덱스
            index = word_to_index.get(word)
            # 재등장한 단어는 해당하는 인덱스의 위치에 1을 더한다.
            bow[index] = bow[index] + 1

    return word_to_index, bow


doc1 = "정부가 발표하는 물가상승률과 소비자가 느끼는 물가상승률은 다르다."
vocab, bow = build_bag_of_words(doc1)
print("vocabulary :", vocab)
print("bag of words vector :", bow)

# 사이킷 런의 BoW 예제
## 단어의 빈도를 Count하여 Vector로 만드는 CountVectorizer 클래스를 지원
from sklearn.feature_extraction.text import CountVectorizer

corpus = ["you know I want your love. because I love you."]
vector = CountVectorizer()

# 코퍼스로부터 각 단어의 빈도수를 기록
## bag of words vector : [[1 1 2 1 2 1]]
print("bag of words vector :", vector.fit_transform(corpus).toarray())

# 각 단어의 인덱스가 어떻게 부여되었는지를 출력
## 불용어인 'I'는 제거됨 (CountVectorizer는 길이가 2 이상인 문자만 토큰으로 인식)
## 영어에서는 길이가 짧은 문자를 제거하는 것 또한 전처리 작업으로 고려
## vocabulary : {'you': 4, 'know': 1, 'want': 3, 'your': 5, 'love': 2, 'because': 0}
print("vocabulary :", vector.vocabulary_)

# CountVectorizer는 띄어쓰기만을 기준으로 토큰화를 진행하여 BoW를 만드므로, 한국어에서는 쓸 수 없다.
corpus = ["정부가 발표하는 물가상승률과 소비자가 느끼는 물가상승률은 다르다."]

## bag of words vector : [[1 1 1 1 1 1 1]]
print("bag of words vector :", vector.fit_transform(corpus).toarray())

## '물가상승률과'와 '물가상승률은' 으로 조사를 포함해서 하나의 단어로 판단하여 서로 다른 두 단어로 인식
## vocabulary : {'정부가': 6, '발표하는': 4, '물가상승률과': 2, '소비자가': 5, '느끼는': 0, '물가상승률은': 3, '다르다': 1}
print("vocabulary :", vector.vocabulary_)

# 불용어를 제거한 BoW
## CountVectorizer는 불용어를 지정하면, 불용어는 제외하고 BoW를 만들 수 있도록 지원
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords

## 사용자가 직접 정의한 불용어 사용
text = ["Family is not an important thing. It's everything."]
vect = CountVectorizer(stop_words=["the", "a", "an", "is", "not"])

### bag of words vector : [[1 1 1 1 1]]
print('bag of words vector :',vect.fit_transform(text).toarray())
### vocabulary : {'family': 1, 'important': 2, 'thing': 4, 'it': 3, 'everything': 0}
print('vocabulary :',vect.vocabulary_)

## CountVectorizer에서 제공하는 자체 불용어 사용
vect = CountVectorizer(stop_words="english")

### bag of words vector : [[1 1 1]]
print('bag of words vector :',vect.fit_transform(text).toarray())
### vocabulary : {'family': 0, 'important': 1, 'thing': 2}
print('vocabulary :',vect.vocabulary_)

## NLTK에서 지원하는 불용어 사용
stop_words = stopwords.words("english")
vect = CountVectorizer(stop_words=stop_words)

### bag of words vector : [[1 1 1 1]]
print('bag of words vector :',vect.fit_transform(text).toarray()) 
### vocabulary : {'family': 1, 'important': 2, 'thing': 3, 'everything': 0}
print('vocabulary :',vect.vocabulary_)