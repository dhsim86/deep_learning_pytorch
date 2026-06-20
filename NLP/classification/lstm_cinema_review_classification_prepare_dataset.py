# 네이버 영화 리뷰 데이터에 대한 이해와 전처리
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import urllib.request
from konlpy.tag import Mecab
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from collections import Counter

# 영화 리뷰 데이터 로드
## 학습 및 테스트용 데이터셋 사용
urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt", filename="ratings_train.txt")
urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_test.txt", filename="ratings_test.txt")

train_data = pd.read_table('ratings_train.txt')
test_data = pd.read_table('ratings_test.txt')

# 데이터셋은 id, document,. label 3개의 열로 구성

print('훈련용 리뷰 개수 :',len(train_data)) # 훈련용 리뷰 개수 출력 (150,000)
#          id                                           document  label
# 0   9976970                                아 더빙.. 진짜 짜증나네요 목소리      0
# 1   3819312                  흠...포스터보고 초딩영화줄....오버연기조차 가볍지 않구나      1
# 2  10265843                                  너무재밓었다그래서보는것을추천한다      0
# 3   9045019                      교도소 이야기구먼 ..솔직히 재미는 없다..평점 조정      0
# 4   6483659  사이몬페그의 익살스런 연기가 돋보였던 영화!스파이더맨에서 늙어보이기만 했던 커스틴 ...      1
print(train_data[:5])

print('테스트용 리뷰 개수 :',len(test_data)) # 테스트용 리뷰 개수 출력 (50,000)
#         id                                           document  label
# 0  6270596                                                굳 ㅋ      1
# 1  9274899                               GDNTOPCLASSINTHECLUB      0
# 2  8544678             뭐야 이 평점들은.... 나쁘진 않지만 10점 짜리는 더더욱 아니잖아      0
# 3  6825595                   지루하지는 않은데 완전 막장임... 돈주고 보기에는....      0
# 4  6723715  3D만 아니었어도 별 다섯 개 줬을텐데.. 왜 3D로 나와서 제 심기를 불편하게 하죠??      0
print(test_data[:5])

############################################################################
## 데이터 정제
### document 열과 label 열의 중복을 제외한 값의 개수, (146182, 2)
### 4,000 여개의 중복 샘플이 존재
print(train_data['document'].nunique(), train_data['label'].nunique())

### document 열의 중복 제거 후 샘플 수 확인
train_data.drop_duplicates(subset=['document'], inplace=True)
print('총 샘플의 수 :',len(train_data)) # 146183

### 학습 데이터셋에서 레이블 값의 분포 확인
train_data['label'].value_counts().plot(kind = 'bar')

### 비교적 레이블 분포가 균일하다.
#    label  count
# 0      0  73342
# 1      1  72841
print(train_data.groupby('label').size().reset_index(name = 'count'))

### null 값 가진 샘플 확인
print(train_data.isnull().values.any())

### document 열에 null을 가진 샘플 하나가 존재
# id          0
# document    1
# label       0
# dtype: int64
print(train_data.isnull().sum())

### null 샘플 인덱스 확인
#             id document  label
# 25857  2172111      NaN      1
print(train_data.loc[train_data.document.isnull()])

### null 샘플 제거
train_data = train_data.dropna(how = 'any') # Null 값이 존재하는 행 제거
print(train_data.isnull().values.any()) # Null 값이 존재하는지 확인

############################################################################
## 데이터 전처리
### 리뷰 텍스트 데이터에서 한글 및 공백만 남기고 모두 제거
train_data['document'] = train_data['document'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","", regex=True)

### 리뷰 텍스트에서 한글 및 공백빼고 모두 제거됨
#          id                                           document  label
# 0   9976970                                  아 더빙 진짜 짜증나네요 목소리      0
# 1   3819312                         흠포스터보고 초딩영화줄오버연기조차 가볍지 않구나      1
# 2  10265843                                  너무재밓었다그래서보는것을추천한다      0
# 3   9045019                          교도소 이야기구먼 솔직히 재미는 없다평점 조정      0
# 4   6483659  사이몬페그의 익살스런 연기가 돋보였던 영화스파이더맨에서 늙어보이기만 했던 커스틴 던...      1
print(train_data[:5])

### 다시 null 값을 가진(한글이 원래 없었던) 샘플 확인 후 제거
### 공백을 제거 후 공백만 있는 데이터는 null로 변경
train_data['document'] = train_data['document'].str.replace('^ +', "", regex=True) # white space 데이터를 empty value로 변경
train_data['document'].replace('', np.nan, inplace=True)
print(train_data.isnull().sum()) # 789
train_data = train_data.dropna(how = 'any')  # Null 값이 존재하는 행 제거
print(len(train_data))

############################################################################
## 테스트 데이터셋에 대해서도 정제 및 전처리
test_data.drop_duplicates(subset = ['document'], inplace=True) # document 열에서 중복인 내용이 있다면 중복 제거
test_data['document'] = test_data['document'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","", regex=True) # 정규 표현식 수행
test_data['document'] = test_data['document'].str.replace('^ +', "", regex=True) # 공백은 empty 값으로 변경
test_data['document'].replace('', np.nan, inplace=True) # 공백은 Null 값으로 변경
test_data = test_data.dropna(how='any') # Null 값 제거
print('전처리 후 테스트용 샘플의 개수 :',len(test_data))

############################################################################
## 토큰화
### 불용어 제거 후 형태소 분석기를 통해 토큰화

### 불용어 정의
stopwords = ['도', '는', '다', '의', '가', '이', '은', '한', '에', '하', '고', '을', '를', '인', '듯', '과', '와', '네', '들', '듯', '지', '임', '게']

### 형태소 분석기로 KoNLPy의 Okt 사용
from konlpy.tag import Okt
okt = Okt()
print(okt.morphs('와 이런 것도 영화라고 차라리 뮤직비디오를 만드는 게 나을 뻔'))

### 학습 데이터셋의 리뷰 텍스트를 토큰화 및 불용어 제거
X_train = []
for sentence in tqdm(train_data['document']):
    tokenized_sentence = okt.morphs(sentence) # 토큰화
    stopwords_removed_sentence = [word for word in tokenized_sentence if not word in stopwords] # 불용어 제거
    X_train.append(stopwords_removed_sentence)

### 샘플 확인
print(X_train[:3])

### 테스트 데이터셋의 리뷰 텍스트를 토큰화 및 불용어 제거
X_test = []
for sentence in tqdm(test_data['document']):
    tokenized_sentence = okt.morphs(sentence) # 토큰화
    stopwords_removed_sentence = [word for word in tokenized_sentence if not word in stopwords] # 불용어 제거
    X_test.append(stopwords_removed_sentence)

############################################################################
## 검증 데이터 추가 준비
### 정답 레이블의 균형 비율을 고려하면서 분할

y_train = np.array(train_data['label'])
y_test = np.array(test_data['label'])

### 검증 데이터셋 분할
### stratify으로 y_train을 지정하여 정답 레이블의 균형을 고려하면서 분할
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=0, stratify=y_train)

print('--------학습 데이터의 비율-----------')
print(f'부정 리뷰 = {round(np.sum(y_train==0)/len(y_train) * 100,3)}%')
print(f'긍정 리뷰 = {round(np.count_nonzero(y_train)/len(y_train) * 100,3)}%')
print('--------검증 데이터의 비율-----------')
print(f'부정 리뷰 = {round(np.sum(y_valid==0)/len(y_valid) * 100,3)}%')
print(f'긍정 리뷰 = {round(np.count_nonzero(y_valid)/len(y_valid) * 100,3)}%')
print('--------테스트 데이터의 비율-----------')
print(f'부정 리뷰 = {round(np.sum(y_test==0)/len(y_test) * 100,3)}%')
print(f'긍정 리뷰 = {round(np.count_nonzero(y_test)/len(y_test) * 100,3)}%')

############################################################################
## 학습/검증/테스트 데이터셋 저장 및 재로드
dataset_path = 'lstm_cinema_review_dataset.pkl'

with open(dataset_path, 'wb') as f:
    pickle.dump(
        {
            'X_train': X_train,
            'X_valid': X_valid,
            'X_test': X_test,
            'y_train': y_train,
            'y_valid': y_valid,
            'y_test': y_test,
        },
        f,
    )

with open(dataset_path, 'rb') as f:
    loaded_dataset = pickle.load(f)

loaded_X_train = loaded_dataset['X_train']
loaded_X_valid = loaded_dataset['X_valid']
loaded_X_test = loaded_dataset['X_test']
loaded_y_train = loaded_dataset['y_train']
loaded_y_valid = loaded_dataset['y_valid']
loaded_y_test = loaded_dataset['y_test']

print('--------저장 후 재로드 확인-----------')
print('X_train:', len(loaded_X_train))
print('X_valid:', len(loaded_X_valid))
print('X_test :', len(loaded_X_test))
print('y_train:', len(loaded_y_train))
print('y_valid:', len(loaded_y_valid))
print('y_test :', len(loaded_y_test))

