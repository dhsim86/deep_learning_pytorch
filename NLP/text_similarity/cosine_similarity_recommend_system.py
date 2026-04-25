# 좋아하는 영화를 입력시 해당 영화의 줄거리와 유사한 줄거리의 영화를 추천시스템 구현

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

############################################################################################################
# 데이터 준비

## 데이터셋 확인
data = pd.read_csv("NLP/text_similarity/movies_metadata.csv", low_memory=False)

#    adult                              belongs_to_collection    budget  ...  video vote_average vote_count
# 0  False  {'id': 10194, 'name': 'Toy Story Collection', ...  30000000  ...  False          7.7     5415.0
# 1  False                                                NaN  65000000  ...  False          6.9     2413.0
# [2 rows x 24 columns]
print(data.head(2))

# title과 overview 컬럼을 사용
## 상위 2만개의 샘플을 data에 저장
data = data.head(20000)

############################################################################################################
# TF-IDF를 쓰기 위한 데이터셋 보정 및 TF-IDF 구해놓기

## TF-IDF를 연산할 때 데이터에 Null 값이 들어있으면 에러가 발생하므로,
## TF-IDF의 대상이 되는 data의 overview 열에 결측값에 해당하는 Null 값이 있는지 확인
## overview 열에 존재하는 모든 null값을 전부 카운트하여 출력
print('overview 열의 결측값의 수:',data['overview'].isnull().sum())

## null값을 가진 행을 제거하는 pandas의 dropna()을 쓰거나, 특정 값으로 채워넣는 fillna()를 사용
## 여기서는 fillna()로 ''으로 대체
data['overview'] = data['overview'].fillna('')

# overview 컬럼에 대한 TF-IDF 구하기
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(data['overview'])

# TF-IDF 행렬의 크기(shape) : (20000, 47487)
## -> 20,000개의 영화를 표현하기 위해 47,487 단어가 사용됨. 벡터의 차원도 47,487
print('TF-IDF 행렬의 크기(shape) :',tfidf_matrix.shape)

############################################################################################################
# 코사인 유사도 구해놓기
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# 코사인 유사도 연산 결과 : (20000, 20000)
## -> 20,000개의 각 문서 벡터와 자기 자신을 포함한 20,000개의 문서 벡터간의 유사도
print('코사인 유사도 연산 결과 :',cosine_sim.shape)

############################################################################################################
# 추천 시스템 준비

# 영화의 타이틀을 key로, 영화의 인덱스를 value로 하는 딕셔너리 생성
title_to_index = dict(zip(data['title'], data.index))

# 영화 제목 Father of the Bride Part II의 인덱스를 리턴
idx = title_to_index['Father of the Bride Part II']
print(idx)

# 영화의 제목을 입력하면 코사인 유사도를 통해 가장 overview가 유사한 10개의 영화를 찾아내는 함수
def get_recommendations(title, cosine_sim=cosine_sim):
    # 선택한 영화의 타이틀로부터 해당 영화의 인덱스를 받아온다.
    idx = title_to_index[title]

    # 해당 영화와 모든 영화와의 유사도를 가져온다.
    sim_scores = list(enumerate(cosine_sim[idx]))

    # 유사도에 따라 영화들을 정렬한다.
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # 가장 유사한 10개의 영화를 받아온다.
    sim_scores = sim_scores[1:11]

    # 가장 유사한 10개의 영화의 인덱스를 얻는다.
    movie_indices = [idx[0] for idx in sim_scores]

    # 가장 유사한 10개의 영화의 제목을 리턴한다.
    return data['title'].iloc[movie_indices]

############################################################################################################
# 다크나이트 라이즈와 유사한 영화를 검색

# 12481                            The Dark Knight
# 150                               Batman Forever
# 1328                              Batman Returns
# 15511                 Batman: Under the Red Hood
# 585                                       Batman
# 9230          Batman Beyond: Return of the Joker
# 18035                           Batman: Year One
# 19792    Batman: The Dark Knight Returns, Part 1
# 3095                Batman: Mask of the Phantasm
# 10122                              Batman Begins
print(get_recommendations('The Dark Knight Rises'))