# RAG 파인튜닝을 위한 학습 데이터셋 확인

import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset

dataset = load_dataset("iamjoon/klue-mrc-ko-rag-dataset")
df = dataset['train'].to_pandas()

## 이 데이터셋에서 다음 컬럼을 사용
## question(질문), search_result(질문에 대한 검색 결과), answer(최종 답변), 
## extracted_ref_numbers(답변에서 인용된 문서 번호의 리스트), type(데이터 유형) 
df = df[['question', 'search_result', 'answer', 'extracted_ref_numbers', 'type']]
print(df.head())

## 어떤 타입이 있는지 확인 (RAG에서 발생할 수 있는 다양한 시나리오에 대응)
# ['mrc_question' 'mrc_question_with_1_to_4_negative' 'synthetic_question' 'paraphrased_question' 'no_answer']
# mrc_question
#   - 장소나 이름, 날짜 등을 묻는 지엽적인 질문. 검색 결과(search_result)는 5개로 고정.
# mrc_question_with_1_to_4_negative
#   - 장소나 이름, 날짜 등을 묻는 지엽적인 질문. 검색 결과(search_result)는 1~4개 사이인 유형.
#   - 학습 데이터 전체의 검색 결과가 5개로 동일하다면(mrc_question)
#     학습 후 RAG가 동작할 때, 검색 결과가 5개가 아닌 경우 모델의 성능이 떨어지는 것을 방지
# paraphrased_question
#   - 장소나 이름, 날짜 등을 묻는 지엽적인 질문이며 질문의 형태가 문장이 아닌 명사구의 형태를 가진다.
#   - 검색 결과(search_result)는 5개로 고정.
# synthetic_question
#   - 이유, 장점, 단점 등과 같은 포괄적인 질문.
#   - 검색 결과(search_result)는 5개로 고정.
#   - 포괄적인 질문이므로 일반적으로 다수의 문서를 인용하게 된다는 특징
#   - 일반적으로 인용 문서 번호 (extracted_ref_numbers)의 값이 2개 이상
#   - 명사구 형태의 질문을 했을 때 모델 성능이 저하되는 현상을 방지하기 위해 추가된 데이터
# no_answer
#   - 질문에 대한 답이 검색 결과에 없는 데이터.
#   - 답변(answer)에는 검색 결과에 질문에 대한 답이 없다고 안내해야만 한다.
#   - 검색 결과(search_result, 네거티브 샘플)는 5개로 고정.
print('데이터 타입 종류:',df['type'].unique())

print("\n=============================================")

######################################################################
# mrc_question 타입
## 질문이 굉장히 지엽적인 데이터 유형

print('5번 샘플의 타입:', df['type'].loc[5]) # mrc_question
print('5번 샘플의 질문:', df['question'].loc[5]) # 에티하드 웰니스 프로그램의 일환으로 위생에 관한 정보를 제공하는 것은 누구인가?

# 에티하드 웰니스 프로그램의 일환으로 위생에 관한 정보를 제공하는 사람은
# 특별 훈련 과정을 거친 에티하드항공의 웰니스 엠버서더입니다. 
# 이들은 여행 전 과정에 걸쳐 조언과 건강 및 위생 조치에 대한
# 세부 사항을 공유하며 맞춤화된 정보를 제공합니다 [[ref3]].
print('5번 샘플의 답변:', df['answer'].loc[5])
print('5번 샘플의 답변에서 인용한 문서 번호:', df['extracted_ref_numbers'].loc[5]) # [3]

## 문서 검색 수
print('5번 샘플의 검색 결과 개수:', len(df['search_result'].loc[5])) # 5

## 답변을 위해 인용한 문서를 확인

# 5번 샘플의 검색 결과 중 세번째 문서: 아랍 에미리트의 국영항공사 에티하드항공
# …중략…
# 예약 과정에서부터 공항 이용은 물론 항공여행에 이르기까지 인공지능 기술을 비롯한 최신 기술을 
# 과감히 도입해 광범위한 예방 조치를 시행하고 있으며 업계 최초로 선보인 에티하드 웰니스 프로그램에서는 
# 특별 훈련 과정을 거친 에티하드항공의 웰니스 엠버서더가 여행 전 과정에 걸친 조언과 건강 및 위생 조치에 대한
# 세부 사항을 공유하며 맞춤화된 정보를 제공한다. 
# …중략…
# 신종코로나바이러스 상황으로 인해 올해 2020비즈니스 트래블러 중동 어워드는 온라인으로 개최되었으며 독자의 투표를 기반으로 선정됐다.
print('5번 샘플의 검색 결과 중 세번째 문서:', df['search_result'].loc[5][2])

