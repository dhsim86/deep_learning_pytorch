# GPT-2를 이용한 한국어 뉴스 긍정, 부정 감성 분류

from datasets import load_dataset

import pandas as pd
import numpy as np
from tqdm import tqdm

import torch
from torch.optim import AdamW
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from transformers import AutoTokenizer, PreTrainedTokenizerFast, AutoModelForSequenceClassification

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score


######################################################################
# 데이터셋 준비
import urllib.request

## 데이터셋 다운로드
# urllib.request.urlretrieve("https://raw.githubusercontent.com/ukairia777/finance_sentiment_corpus/main/finance_data.csv", filename="finance_data.csv")

df = pd.read_csv('finance_data.csv')
print('샘플의 개수 :', len(df)) # 4846

#     labels                                           sentence                                       kor_sentence
# 0   neutral  According to Gran, the company has no plans to...  Gran에 따르면, 그 회사는 회사가 성장하고 있는 곳이지만, 모든 생산을 러시아로...
# 1   neutral  Technopolis plans to develop in stages an area...  테크노폴리스는 컴퓨터 기술과 통신 분야에서 일하는 회사들을 유치하기 위해 10만 평...
# 2  negative  The international electronic industry company ...  국제 전자산업 회사인 엘코텍은 탈린 공장에서 수십 명의 직원을 해고했으며, 이전의 ...
# 3  positive  With the new production plant the company woul...  새로운 생산공장으로 인해 회사는 예상되는 수요 증가를 충족시킬 수 있는 능력을 증가...
# 4  positive  According to the company's updated strategy fo...  2009-2012년 회사의 업데이트된 전략에 따르면, Basware는 20% - 4...
print(df.head())

# labels
# neutral     2879
# positive    1363
# negative     604
# Name: count, dtype: int64
print(df["labels"].value_counts()) # 레이블은 중립, 긍정, 부정의 3개 레이블을 가진다.

## 학습을 위해 데이터셋의 레이블('neutral', 'positive', 'negative')을 각각 0,1,2로 변환
df["labels"] = df["labels"].replace(["neutral", "positive", "negative"], [0, 1, 2])

#    labels                                           sentence                                       kor_sentence
# 0       0  According to Gran, the company has no plans to...  Gran에 따르면, 그 회사는 회사가 성장하고 있는 곳이지만, 모든 생산을 러시아로...
# 1       0  Technopolis plans to develop in stages an area...  테크노폴리스는 컴퓨터 기술과 통신 분야에서 일하는 회사들을 유치하기 위해 10만 평...
# 2       2  The international electronic industry company ...  국제 전자산업 회사인 엘코텍은 탈린 공장에서 수십 명의 직원을 해고했으며, 이전의 ...
# 3       1  With the new production plant the company woul...  새로운 생산공장으로 인해 회사는 예상되는 수요 증가를 충족시킬 수 있는 능력을 증가...
# 4       1  According to the company's updated strategy fo...  2009-2012년 회사의 업데이트된 전략에 따르면, Basware는 20% - 4...
print(df.head())

## 변환한 df를 임시 CSV로 저장한 뒤, load_dataset()으로 다시 읽어 DatasetDict 생성
df.to_csv("finance_data_processed.csv", index=False, encoding="utf-8-sig")
all_data = load_dataset(
    "csv",
    data_files={
        "train": "finance_data_processed.csv", # train으로 로드
    },
)

# DatasetDict({
#     train: Dataset({
#         features: ['labels', 'sentence', 'kor_sentence'],
#         num_rows: 4846
#     })
# })
print(all_data)

## 훈련/검증/테스트 데이터셋으로 분할
cs = all_data['train'].train_test_split(0.2, seed=777)
train_cs = cs["train"]
test_cs = cs["test"]

cs = train_cs.train_test_split(0.2, seed=777)
train_cs = cs["train"]
valid_cs = cs["test"]

## 훈련/검증/테스트 데이터셋 행 갯수
print(f"훈련 데이터셋 행 갯수: {len(train_cs)}") # 3100
print(f"검증 데이터셋 행 갯수: {len(valid_cs)}") # 776
print(f"테스트 데이터셋 행 갯수: {len(test_cs)}") # 970

print("\n=============================================")

######################################################################
# 데이터 전처리
## 한국어 뉴스 문장을 파이썬 리스트로 별도 저장
train_sentences = list(train_cs['kor_sentence'])
validation_sentences = list(valid_cs['kor_sentence'])
test_sentences = list(test_cs['kor_sentence'])

## 레이블 별도 저장
train_labels = train_cs['labels']
validation_labels = valid_cs['labels']
test_labels = test_cs['labels']

## 샘플 확인
# [
#   '오전 10.58시 아우토쿰푸는 2.74pct 하락한 24.87유로, OMX 헬싱키 25지수는 0.55pct 상승한 2,825.14, OMX 헬싱키는 0.64pct 하락한 9,386.89유로에 거래됐다.', 
#   '10월부터 12월까지의 판매량은 302 mln 유로로 전년 동기 대비 25.3 pct 증가했다.', 
#   '매디슨, 위스콘신, 2월 6일 - PRNewswire - - 피스카스는 미국 특허청이 상징적인 가위 손잡이에 오렌지색 상표 등록을 허가했다고 발표한다.', 
#   "M-real로 평가된 분석가들 중 총 6명은 ''매수' - ''누적''을 주었고, 3명은 ''보유'', 1명만이 ''매도''를 주었다.", 
#   '주요 양조업체들은 지난해 국내 맥주 판매량을 2004년 2억4592만 리터에서 2억5688만 리터로 4.5% 늘렸다.'
# ]
print(test_sentences[:5])

# [0, 1, 1, 0, 1]
print(test_labels[:5])

print("\n=============================================")

######################################################################
# GPT 토크나이저를 이용한 전처리

## 한국어 GPT 중 하나인 'skt/kogpt2-base-v2'를 사용.
tokenizer = PreTrainedTokenizerFast.from_pretrained(
    'skt/kogpt2-base-v2',
    bos_token='</s>', eos_token='</s>', unk_token='<unk>',
    pad_token='<pad>', mask_token='<mask>')

## 패딩을 수행하는 함수 정의 (서로 다른 길이를 가진 시퀀스들을 동일한 길이로 맞춘다.)
## -> 배치 단위로 데이터 처리시, 배치 내 모든 샘플들을 같은 길이로 맞춰주기 위해 사용
def pad_sequences(sequences, maxlen=None, dtype='int32',
                  padding='pre', truncating='pre', value=0.0):
    """
    시퀀스들을 동일한 길이로 패딩하는 함수

    Args:
        sequences: 패딩할 시퀀스들의 리스트
        maxlen: 최대 시퀀스 길이. None이면 가장 긴 시퀀스 길이 사용
        dtype: 출력 배열의 데이터 타입
        padding: 'pre' (앞쪽 패딩) 또는 'post' (뒤쪽 패딩)
        truncating: 'pre' (앞쪽 자르기) 또는 'post' (뒤쪽 자르기)
        value: 패딩에 사용할 값

    Returns:
        패딩된 numpy 배열
    """
    if not sequences:
        return np.array([])

    # 최대 길이 결정
    if maxlen is None:
        maxlen = max(len(seq) for seq in sequences)

    # 결과 배열 초기화 (0, 패딩으로 초기화)
    if dtype == 'long':
        result = np.full((len(sequences), maxlen), value, dtype=np.int64)
    else:
        result = np.full((len(sequences), maxlen), value, dtype=dtype)

    for i, seq in enumerate(sequences):
        if len(seq) == 0:
            continue

        # 시퀀스가 maxlen보다 긴 경우 자르기
        #   최대 길이가 5이고 시퀀스가 [1,2,3,4,5,6,7]일 때
        #   -> 'pre': [3,4,5,6,7]
        #   -> 'post': [1,2,3,4,5]
        if len(seq) > maxlen:
            if truncating == 'pre':
                seq = seq[-maxlen:]  # 앞쪽 자르기
            else:  # truncating == 'post'
                seq = seq[:maxlen]   # 뒤쪽 자르기

        # 패딩 적용
        #   최대 길이가 5이고 시퀀스가 [1,2,3]일 때
        #   -> 'pre': [0,0,1,2,3]
        #   -> 'post': [1,2,3,0,0]
        if padding == 'pre':
            # 앞쪽 패딩: 뒤쪽부터 채우기
            result[i, -len(seq):] = seq
        else:  # padding == 'post'
            # 뒤쪽 패딩: 앞쪽부터 채우기
            result[i, :len(seq)] = seq

    return result
