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

## 정수 인코딩 및 패딩, 어텐션 마스크를 생성하는 함수 정의
# 최대 길이는 128
max_len = 128

def data_to_tensor (sentences, labels, MAX_LEN):
    # 정수 인코딩 과정. 각 텍스트를 토큰화한 후에 Vocabulary에 맵핑되는 정수 시퀀스로 변환한다.
    # ex) ['안녕하세요'] ==> ['안', '녕', '하세요'] ==> [231, 52, 45]
    tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]
    input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]

    # 자연어 처리에서는 남는 공간을 뒤에 두어 패딩 토큰으로 채우는 것이 모델 학습에 더 효과적
    # pad_sequences는 패딩을 위한 모듈. 주어진 최대 길이를 위해서 뒤에서 패딩 토큰의 번호로 채워준다.
    # ex) [231, 52, 45] ==> [231, 52, 45, 패딩 토큰, 패딩 토큰, 패딩 토큰]
    pad_token = tokenizer.encode('<pad>')[0]
    input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, value=pad_token, dtype="long", truncating="post", padding="post") 

    attention_masks = []

    # 어텐션 마스크 생성, 패딩 토큰은 0 / 일반 토큰은 1
    for seq in input_ids:
        seq_mask = [float(i != pad_token) for i in seq]
        attention_masks.append(seq_mask)

    tensor_inputs = torch.tensor(input_ids)
    tensor_labels = torch.tensor(labels)
    tensor_masks = torch.tensor(attention_masks)

    return tensor_inputs, tensor_labels, tensor_masks

## 학습 데이터, 검증 데이터, 테스트 데이터에 대해서
## 정수 인코딩 결과, 레이블, 어텐션 마스크를 각각 inputs, labels, masks에 저장.
train_inputs, train_labels, train_masks = data_to_tensor(train_sentences, train_labels, max_len)
validation_inputs, validation_labels, validation_masks = data_to_tensor(validation_sentences, validation_labels, max_len)
test_inputs, test_labels, test_masks = data_to_tensor(test_sentences, test_labels, max_len)

## 결과 확인
# 첫 번째 샘플의 정수 인코딩 결과
sample_input = test_inputs[0]
sample_label = test_labels[0]
sample_mask = test_masks[0]

# koGPT는 패딩 토큰이 3이다.
# 정수 인코딩 결과: tensor([26098,  9292, 10035,   399,  7888, 12844,  8563,  8488, 40174, 10621,
#         17405,   454, 19889, 24868,  8704, 10656, 11452,   398,  8125, 10034,
#         11324,   419,   430, 50061, 10373,  8263, 10583, 20853,   396,   454,
#         19889, 12734,  8704, 38695, 10355, 10179, 16366, 11324,   419,   430,
#         42880, 11032, 32419,   395,   454, 19889, 24868,  8704, 17629, 14915,
#           397, 11452,   400,  8125, 17329, 10664,  7250,  9016,     3,     3,
#             3,     3,     3,     3,     3,     3,     3,     3,     3,     3,
#             3,     3,     3,     3,     3,     3,     3,     3,     3,     3,
#             3,     3,     3,     3,     3,     3,     3,     3,     3,     3,
#             3,     3,     3,     3,     3,     3,     3,     3,     3,     3,
#             3,     3,     3,     3,     3,     3,     3,     3,     3,     3,
#             3,     3,     3,     3,     3,     3,     3,     3,     3,     3,
#             3,     3,     3,     3,     3,     3,     3,     3])
print("정수 인코딩 결과:", sample_input)
print("-" * 20)

# 원본 문장 복원
decoded_text = tokenizer.decode(sample_input)
# 원본 문장 복원 결과: 오전 10.58시 아우토쿰푸는 2.74pct 하락한 24.87유로, OMX 헬싱키 25지수는 0.55pct 상승한 2,825.14, 
# OMX 헬싱키는 0.64pct 하락한 9,386.89유로에 거래됐다.
# <pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad>
# <pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad>
# <pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad>
# <pad><pad><pad><pad>
print("원본 문장 복원 결과:", decoded_text)
print("-" * 20)

# 어텐션 마스크: tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
#         1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
#        1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
#        1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
#        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
#        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
#        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
#        0., 0.])
print("어텐션 마스크:", sample_mask)
print("-" * 20)

# 샘플의 길이
print("샘플의 길이:", len(sample_input)) # 128
print("-" * 20)

# 레이블
print("레이블:", sample_label) # tensor(0)

print("\n=============================================")

######################################################################
# 배치 학습 준비

batch_size = 32

train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
validation_sampler = SequentialSampler(validation_data)
validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

test_data = TensorDataset(test_inputs, test_masks, test_labels)
test_sampler = RandomSampler(test_data)
test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

######################################################################
# 모델 학습 준비
## 사용할 디바이스 지정
## NVIDIA GPU(cuda) > Apple Silicon GPU(mps) > cpu 순으로 사용 가능한 디바이스 선택
device = torch.device(
    'cuda' if torch.cuda.is_available()
    else 'mps' if torch.backends.mps.is_available()
    else 'cpu'
)
print('사용할 디바이스:', device)

## 모델 로드
## GPT를 사용하여 텍스트 분류시 AutoModelForSequenceClassification를 활용
num_labels = 3
model = AutoModelForSequenceClassification.from_pretrained("skt/kogpt2-base-v2", num_labels=num_labels)
model.to(device)

## 학습 파라미터 및 옵티마이저 설정
epochs = 3
optimizer = AdamW(model.parameters(), lr = 2e-5)

## 평가 함수 정의 (학습 및 테스트 단계에서 사용)
## 모델의 예측과 정답 레이블을 입력으로 받아 정확도(accuracy)와 f1 스코어를 계산하여 반환
def metrics(predictions, labels):
    # predictions: 모델이 예측한 결과값들의 리스트 또는 배열
    # labels: 실제 정답 레이블들의 리스트 또는 배열
    
    y_pred = predictions
    y_true = labels

    # 정확도(accuracy) 계산
    # 전체 예측 중에서 올바르게 예측한 비율
    accuracy = accuracy_score(y_true, y_pred)

    # 매크로 평균 F1 점수 (Macro-avraged F1 Score)
    # 클래스별로 F1 점수 계산 후 평균을 구한다. 모든 클래스를 동등하게 고려
    # zero_division=0 옵션: 분모가 0일 경우 0을 반환하도록 설정
    f1_macro_average = f1_score(y_true=y_true, y_pred=y_pred, average='macro', zero_division=0)

    # 마이크로 평균 F1 점수 (Micro-avraged F1 Score)
    # 전체 데이터에 대해 단일 F1 점수 계산
    # 클래스 불균형이 심할 경우에 적합
    f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average='micro', zero_division=0)

    # 가중 평균 F1 점수 (Weighted-avraged F1 Score)
    # 각 클래스의 F1 점수에 해당 클래스의 샘플 수를 가중치로 곱한 후 평균을 계산
    # 클래스 불균형을 고려
    f1_weighted_average = f1_score(y_true=y_true, y_pred=y_pred, average='weighted', zero_division=0)

    # 계산된 메트릭 결과를 딕셔너리 형태로 리턴
    metrics = {
        'accuracy': accuracy,
        'f1_macro': f1_macro_average,
        'f1_micro': f1_micro_average,
        'f1_weighted': f1_weighted_average
    }
    return metrics

## 학습 함수 정의
## 데이터 로더를 받아서 배치 크기만큼 데이터를 모델에 전달하여 학습
def train_epoch(model, train_dataloader, optimizer, device):
    """
    한 epoch 동안 모델 학습하는 함수

    Parameters:
      - model (torch.nn.Module): 학습시킬 모델 객체.
      - train_dataloader (torch.utils.data.DataLoader): 학습 데이터셋의 DataLoader.
      - optimizer (torch.optim.Optimizer): 최적화 알고리즘을 구현하는 객체.
      - device (torch.device): 학습에 사용할 장치(CPU 또는 CUDA).
    Returns:
      - float: 평균 학습 손실값.
    """
    total_train_loss = 0 # 학습 손실 누적할 변수
    model.train() # 모델을 학습 모드로 설정

    # 학습 데이터 로더를 순회하며 배치 단위로 학습
    for step, batch in tqdm(enumerate(train_dataloader), desc="Training Batch"):
        batch = tuple(t.to(device) for t in batch) # 데이터로더로부터 배치를 받아 각 텐서를 장치로 이동
        b_input_ids, b_input_mask, b_labels = batch # 정수 시퀀스, 어텐션 마스크, 정답 레이블 추출

        # 인퍼런스 및 손실 계산
        outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)

        # 손실값 추출
        loss = outputs.loss

        optimizer.zero_grad() # 기울기(gradient) 초기화
        loss.backward() # 역전파를 통해 기울기(gradient) 계산
        optimizer.step() # 모델 파라미터 업데이트

        total_train_loss += loss.item() # 총 손실에 더함

    avg_train_loss = total_train_loss / len(train_dataloader) # 평균 학습 손실 계산
    return avg_train_loss # 평균 학습 손실 반환


## 검증 데이터셋을 통해 학습된 모델의 성능을 평가하는 함수 정의
def evaluate(model, validation_dataloader, device):
    """
    검증 데이터셋에 대한 평가 수행 함수

    Parameters:
      - model (torch.nn.Module): 평가할 모델 객체.
      - validation_dataloader (torch.utils.data.DataLoader): 검증 데이터셋의 DataLoader.
      - device (torch.device): 평가에 사용할 장치(CPU 또는 CUDA).
    Returns:
      - float: 평균 검증 손실값.
      - dict: 다양한 평가 지표(metrics)에 대한 값들을 담은 사전.
    """

    model.eval() # 모델을 평가 모드로 설정

    total_eval_loss = 0 # 검증 손실을 누적할 변수
    predictions, true_labels = [], [] # 예측값과 실제 레이블값을 저장할 리스트

    # 검증 데이터로더를 순회하며 배치 단위로 평가
    for batch in validation_dataloader:
        batch = tuple(t.to(device) for t in batch) # 배치 데이터를 디바이스로 이동

        b_input_ids, b_input_mask, b_labels = batch # 배치에서 토큰 ID, 어텐션 마스크, 레이블 추출

        with torch.no_grad(): # 기울기 계산 비활성화
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)

        # 모델 출력에서 손실 추출
        if outputs.loss is not None:
            loss = outputs.loss
            total_eval_loss += loss.item()

        logits = outputs.logits.detach().cpu().numpy() # 모델 예측값인 로짓을 numpy 배열로 변환
        label_ids = b_labels.to('cpu').numpy() # 실제 라벨값을 numpy 배열로 변환

        # 3개의 예측값중 가장 큰 값을 예측한 인덱스로 지정
        # 예시) logits = [3.513, -0.309, -2.111] => 0
        predictions.extend(np.argmax(logits, axis=1).flatten()) # 예측된 클래스를 리스트에 추가
        true_labels.extend(label_ids.flatten()) # 실제 레이블 값을 리스트에 추가

    eval_metrics = metrics(predictions, true_labels)

    return total_eval_loss / len(validation_dataloader), eval_metrics

######################################################################
# 모델 학습 진행
min_val_loss = float('inf') # 최소 검증 손실 초기화

## 학습 전 검증 데이터셋으로 한번 평가
avg_val_loss, eval_metrics = evaluate(model, validation_dataloader, device)

# Validation Loss: 1.37
# Accuracy: 0.23
# F1 Macro: 0.21
# F1 Micro: 0.23
# F1 Weighted: 0.26
print(" Validation Loss: {0:.2f}".format(avg_val_loss))
print(" Accuracy: {0:.2f}".format(eval_metrics['accuracy']))
print(" F1 Macro: {0:.2f}".format(eval_metrics['f1_macro']))
print(" F1 Micro: {0:.2f}".format(eval_metrics['f1_micro']))
print(" F1 Weighted: {0:.2f}".format(eval_metrics['f1_weighted']))

## 학습 및 평가 루프

# ======== Epoch 1 / 3 ========
# Training Batch: 97it [01:00,  1.61it/s]

# Running Validation...
#  Validation Loss: 0.43
#  Accuracy: 0.83
#  F1 Macro: 0.78
#  F1 Micro: 0.83
#  F1 Weighted: 0.83
# Validation loss decreased (inf --> 0.43). Saving model ...
# ======== Epoch 2 / 3 ========
# Training Batch: 97it [00:59,  1.62it/s]

# Running Validation...
#  Validation Loss: 0.44
#  Accuracy: 0.84
#  F1 Macro: 0.79
#  F1 Micro: 0.84
#  F1 Weighted: 0.83
# ======== Epoch 3 / 3 ========
# Training Batch: 97it [00:59,  1.63it/s]

# Running Validation...
#  Validation Loss: 0.57
#  Accuracy: 0.81
#  F1 Macro: 0.78
#  F1 Micro: 0.81
#  F1 Weighted: 0.81
for epoch_i in range(0, epochs):
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))

    # 학습 단계
    train_epoch(model, train_dataloader, optimizer, device)
    print("\nRunning Validation...")

    # 검증 데이터셋을 통한 모델 성능 평가 및 검증
    avg_val_loss, eval_metrics = evaluate(model, validation_dataloader, device)
    print(" Validation Loss: {0:.2f}".format(avg_val_loss))
    print(" Accuracy: {0:.2f}".format(eval_metrics['accuracy']))
    print(" F1 Macro: {0:.2f}".format(eval_metrics['f1_macro']))
    print(" F1 Micro: {0:.2f}".format(eval_metrics['f1_micro']))
    print(" F1 Weighted: {0:.2f}".format(eval_metrics['f1_weighted']))

    # 검증 손실이 현재까지의 검증 손실 최소값보다 작은 경우, 모델 체크포인트 저장
    if avg_val_loss < min_val_loss:
        print(f"Validation loss decreased ({min_val_loss:.2f} --> {avg_val_loss:.2f}). Saving model ...")
        # 베스트 모델 저장
        torch.save(model.state_dict(), 'model_checkpoint.pt')
        # 최소 검증 손실 업데이트
        min_val_loss = avg_val_loss

######################################################################
# 모델 로드 및 평가

## 검증 데이터셋 기준 베스트 모델 로드 후 테스트 데이터셋으로 평가
model.load_state_dict(torch.load("model_checkpoint.pt"))

avg_val_loss, eval_metrics = evaluate(model, test_dataloader, device)

#  Test Loss: 0.40
#  Accuracy: 0.84
#  F1 Macro: 0.82
#  F1 Micro: 0.84
#  F1 Weighted: 0.84
print(" Test Loss: {0:.2f}".format(avg_val_loss))
print(" Accuracy: {0:.2f}".format(eval_metrics['accuracy']))
print(" F1 Macro: {0:.2f}".format(eval_metrics['f1_macro']))
print(" F1 Micro: {0:.2f}".format(eval_metrics['f1_micro']))
print(" F1 Weighted: {0:.2f}".format(eval_metrics['f1_weighted']))

## 추론
## pipeline 함수 사용
##   원래 임의의 데이터에 대해서 예측 결과를 얻으려면 예측 함수를 만들어야 하는데,
##   pipeline은 이런 과정을 자동화
from transformers import pipeline

## 현재 풀고자 하는 문제, 모델 및 토크나이저를 알려주면 임의의 입력에 대해 예측 가능
## text-classification: 텍스트 분류 문제 (현재 3개의 레이블에 대한 다중 클래스 분류 문제를 풀고 있다.)
## max_length: 입력의 최대 길이는 512
## return_all_scores=True: 예측시 모든 카테고리에 대해서 점수를 반환하도록 설정
## function_to_apply='softmax': 각 스코어는 총합이 1이 되도록 설정
pipe = pipeline("text-classification", model=model.to(device), tokenizer=tokenizer, 
                device=0, max_length=512, return_all_scores=True, function_to_apply='softmax')

## 인퍼런스 테스트
result = pipe('SK하이닉스가 매출이 급성장하였다')
# [{'label': 'LABEL_1', 'score': 0.7257257699966431}]
print(result)

## 한글로 레이블 예측 결과를 얻는 함수 정의 및 사용
label_dict = {'LABEL_0' : '중립', 'LABEL_1' : '긍정', 'LABEL_2' : '부정'}
def prediction(text):
    result = pipe(text)
    return [label_dict[result[0]['label']]]

# 긍정
print(prediction('네이버가 매출이 급성장하였다'))
# 부정
print(prediction('ChatGPT의 등장으로 인공지능 스타트업들은 위기다'))
# 중립
print(prediction('인공지능 기술의 발전으로 누군가는 기회를 얻을것이고, 누군가는 얻지 못할 것이다'))

# 긍정
print(prediction('SK하이닉스(000660)가 4% 넘게 상승하고 있다. 글로벌 투자은행(IB)들이 메모리 반도체주의 주가 조정이 마무리 국면에 접어들었다는 진단을 내놓으면서 투자심리가 회복된 영향으로 풀이된다.'))
# 부정
print(prediction('미국 반도체 기업의 주가 하락과 외국인 및 기관 투자자의 대규모 매도세가 맞물리면서 국내 증시 대장주인 삼성전자와 SK하이닉스 주가가 동반 폭락했다.'))
