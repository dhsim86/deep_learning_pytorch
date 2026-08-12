# BERT를 이용한 금융 뉴스 긍정/부정 분류

from datasets import load_dataset

import pandas as pd
import numpy as np
import datetime
from tqdm import tqdm

import torch
from torch.optim import AdamW
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from transformers import BertTokenizer
from transformers import BertForSequenceClassification, BertConfig

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

######################################################################
# 데이터셋 준비
df = pd.read_csv("finance_data.csv")
print("샘플의 개수 :", len(df)) # 4846

#     labels  ...                                       kor_sentence
# 0   neutral  ...  Gran에 따르면, 그 회사는 회사가 성장하고 있는 곳이지만, 모든 생산을 러시아로...
# 1   neutral  ...  테크노폴리스는 컴퓨터 기술과 통신 분야에서 일하는 회사들을 유치하기 위해 10만 평...
# 2  negative  ...  국제 전자산업 회사인 엘코텍은 탈린 공장에서 수십 명의 직원을 해고했으며, 이전의 ...
# 3  positive  ...  새로운 생산공장으로 인해 회사는 예상되는 수요 증가를 충족시킬 수 있는 능력을 증가...
# 4  positive  ...  2009-2012년 회사의 업데이트된 전략에 따르면, Basware는 20% - 4...
print(df.head())

# [5 rows x 3 columns]
# labels
# neutral     2879
# positive    1363
# negative     604
# Name: count, dtype: int64
print(df["labels"].value_counts()) # 레이블은 중립, 긍정, 부정의 3개 레이블을 가진다.

## 학습을 위해 데이터셋의 레이블('neutral', 'positive', 'negative')을 각각 0,1,2로 변환
df["labels"] = df["labels"].replace(["neutral", "positive", "negative"], [0, 1, 2])

#    labels  ...                                       kor_sentence
# 0       0  ...  Gran에 따르면, 그 회사는 회사가 성장하고 있는 곳이지만, 모든 생산을 러시아로...
# 1       0  ...  테크노폴리스는 컴퓨터 기술과 통신 분야에서 일하는 회사들을 유치하기 위해 10만 평...
# 2       2  ...  국제 전자산업 회사인 엘코텍은 탈린 공장에서 수십 명의 직원을 해고했으며, 이전의 ...
# 3       1  ...  새로운 생산공장으로 인해 회사는 예상되는 수요 증가를 충족시킬 수 있는 능력을 증가...
# 4       1  ...  2009-2012년 회사의 업데이트된 전략에 따르면, Basware는 20% - 4...
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
## 데이터셋에서 학습에 쓸 kor_sentence에 대해 '[CLS] 문장 [SEP]' 구조로 변환
train_sentences = list(map(lambda x: '[CLS] ' + str(x) + ' [SEP]', train_cs['kor_sentence']))
validation_sentences = list(map(lambda x: '[CLS] ' + str(x) + ' [SEP]', valid_cs['kor_sentence']))
test_sentences = list(map(lambda x: '[CLS] ' + str(x) + ' [SEP]', test_cs['kor_sentence']))

## 레이블 별도 저장
train_labels = train_cs['labels']
validation_labels = valid_cs['labels']
test_labels = test_cs['labels']

## 샘플 확인
# [
#   '[CLS] 핀란드의 라우트 프리시젼은 멕시코와 미국으로부터 대규모 유리 배치 공장 및 모르타르 공장을 수주했다. [SEP]', 
#   '[CLS] `` 우리는 K-city market 고객들에게 흥미롭고 주제적인 선택을 제공함으로써 그들에게 서비스를 제공하고 싶습니다. [SEP]', 
#   '[CLS] 2007년에도 후타마키는 유기성장을 위한 투자를 계속할 예정이다. [SEP]', 
#   '[CLS] 건물 및 주택 개량 무역에서 순매출은 총 1,173만 유로로 전년도의 1,566만 유로에 비해 감소했다. [SEP]', 
#    [CLS] 우리는 시스템의 구현 모델을 개발하기 위해 장기적인 투자를 해왔습니다. [SEP]'
# ]
print(train_sentences[10:15])

######################################################################
# BERT 토크나이저를 이용한 전처리
## 한국어 BERT 중 하나인 'klue/bert-base'를 사용.
tokenizer = BertTokenizer.from_pretrained('klue/bert-base')

## 토큰화 및 정수 인코딩 테스트
tokenized_text = tokenizer.tokenize('[CLS] 우리는 시스템의 구현 모델을 개발하기 위해 장기적인 투자를 해왔습니다. [SEP]')
input_id = tokenizer.convert_tokens_to_ids(tokenized_text)

# ['[CLS]', '우리', '##는', '시스템', '##의', '구현', '모델', '을', '개발', '##하기', '위해', '장기', '##적인', '투자', '##를', '해왔', '##습', '##니다', '.', '[SEP]']
print('토큰화된 문장 :', tokenized_text)
# [2, 3616, 2259, 4119, 2079, 6948, 4347, 1498, 3720, 31302, 3627, 4376, 31221, 3703, 2138, 19540, 2219, 3606, 18, 3]
print('정수 인코딩된 문장 :', input_id)

print("\n=============================================")

## BERT 토크나이저를 활용한 토큰화 및 정수 인코딩 / 패딩 함수 정의
max_len = 128
def data_to_tensor(sentences, labels, max_len):
    # 정수 인코딩 과정. 각 텍스트를 토큰화한 후에 Vocabulary에 맵핑되는 정수 시퀀스로 변환한다.
    # ex) ['안녕하세요'] ==> ['안', '녕', '하세요'] ==> [231, 52, 45]
    tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]
    input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]

    # 패딩
    # 각 시퀀스를 max_len으로 조정 (truncate 및 padding)
    # ex) [231, 52, 45] ==> [231, 52, 45, 0, 0, 0] (max_len=6일 때)
    padded_input_ids = []
    for seq in input_ids:
        if len(seq) < max_len:
            padded_seq = seq + [0] * (max_len - len(seq))
        else:
            padded_seq = seq[:max_len]
        padded_input_ids.append(padded_seq)

    # 실제 토큰 위치에는 1, 패딩 토큰 위치에는 0을 넣은 리스트인 어텐션 마스크 제작.
    # 정수 인코딩 결과가 [231, 52, 45, 0, 0, 0]이 있다면 231, 52, 45는 실제 토큰이고 0은 패딩 토큰이므로
    # 어텐션 마스크는 [1, 1, 1, 0, 0, 0]
    attention_masks = []
    for seq in padded_input_ids:
        seq_mask = [float(i > 0) for i in seq]
        attention_masks.append(seq_mask)

    tensor_inputs = torch.tensor(padded_input_ids, dtype=torch.long)
    tensor_labels = torch.tensor(labels)
    tensor_masks = torch.tensor(attention_masks)

    return tensor_inputs, tensor_labels, tensor_masks

## 훈련, 검증, 테스트 데이터에 대해 토큰화, 정수 인코딩 후 토큰ID, 레이블, 어텐션 마스크를 구한다.
train_inputs, train_labels, train_masks = data_to_tensor(train_sentences, train_labels, max_len)
validation_inputs, validation_labels, validation_masks = data_to_tensor(validation_sentences, validation_labels, max_len)
test_inputs, test_labels, test_masks = data_to_tensor(test_sentences, test_labels, max_len)

## 샘플 1번에 대해 전처리 결과 확인
# 정수 인코딩 결과: tensor([    2,  4400,  3633,    18,  6264,  2067,     1,    22,    18,  6946,
#          2006, 11453,  4720,  2470,  4056,    18,  7725,  2298,  2200,    16,
#            51,  2110,  2312,  1915,  2942,  2089,  3912, 18618,  2259,    20,
#            18,  5547,  2006, 11453,  4131,  2470,    22,    16,  7569,  2049,
#            18,  3909,    16,    51,  2110,  2312,  1915,  2942,  2089,  2259,
#            20,    18,  6350,  2006, 11453,  4720,  2470,    29,    16, 23010,
#            18,  7948,  2298,  2200,  2170,  4083,  2367,  2062,    18,     3,
#             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
#             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
#             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
#             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
#             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
#             0,     0,     0,     0,     0,     0,     0,     0])
# --------------------
# 원본 문장: [CLS] 오전 10.58시 아우토쿰푸는 2.74pct 하락한 24.87유로, OMX 헬싱키 25지수는 0.55pct 상승한 2,825.14, OMX 헬싱키는 0.64pct 하락한 9,386.89유로에 거래됐다. [SEP]
# --------------------
# 원본 문장 복원 결과: [CLS] 오전 10. 58시 [UNK] 2. 74pct 하락한 24. 87유로, OMX 헬싱키 25지수는 0. 55pct 상승한 2, 825. 14, OMX 헬싱키는 0. 64pct 하락한 9, 386. 89유로에 거래됐다. [SEP] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD]
# --------------------
# 어텐션 마스크: tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
#         1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
#         1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
#         1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0.,
#         0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
#         0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
#         0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
#         0., 0.])
# --------------------
# 샘플의 길이: 128
# --------------------
# 레이블: tensor(0)
print('정수 인코딩 결과:', test_inputs[0])
print('-' * 20)
print('원본 문장:', test_sentences[0])
print('-' * 20)
print('원본 문장 복원 결과:', tokenizer.decode(test_inputs[0]))
print('-' * 20)
print('어텐션 마스크:', test_masks[0])
print('-' * 20)
print('샘플의 길이:', len(test_inputs[0]))
print('-' * 20)
print('레이블:', test_labels[0])

print("\n=============================================")

######################################################################
# 배치 학습 준비
batch_size = 32

train_data = TensorDataset(train_inputs, train_masks, train_labels) # 학습 데이터 및 어텐션 마스크, 레이블로 생성
## 데이터에서 무작위로 데이터 추출하는 샘플러
## 모델 학습시 데이터를 랜덤하게 섞어 과적합 방지에 도움을 준다.
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
## 데이터를 순차적으로 샘플링하는 샘플러
## 검증 과정에서는 데이터 순서를 유지
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

## 텍스트 분류하기 위해 'BertForSequenceClassification' 사용. num_labels에 레이블 갯수를 지정
num_labels = 3
model = BertForSequenceClassification.from_pretrained("klue/bert-base", num_labels=num_labels)
## model.cuda()는 모델의 파라미터와 버퍼를 NVIDIA GPU 메모리로 옮기는 메서드(= model.to('cuda'))다.
## model.cuda()는 NVIDIA GPU 전용. to(device)를 쓰면 cuda/mps/cpu 어디서든 동작한다.
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

# Validation Loss: 1.01
# Accuracy: 0.53
# F1 Macro: 0.33
# F1 Micro: 0.53
# F1 Weighted: 0.48
print(" Validation Loss: {0:.2f}".format(avg_val_loss))
print(" Accuracy: {0:.2f}".format(eval_metrics['accuracy']))
print(" F1 Macro: {0:.2f}".format(eval_metrics['f1_macro']))
print(" F1 Micro: {0:.2f}".format(eval_metrics['f1_micro']))
print(" F1 Weighted: {0:.2f}".format(eval_metrics['f1_weighted']))

## 학습 및 평가 루프

# ======== Epoch 1 / 3 ========
# Training Batch: 97it [00:51,  1.89it/s]

# Running Validation...
# Validation Loss: 0.44
#  Accuracy: 0.83
#  F1 Macro: 0.79
#  F1 Micro: 0.83
#  F1 Weighted: 0.83
# Validation loss decreased (inf --> 0.44). Saving model ...
# ======== Epoch 2 / 3 ========
# Training Batch: 97it [00:51,  1.90it/s]

# Running Validation...
#  Validation Loss: 0.39
#  Accuracy: 0.84
#  F1 Macro: 0.81
#  F1 Micro: 0.84
#  F1 Weighted: 0.84
# Validation loss decreased (0.44 --> 0.39). Saving model ...
# ======== Epoch 3 / 3 ========
# Training Batch: 97it [00:54,  1.77it/s]

# Running Validation...
#  Validation Loss: 0.44
#  Accuracy: 0.84
#  F1 Macro: 0.80
#  F1 Micro: 0.84
#  F1 Weighted: 0.84
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

#  Test Loss: 0.37
#  Accuracy: 0.85
#  F1 Macro: 0.85
#  F1 Micro: 0.85
#  F1 Weighted: 0.85
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
# [{'label': 'LABEL_1', 'score': 0.968245267868042}]
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
