# 한국어 KorNLI 데이터셋을 가지고, 두 개의 문장을 입력받아 관계를 분류하는 문제

from datasets import load_dataset

import pandas as pd
import numpy as np
import random
import time
import datetime
from tqdm import tqdm

import torch
from torch.optim import AdamW
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from transformers import BertTokenizer
from transformers import get_linear_schedule_with_warmup
from transformers import BertForSequenceClassification, BertConfig

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

######################################################################
# 데이터셋 준비

## KorNLI 데이터 중 학습 데이터 로드
cs = load_dataset("klue/klue", "nli", split="train")

## KorNLI 데이터 중 테스트 데이터 로드
test_cs = load_dataset("klue/klue", "nli", split="validation")

# 학습 데이터를 9:1로 분할 후 각각 학습/검증 데이터로 사용
cs = cs.train_test_split(0.1, seed=777)
train_cs = cs["train"]
valid_cs = cs["test"]

# {
#   'guid': 'klue-nli-v1_train_21438', 
#   'source': 'policy', 
#   'premise': '최근에는 새로운 서핑 장소로도 각광받고 있다.',
#   'hypothesis': '요즘에 각광받는 새로운 서핑 장소다.',
#   'label': 0
# }
print(train_cs[0])

print("\n=============================================")
######################################################################
# 데이터셋 전처리

## 학습/검증/테스트 데이터(premise 및 hypothesis 컬럼)에 대해 [CLS] 문장 [SEP] 구조로 변환
## [CLS] premise [SEP] hypothesis [SEP]
train_sentences = list(map(lambda train_cs: '[CLS] ' + str(train_cs['premise']) +' [SEP] ' + str(train_cs['hypothesis']) + ' [SEP]', train_cs))
validation_sentences = list(map(lambda valid_cs: '[CLS] ' + str(valid_cs['premise']) + ' [SEP] ' + str(valid_cs['hypothesis']) + ' [SEP]', valid_cs))
test_sentences = list(map(lambda test_cs: '[CLS] ' + str(test_cs['premise']) + ' [SEP] ' + str(test_cs['hypothesis']) + ' [SEP]', test_cs))

# [
#   '[CLS] 우리는 마당에서 물놀이도 할 수 있었다 [SEP] 마당이 넓고 좋았다. [SEP]', 
#   '[CLS] 미국의 상원법사 위원회는 코미 전 국장에게 러시아 대선 개입 관련 메모를 제출하도록 요구하였고, 백악관도 관련 녹취기록을 요구하였다. [SEP] 미국의 상원법사 위원회는 백악관에 녹취기록 요구는 하지 않았다. [SEP]', 
# ...
print(validation_sentences[:5])

## 레이블 별도 저장
train_labels = train_cs['label']
validation_labels = valid_cs['label']
test_labels = test_cs['label']

# [1, 2, 2, 0, 1] (0: 함의 관계 / 1: 중립 관계 / 2: 모순 관계)
print(validation_labels[:5])

print("\n=============================================")
######################################################################
# BERT 토크나이저를 통한 전처리
tokenizer = BertTokenizer.from_pretrained('klue/bert-base')

## 토크나이저 로드 후 토큰화 및 정수 인코딩
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

# 정수 인코딩 결과: tensor([    2, 25313,  2377,  2031,  2073, 20812,  2116,  1513,  2259,  1129,
#         24094, 20812, 27135,  9753,  2052,  3662, 11800,    18,     3,  3711,
#          1129, 27135,  2119,  9753,  2073,  5040,  3598,  3606,    18,     3,
#             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
#             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
#             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
#             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
#             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
#             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
#             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
#             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
#             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
#             0,     0,     0,     0,     0,     0,     0,     0])
# --------------------
# 원본 문장: [CLS] 흡연자분들은 발코니가 있는 방이면 발코니에서 흡연이 가능합니다. [SEP] 어떤 방에서도 흡연은 금지됩니다. [SEP]
# --------------------
# 원본 문장 복원 결과: [CLS] 흡연자분들은 발코니가 있는 방이면 발코니에서 흡연이 가능합니다. [SEP] 어떤 방에서도 흡연은 금지됩니다. [SEP] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD]
# --------------------
# 어텐션 마스크: tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
#         1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0.,
#         0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
#         0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
#         0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
#         0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
#         0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
#         0., 0.])
# --------------------
# 샘플의 길이: 128
# --------------------
# 레이블: tensor(2)
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

print("\n=============================================")

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

# Validation Loss: 1.15
# Accuracy: 0.33
# F1 Macro: 0.28
# F1 Micro: 0.33
# F1 Weighted: 0.28
print(" Validation Loss: {0:.2f}".format(avg_val_loss))
print(" Accuracy: {0:.2f}".format(eval_metrics['accuracy']))
print(" F1 Macro: {0:.2f}".format(eval_metrics['f1_macro']))
print(" F1 Micro: {0:.2f}".format(eval_metrics['f1_micro']))
print(" F1 Weighted: {0:.2f}".format(eval_metrics['f1_weighted']))

## 학습 및 평가 루프

# ======== Epoch 1 / 3 ========
# Training Batch: 704it [06:38,  1.77it/s]

# Running Validation...
#  Validation Loss: 0.45
#  Accuracy: 0.81
#  F1 Macro: 0.81
#  F1 Micro: 0.81
#  F1 Weighted: 0.81
# Validation loss decreased (inf --> 0.45). Saving model ...
# ======== Epoch 2 / 3 ========
# Training Batch: 704it [06:18,  1.86it/s]

# Running Validation...
#  Validation Loss: 0.46
#  Accuracy: 0.83
#  F1 Macro: 0.83
#  F1 Micro: 0.83
#  F1 Weighted: 0.83
# ======== Epoch 3 / 3 ========
# Training Batch: 704it [06:27,  1.82it/s]

# Running Validation...
#  Validation Loss: 0.52
#  Accuracy: 0.83
#  F1 Macro: 0.82
#  F1 Micro: 0.83
#  F1 Weighted: 0.83
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

# Test Loss: 0.56
# Accuracy: 0.77
# F1 Macro: 0.77
# F1 Micro: 0.77
# F1 Weighted: 0.77
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
## BERT의 입력으로 '[CLS] 문장1 [SEP] 문장2 [SEP]'와 같은 형태의 입력을
## pipeline 도구에서 진행할려면 반드시 {"text": 문장 1, "text_pair": 문장 2} 형태로 입력을 사용
inputs = {
    "text" : "흡연자분들은 발코니가 있는 방이면 발코니에서 흡연이 가능합니다.", 
    "text_pair" : "어떤 방에서도 흡연은 금지됩니다."
}

# [{'label': 'LABEL_2', 'score': 0.5892113447189331}]
result = pipe([inputs])
print(result)

## 한글로 레이블 예측 결과를 얻는 함수 정의 및 사용
label_dict = {'LABEL_0' : '얽힘', 'LABEL_1' : '중립', 'LABEL_2' : '모순'}
def prediction(sent1, sent2):
    text = {"text" : sent1, "text_pair" : sent2}
    result = pipe(text)
    return [label_dict[result['label']]]

# 모순
sent1 = "흡연자분들은 발코니가 있는 방이면 발코니에서 흡연이 가능합니다."
sent2 = "어떤 방에서도 흡연은 금지됩니다."
print(prediction(sent1, sent2))

# 중립
sent1 = "저는 , 그냥 알아내려고 거기 있었어요."
sent2 = "나는 돈이 어디로 갔는지 이해하려고 했어요."
print(prediction(sent1, sent2))

# 얽힘
sent1 = "저는 그것을 이해하려고 거기 있었어요."
sent2 = "저는 이해하려고 노력하고 있었어요."
print(prediction(sent1, sent2))

# 모순
sent1 = "주식이 올라서 수익률이 아주 좋다."
sent2 = "주식이 올랐지만 아직 수익률은 마이너스이다."
print(prediction(sent1, sent2))

# 중립
sent1 = "주식이 올라서 수익률이 아주 좋다."
sent2 = "앞으로 주식 매도 후 리밸런싱할 예정이다."
print(prediction(sent1, sent2))

# 중립
sent1 = "주식이 올라서 수익률이 아주 좋다."
sent2 = "현재 주식 포트폴리오를 구성하는 종목 중 일부 종목의 수익률이 좋다."
print(prediction(sent1, sent2))