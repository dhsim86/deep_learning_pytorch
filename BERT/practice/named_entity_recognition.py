# BERT를 이용한 개체명 인식

from datasets import load_dataset

import os

import pandas as pd
import numpy as np
import datetime
from tqdm import tqdm

import urllib.request

import torch
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from transformers import BertTokenizer
from transformers import BertForTokenClassification, BertConfig

from seqeval.metrics import f1_score, classification_report

######################################################################
# 데이터셋 준비

## 훈련/테스트 데이터, 레이블 정보를 다운로드
# urllib.request.urlretrieve("https://raw.githubusercontent.com/ukairia777/tensorflow-nlp-tutorial/main/18.%20Fine-tuning%20BERT%20(Cls%2C%20NER%2C%20NLI)/dataset/ner_train_data.csv", filename="ner_train_data.csv")
# urllib.request.urlretrieve("https://raw.githubusercontent.com/ukairia777/tensorflow-nlp-tutorial/main/18.%20Fine-tuning%20BERT%20(Cls%2C%20NER%2C%20NLI)/dataset/ner_test_data.csv", filename="ner_test_data.csv")
# urllib.request.urlretrieve("https://raw.githubusercontent.com/ukairia777/tensorflow-nlp-tutorial/main/18.%20Fine-tuning%20BERT%20(Cls%2C%20NER%2C%20NLI)/dataset/ner_label.txt", filename="ner_label.txt")

train_ner_df = pd.read_csv("ner_train_data.csv")
test_ner_df = pd.read_csv("ner_test_data.csv")

#                                             Sentence                                                Tag
# 0                      정은 씨를 힘들게 한 가스나그, 가만둘 수 없겠죠 .                              PER-B O O O O O O O O
# 1                          ▶ 쿠마리 한동수가 말하는 '가넷 & 에르덴'                      O PER-B PER-I O PER-B O PER-B
# 2                    슈나이더의 프레젠테이션은 말 청중을 위한 특별한 쇼다 .                            PER-B O O CVL-B O O O O
# 3  지구 최대 연료탱크 수검 회사 구글이 연내 22명 안팎의 인력을 갖춘 연구개발(R&...  O O TRM-B O O ORG-B DAT-B NUM-B O O O ORG-B LO...
# 4  5. <10:00:TI_HOUR> 도이치증권대 <0:1:QT_SPORTS> 연예오락...                              NUM-B O ORG-B O ORG-B
print(train_ner_df.head())

print("학습 데이터 샘플 개수 :", len(train_ner_df)) # 81000
print("테스트 데이터 샘플 개수 :", len(test_ner_df)) # 9000

## 훈련/테스트 데이터에서 문장과 레이블을 각각 분리하여 저장
train_data_sentence = [sent.split() for sent in train_ner_df['Sentence'].values]
test_data_sentence = [sent.split() for sent in test_ner_df['Sentence'].values]
train_data_label = [tag.split() for tag in train_ner_df['Tag'].values]
test_data_label = [tag.split() for tag in test_ner_df['Tag'].values]

## 이 데이터는 형태소 단위가 아니라 어절 단위(띄어쓰기 단위)로 개체명 인식 레이블이 태깅되어 있음
# ['슈나이더의', '프레젠테이션은', '말', '청중을', '위한', '특별한', '쇼다', '.']
print(train_data_sentence[2])
# ['PER-B', 'O', 'O', 'CVL-B', 'O', 'O', 'O', 'O']
print(train_data_label[2])

## 개체명 태깅 정보의 종류 확인
# ['O', 'PER-B', 'PER-I', 'FLD-B', 'FLD-I', 'AFW-B', 'AFW-I', 
#  'ORG-B', 'ORG-I', 'LOC-B', 'LOC-I', 'CVL-B', 'CVL-I', 
#  'DAT-B', 'DAT-I', 'TIM-B', 'TIM-I', 'NUM-B', 'NUM-I', 
#  'EVT-B', 'EVT-I', 'ANM-B', 'ANM-I', 'PLT-B', 'PLT-I', 
#  'MAT-B', 'MAT-I', 'TRM-B', 'TRM-I']
labels = [label.strip() for label in open('ner_label.txt', 'r', encoding='utf-8')]
print('개체명 태깅 정보 :', labels)

## 개체명 태깅 정보를 저장한 lables로부터 태깅 정보와 정수를 매핑하는 딕셔너리 생성
tag_to_index = {tag: index for index, tag in enumerate(labels)} # 태깅 정보 -> 정수 매핑
index_to_tag = {index: tag for index, tag in enumerate(labels)} # 정수 -> 태깅 정보 매핑

print(tag_to_index) # {'O': 0, 'PER-B': 1, 'PER-I': 2, 'FLD-B': 3, 'FLD-I': ...
print(index_to_tag) # {0: 'O', 1: 'PER-B', 2: 'PER-I', 3: 'FLD-B' ...

## 29개의 개체명 태깅 종류가 있다.
tag_size = len(tag_to_index)
print('개체명 태깅 정보의 개수 :',tag_size) # 29

print("\n=============================================")

######################################################################
# 데이터 전처리 이전에 서브워드 토크나이저 호환 확인
tokenizer = BertTokenizer.from_pretrained("klue/bert-base")

## 임의로 훈련 데이터 중 하나를 골라 전처리
sent = train_data_sentence[1]
label = train_data_label[1]
print('문장 :', sent) # ['▶', '쿠마리', '한동수가', '말하는', "'가넷", '&', "에르덴'"]
print('레이블 :', label) # ['O', 'PER-B', 'PER-I', 'O', 'PER-B', 'O', 'PER-B']
print('레이블의 정수 인코딩 :',[tag_to_index[idx] for idx in label]) # [0, 1, 2, 0, 1, 0, 1]
print('문장의 길이 :', len(sent)) # 7
print('레이블의 길이 :', len(label)) # 7

## BERT 토크나이저는 서브워드로 분리하므로 이를 위한 별도 처리가 필요하다.
tokens = []
for one_word in sent:
    # 각 단어가 서브워드로 분리된다.
    # ex) one_word = '쿠마리' ===> subword_tokens = ['쿠', '##마리']
    # ex) one_word = '한동수가' ===> subword_tokens = ['한동', '##수', '##가']
    subword_tokens = tokenizer.tokenize(one_word)
    tokens.extend(subword_tokens)

# ['▶', '쿠', '##마리', '한동', '##수', '##가', '말', '##하', '##는', "'", '가', '##넷', '&', '에르', '##덴', "'"]
print('BERT 토크나이저 적용후 문장 :', tokens)
print('레이블 :', label) # ['O', 'PER-B', 'PER-I', 'O', 'PER-B', 'O', 'PER-B']
print('레이블의 정수 인코딩 :', [tag_to_index[idx] for idx in label]) #  [0, 1, 2, 0, 1, 0, 1]
print('문장의 길이 :', len(tokens)) # 16
print('레이블의 길이 :', len(label)) # 7

## 단어들이 서브워드로 분리되어 문장의 길이가 레이블의 길이와 달라지므로, 이를 일치시키도록 추가 처리가 필요
## 단어로부터 분리된 서브워드들에 대해서, 첫 번째 서브워드에만 레이블을 부여하고 나머지는 부여하지 않는다.
## -> 첫 번째 서브워드에만 원래 레이블 부여
## -> 나머지 서브워드들은 별도 레이블 정수(-100)를 부여 (이후 학습 때 손실 함수가 이를 무시하도록 적용)
##
## [왜 하필 -100 인가?]
## -100은 "29개 개체명 태그 중 하나"가 아니다. 태그 정수는 0~28이므로 -100은 애초에 유효한 태그가 아니다.
## -100은 손실(오차)을 계산할 때 "이 위치는 아예 없는 것으로 취급하고 건너뛰라"는 뜻으로
## 미리 약속된 표시값(센티널, sentinel)이다.
## 이 약속은 PyTorch가 정한 것으로, nn.CrossEntropyLoss의 ignore_index 인자 기본값이 -100이다.
## -> 그래서 레이블에 -100을 넣어두는 것만으로 별도 코드 없이 손실 계산에서 자동 제외된다.
##    (자세한 확인은 아래 "손실 함수에서 -100 레이블은 제외 예시" 섹션과 학습 루프의 주석 참고)
tokens = []
labels_ids = []
for one_word, label_token in zip(train_data_sentence[1], train_data_label[1]):
    subword_tokens = tokenizer.tokenize(one_word)
    tokens.extend(subword_tokens)
    labels_ids.extend([tag_to_index[label_token]] + [-100] * (len(subword_tokens) - 1))

# ['▶', '쿠', '##마리', '한동', '##수', '##가', '말', '##하', '##는', "'", '가', '##넷', '&', '에르', '##덴', "'"]
print('토큰화 후 문장 :',tokens)
## 레이블 쪽에서 -100은 "학습에서 무시할 자리"를 뜻한다. 문장 길이를 맞추는 패딩 자리에도 -100을 부여한다.
## -> 즉 -100이 붙는 자리는 (1)첫 번째 서브워드가 아닌 자리, (2)[CLS]/[SEP] 자리, (3)패딩 자리 세 종류다.
## 주의: -100은 토크나이저의 [PAD] 토큰 ID(0)와는 전혀 다른 값이다. 아래 print에서 -100을 '[PAD]'라는
##      글자로 바꿔 출력하는 것은 사람이 읽기 편하도록 표시만 그렇게 한 것이며, 실제 값은 -100이다.
# ['O', 'PER-B', '[PAD]', 'PER-I', '[PAD]', '[PAD]', 'O', '[PAD]', '[PAD]', 'PER-B', '[PAD]', '[PAD]', 'O', 'PER-B', '[PAD]', '[PAD]']
print('레이블 :', ['[PAD]' if idx == -100 else index_to_tag[idx] for idx in labels_ids])
print('레이블의 정수 인코딩 :', labels_ids) # [0, 1, -100, 2, -100, -100, 0, -100, -100, 1, -100, -100, 0, 1, -100, -100]
print('문장의 길이 :', len(tokens)) # 16
print('레이블의 길이 :', len(labels_ids)) # 16

print("\n=============================================")

######################################################################
# 데이터 전처리

## 문장과 레이블에 대해 정수 인코딩 및 세그먼트 인코딩과 어텐션 마스크 생성
def convert_examples_to_features(examples, labels, max_seq_len, tokenizer, pad_token_id_for_segment=0, pad_token_id_for_label=-100):
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    pad_token_id = tokenizer.pad_token_id

    input_ids, attention_masks, token_type_ids, data_labels = [], [], [], []

    for example, label in tqdm(zip(examples, labels), total=len(examples)):
        tokens = []
        labels_ids = []

        # 각 단어에 대해 토크나이저를 통해 서브워드 토큰화, 이에 맞게 레이블도 채운다.
        for one_word, label_token in zip(example, label):
            # 하나의 단어에 대해서 서브워드로 토큰화
            subword_tokens = tokenizer.tokenize(one_word)
            tokens.extend(subword_tokens)
            # 서브워드 중 첫번째 서브워드만 개체명 레이블을 부여하고 그외에는 -100으로 채운다.
            # -> 여기서 채운 -100이 나중에 학습 시 손실 계산에서 자동으로 제외되고,
            #    평가 시에도 sequences_to_tags에서 걸러진다. (즉 이 한 줄이 '제외 처리'의 전부다)
            labels_ids.extend([tag_to_index[label_token]] + [pad_token_id_for_label] * (len(subword_tokens) - 1))

        # [CLS]와 [SEP]를 후에 추가할 것을 고려하여 최대 길이를 초과하는 샘플의 경우 max_seq_len - 2의 길이로 변환.
        # ex) max_seq_len = 64라면 길이가 62보다 긴 샘플은 뒷부분을 자르고 길이 62로 변환.
        special_tokens_count = 2
        if len(tokens) > max_seq_len - special_tokens_count:
            tokens = tokens[:(max_seq_len - special_tokens_count)]
            labels_ids = labels_ids[:(max_seq_len - special_tokens_count)]

        # [SEP] 및 [CLS]에 대해 개체명 예측이 의미가 없으므로 -100 레이블 부여
        # [SEP]를 추가하는 코드
        # 1. 토큰화 결과의 맨 뒷부분에 [SEP] 토큰 추가
        # 2. 레이블에도 맨 뒷부분에 -100 추가.
        tokens += [sep_token]
        labels_ids += [pad_token_id_for_label]

        # [CLS]를 추가하는 코드
        # 1. 토큰화 결과의 앞부분에 [CLS] 토큰 추가
        # 2. 레이블의 맨 앞부분에도 -100 추가.
        tokens = [cls_token] + tokens
        labels_ids = [pad_token_id_for_label] + labels_ids

        # 정수 인코딩
        input_id = tokenizer.convert_tokens_to_ids(tokens)

        # 어텐션 마스크 생성
        attention_mask = [1] * len(input_id)

        # 정수 인코딩에 추가할 패딩 길이 연산
        padding_count = max_seq_len - len(input_id)

        # 정수 인코딩, 어텐션 마스크에 패딩 추가
        input_id = input_id + ([pad_token_id] * padding_count)
        attention_mask = attention_mask + ([0] * padding_count)

        # 세그먼트 인코딩. 개체명 인식은 두 개 이상의 문장을 구분할 필요가 없으므로 전부 0으로 채운다.
        token_type_id = [pad_token_id_for_segment] * max_seq_len

        # 레이블 패딩. (단, 이 경우는 패딩 토큰의 ID가 -100)
        label = labels_ids + ([pad_token_id_for_label] * padding_count)

        assert len(input_id) == max_seq_len, "Error with input length {} vs {}". format(len(input_id), max_seq_len)
        assert len(attention_mask) == max_seq_len, "Error with attention mask length {} vs {}".format(len(attention_mask), max_seq_len)
        assert len(token_type_id) == max_seq_len, "Error with token type length {} vs {}".format(len(token_type_id), max_seq_len)
        assert len(label) == max_seq_len, "Error with labels length {} vs {}".format(len(label), max_seq_len)

        input_ids.append(input_id)
        attention_masks.append(attention_mask)
        token_type_ids.append(token_type_id)
        data_labels.append(label)

    input_ids = np.array(input_ids, dtype=int)
    attention_masks = np.array(attention_masks, dtype=int)
    token_type_ids = np.array(token_type_ids, dtype=int)
    data_labels = np.asarray(data_labels, dtype=np.int32)
    
    return (input_ids, attention_masks, token_type_ids), data_labels

## 훈련/테스트 데이터에 대해 전처리
X_train, y_train = convert_examples_to_features(train_data_sentence, train_data_label, max_seq_len=128, tokenizer=tokenizer)
X_test, y_test = convert_examples_to_features(test_data_sentence, test_data_label, max_seq_len=128, tokenizer=tokenizer)

# 기존 원문 : ['정은', '씨를', '힘들게', '한', '가스나그,', '가만둘', '수', '없겠죠', '.']
# 기존 레이블 : ['PER-B', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']
print('기존 원문 :', train_data_sentence[0])
print('기존 레이블 :', train_data_label[0])
print('-' * 50)
# 토큰화 후 원문 : ['[CLS]', '정은', '씨', '##를', '힘들', '##게', '한', '가스', '##나', '##그', ',', '가만', '##둘', '수', '없', '##겠', '##죠', '.', '[SEP]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]']
# 토큰화 후 레이블 : ['[PAD]', 'PER-B', 'O', '[PAD]', 'O', '[PAD]', 'O', 'O', '[PAD]', '[PAD]', '[PAD]', 'O', '[PAD]', 'O', 'O', '[PAD]', '[PAD]', 'O', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]']
print('토큰화 후 원문 :', [tokenizer.decode([word]) for word in X_train[0][0]])
print('토큰화 후 레이블 :', ['[PAD]' if idx == -100 else index_to_tag[idx] for idx in y_train[0]])
print('-' * 50)
# 정수 인코딩 결과 : [    2 17915  1370  2138  4390  2318  1891  5809  2075  2029    16  6836
#   3056  1295  1415  2918  2321    18     3     0     0     0     0     0
#      0     0     0     0     0     0     0     0     0     0     0     0
#      0     0     0     0     0     0     0     0     0     0     0     0
#      0     0     0     0     0     0     0     0     0     0     0     0
#      0     0     0     0     0     0     0     0     0     0     0     0
#      0     0     0     0     0     0     0     0     0     0     0     0
#      0     0     0     0     0     0     0     0     0     0     0     0
#      0     0     0     0     0     0     0     0     0     0     0     0
#      0     0     0     0     0     0     0     0     0     0     0     0
#      0     0     0     0     0     0     0     0]
# 정수 인코딩 레이블 : [-100    1    0 -100    0 -100    0    0 -100 -100 -100    0 -100    0
#     0 -100 -100    0 -100 -100 -100 -100 -100 -100 -100 -100 -100 -100
#  -100 -100 -100 -100 -100 -100 -100 -100 -100 -100 -100 -100 -100 -100
#  -100 -100 -100 -100 -100 -100 -100 -100 -100 -100 -100 -100 -100 -100
#  -100 -100 -100 -100 -100 -100 -100 -100 -100 -100 -100 -100 -100 -100
#  -100 -100 -100 -100 -100 -100 -100 -100 -100 -100 -100 -100 -100 -100
#  -100 -100 -100 -100 -100 -100 -100 -100 -100 -100 -100 -100 -100 -100
#  -100 -100 -100 -100 -100 -100 -100 -100 -100 -100 -100 -100 -100 -100
#  -100 -100 -100 -100 -100 -100 -100 -100 -100 -100 -100 -100 -100 -100
#  -100 -100]
print('정수 인코딩 결과 :', X_train[0][0])
print('정수 인코딩 레이블 :', y_train[0])

# 세 그 먼 트 인 코 딩 : [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
# 0 0 0 0 0
# 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
# 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
# 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
# 어 텐 션 마 스 크 : [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0
# 0 0 0 0
# 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
# 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
# 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
print('세그먼트 인코딩 :', X_train[2][0])
print('어텐션 마스크 :', X_train[1][0])

print("\n=============================================")

######################################################################
# 손실 함수에서 -100 레이블은 제외 예시
## 이 섹션이 중요한 이유: 여기서 손으로 확인하는 동작이 곧 BertForTokenClassification 내부에서
## 그대로 일어나는 동작이다. 그래서 학습 루프에서는 labels에 -100을 그냥 넘기기만 하면 된다.
import torch.nn as nn

## nn.CrossEntropy가 레이블이 -100인 위치에서 오차를 어떻게 무시하는지 확인
## batch_size: 4 / 예측해야될 정답의 클래스 갯수: 3
## -> 모델의 출력층은 처음부터 바로 정답 클래스를 예측하는 것이 아니고, 각 레이블에 대한 로짓 벡터를 출력
## -> 출력층에서 출력된 로짓 벡터가 소프트맥스 함수를 통과하면 각 레이블에 대한 확률 분포 벡터가 만들어진다. (합이 1)
## -> 이 중에서 가장 큰 값을 가진 인덱스를 정답 레이블로 간주한다.

# 모델의 예측값
# 첫번째 예측값을 보자. [1.0, 2.0, 3.0] => 2번 인덱스가 가장 값이 크다. 2번으로 예측한 것이다.
# 두번째 예측값을 보자. [2.0, 1.0, 3.0] => 2번 인덱스가 가장 값이 크다. 2번으로 예측한 것이다.
# 세번째 예측값을 보자. [3.0, 2.0, 1.0] => 0번 인덱스가 가장 값이 크다. 0번으로 예측한 것이다.
# 네번째 예측값을 보자. [1.0, 3.0, 2.0] => 1번 인덱스가 가장 값이 크다. 1번으로 예측한 것이다.
# 다시 말해 outputs_with_ignore가 4개의 데이터에 대해서 예측하고 있는 카테고리의 값은 [2, 2, 0, 1]이다.
outputs_with_ignore = torch.tensor([[1.0, 2.0, 3.0],
    [2.0, 1.0, 3.0],
    [3.0, 2.0, 1.0],
    [1.0, 3.0, 2.0]])

# 레이블 (-100인 레이블에 대해서는 오차를 계산하지 않는다.)
## -> 이 때 1번 및 3번 인덱스에 대해서는 오차를 계산하지 않는다. (-100인 레이블 위치의 데이터는 없는 취급)
## -> 즉, 2개 인덱스만 대상으로 오차를 계산
## -> [2, 2, 0, 1]를 모델이 예측했더라도 [2, x, 0, x]에 대해서만 손실을 계산
targets_with_ignore = torch.tensor([2, -100, 0, -100])

# -100을 무시하는 설정으로 손실 계산
loss_fn_with_ignore = nn.CrossEntropyLoss(ignore_index=-100)
loss_with_ignore = loss_fn_with_ignore(outputs_with_ignore, targets_with_ignore)
# 오차
print(f'Loss with ignore_index=-100: {loss_with_ignore.item()}') # 0.40760594606399536

# 위의 데이터에서 -100이 있는 위치의 값을 실제로 제거한 데이터
# 모델의 예측값
outputs = torch.tensor([[1.0, 2.0, 3.0],
    [3.0, 2.0, 1.0]])
# 레이블
targets = torch.tensor([2, 0])
loss_fn = nn.CrossEntropyLoss()

loss = loss_fn(outputs, targets)
# 오차
# 즉, nn.CrossEntropyLoss(ignore_index=-100)로 하고 정답 레이블이 [2, -100, 0, -100]일 때와 오차가 같다.
# -> ignore_index=-100은 "그 위치의 데이터를 애초에 빼고 계산한 것"과 수학적으로 동일하다는 의미다.
#    (평균을 낼 때 분모에서도 빠진다. 위 예시는 4개가 아니라 유효한 2개로 나눈 값이다.)
print(f'calculated loss: {loss.item()}') # 0.40760594606399536

print("\n=============================================")

######################################################################
# 배치 학습 준비
batch_size = 32
# 이 부분은 convert_examples_to_features에서 반환하는 형식에 맞게 텐서를 분할

# 학습 데이터에서 각 입력값(input_ids, attention_masks, token_type_ids)과 레이블(labels)을 추출
train_input_ids, train_attention_masks, train_token_type_ids = X_train
train_labels = y_train

# 테스트 데이터에서 각 입력값(input_ids, attention_masks, token_type_ids)과 레이블(labels)을 추출
test_input_ids, test_attention_masks, test_token_type_ids = X_test
test_labels = y_test

# 학습 데이터의 각 부분을 파이토치 텐서로 변환 (정수형)
train_input_ids = torch.tensor(train_input_ids, dtype=torch.long)
train_attention_masks = torch.tensor(train_attention_masks, dtype=torch.long)
train_token_type_ids = torch.tensor(train_token_type_ids, dtype=torch.long)
train_labels = torch.tensor(train_labels, dtype=torch.long)

# 테스트 데이터의 각 부분을 파이토치 텐서로 변환 (정수형)
test_input_ids = torch.tensor(test_input_ids, dtype=torch.long)
test_attention_masks = torch.tensor(test_attention_masks, dtype=torch.long)
test_token_type_ids = torch.tensor(test_token_type_ids, dtype=torch.long)
test_labels = torch.tensor(test_labels, dtype=torch.long)

# 학습 데이터와 테스트 데이터 텐서들을 하나의 TensorDataset으로 묶음
train_data = TensorDataset(train_input_ids, train_attention_masks, train_token_type_ids, train_labels)
test_data = TensorDataset(test_input_ids, test_attention_masks, test_token_type_ids, test_labels)

# 학습 데이터와 테스트 데이터에 대해서 데이터 로더를 생성, batch_size로 데이터를 묶어 모델에 전달할 준비를 함
train_loader = DataLoader(train_data, batch_size=batch_size)
test_loader = DataLoader(test_data, batch_size=batch_size)

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
### 개체명 인식, 즉 각 입력 토큰 위치의 출력 결과를 각각 분류하기 위한 BertForTokenClassification 사용
### 레이블의 갯수로 태깅 정보 갯수 지정
### -> num_labels=29 이므로 모델의 출력층은 토큰 위치마다 29개의 로짓만 내보낸다.
###    따라서 예측 결과(argmax)는 항상 0~28 사이의 값이며, 모델이 -100을 예측하는 일은 구조적으로 불가능하다.
###    (-100은 '정답 후보'가 아니라 '손실 계산에서 빼라'는 표시값이라는 점을 기억할 것)
### -> 또한 BertForTokenClassification은 labels를 함께 넘겨주면 내부에서 손실까지 계산해 준다.
###    그 손실 함수가 nn.CrossEntropyLoss()이며 ignore_index 기본값이 -100이다. (학습 루프 주석에서 상세 설명)
model = BertForTokenClassification.from_pretrained("klue/bert-base", num_labels=tag_size) 
optimizer = Adam(model.parameters(), lr=5e-5)
model.to(device)

## 정수 시퀀스로부터 개채명 태깅 정보 시퀀스로 변환하는 함수 정의
### f1 score과 같은 정확한 점수를 계산하기 위해, 예측값과 레이블이 정수 시퀀스가 아닌 태깅 정보들의 시퀀스끼리 비교
### -> 모델의 예측값이 [0 1 0 2 0 0]이고, 레이블이 [‑100 1 ‑100 2 ‑100 ‑100]인 경우
###    -100인 위치에 대한 예측값과 레이블을 제외하고 태깅 정보 시퀀스로 변경 후 평가
### -> 여기서 제외하는 위치는 학습 시 손실 계산에서 제외된 위치와 정확히 같다. (학습 기준 == 평가 기준)
###    첫 번째 서브워드가 아닌 위치에서 모델이 어떤 태그를 출력하든 아예 읽지 않으므로 점수에 영향이 없다.
###    예시의 0번 위치처럼 모델이 'O'(0)를 출력해도, 레이블이 -100이면 그 예측값은 버려진다.
def sequences_to_tags(label_ids, pred_ids, index_to_tag):
    # label_ids: 실제 레이블의 인덱스 시퀀스 리스트 (2D 리스트)
    # pred_ids: 예측된 레이블의 인덱스 시퀀스 리스트 (2D 리스트)
    # index_to_tag: 인덱스를 태그로 변환하는 딕셔너리

    label_list = []
    pred_list = []

    # 각 시퀀스에 대해 반복
    for i in range(0, len(label_ids)):
        label_tag = [] # 현재 시퀀스의 실제 레이블 태그들을 저장할 리스트
        pred_tag = [] # 현재 시퀀스의 예측된 레이블 태그들을 저장할 리스트

        # 각 시퀀스의 레이블 및 예측값 쌍에 대해 반복
        ## -100의 경우는 제외한다.
        for label_index, pred_index in zip(label_ids[i], pred_ids[i]):
            if label_index != -100: # 유효하지 않은 레이블 (예: 패딩 등) 제외
                label_tag.append(index_to_tag[label_index]) # 실제 레이블 태그 추가
                pred_tag.append(index_to_tag[pred_index]) # 예측된 레이블 태그 추가

        label_list.append(label_tag) # 현재 시퀀스의 실제 레이블 태그 리스트를 전체 리스트에 추가
        pred_list.append(pred_tag) # 현재 시퀀스의 예측된 레이블 태그 리스트를 전체 리스트에 추가
    return label_list, pred_list # 실제 레이블과 예측된 레이블의 태그 리스트 반환

## 평가 함수 정의
### 모델의 예측값과 실제 레이블을 비교 후 F1 점수와 분류 리포트를 출력
def evaluate(model, test_loader, index_to_tag):
    # 모델을 평가 모드로 전환 (드롭아웃 등 비활성화)
    model.eval()

    total_labels, total_preds = [], [] # 전체 레이블과 예측값을 저장할 리스트 초기화

    # 그레디언트 계산을 비활성화하여 메모리 사용량 및 연산속도를 최적화
    with torch.no_grad():
        # 테스트 데이터 셋의 각 배치에 대해 반복
        for input_ids, attention_masks, token_type_ids, labels in test_loader:
            # 데이터를 GPU 또는 지정된 디바이스로 이동
            input_ids = input_ids.to(device)
            attention_masks = attention_masks.to(device)
            token_type_ids = token_type_ids.to(device)
            labels = labels.to(device)

            # 모델에 입력 데이터를 주고 예측값(logits) 출력
            outputs = model(input_ids, attention_mask=attention_masks, token_type_ids=token_type_ids)

            # 예측값(logits)을 CPU로 이동시키고 넘파이 배열로 변환
            logits = outputs.logits.detach().cpu().numpy()
            # 레이블을 CPU로 이동시키고 넘파이 배열로 변환
            labels = labels.cpu().numpy()

            # 예측값에서 가장 높은 확률의 인덱스를 선택하여 예측된 레이블 생성
            y_predicted = np.argmax(logits, axis=2)

            # 실제 레이블과 예측된 레이블을 태그로 변환
            label_list, pred_list = sequences_to_tags(labels, y_predicted, index_to_tag)

            # 전체 레이블 리스트와 예측 리스트에 현재 배치의 결과를 추가
            total_labels.extend(label_list)
            total_preds.extend(pred_list)

    # 전체 레이블과 예측값에 대한 F1 점수를 계산
    score = f1_score(total_labels, total_preds, suffix=True)

    # F1 점수를 출력
    print(' - f1: {:04.2f}'.format(score * 100))
    # 전체 레이블과 예측값에 대한 분류 리포트를 출력
    print(classification_report(total_labels, total_preds, suffix=True))

print("\n=============================================")

######################################################################
# 모델 학습 진행
## 학습 동안 1 epoch 당 모델 파라미터 업데이트 횟수
steps = len(train_input_ids) // batch_size + 1
print("1 epoch 당 모델 학습 파라미터 업데이트 횟수: ", steps) # 2532

## 모델 학습
epochs = 3

for epoch in range(epochs):
    model.train() # 학습 모드로 전환

    # 학습 데이터 로더에서 배치를 하나씩 가져와서 반복
    for input_ids, attention_masks, token_type_ids, labels in tqdm(train_loader, total=steps):
        # 각 입력 데이터를 GPU 또는 지정된 디바이스로 이동
        input_ids = input_ids.to(device)
        attention_masks = attention_masks.to(device)
        token_type_ids = token_type_ids.to(device)
        labels = labels.to(device)

        # 모델에 데이터를 입력하여 예측값(outputs)을 계산하고 손실(loss)도 계산
        # labels에 -100이 그대로 들어있지만, 여기서 따로 걸러낼 필요가 없다.
        # 레이블에 -100을 넣어둔 것 자체가 이미 "손실 계산에서 제외" 처리이기 때문이다.
        #
        # [왜 별도 처리 없이 그냥 넘겨도 되는가]
        # BertForTokenClassification의 forward 내부는 아래와 같이 구현되어 있다.
        #   (transformers/models/bert/modeling_bert.py)
        #       if labels is not None:
        #           loss_fct = CrossEntropyLoss()                                     # 인자 없이 생성
        #           loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        # 여기서 쓰이는 CrossEntropyLoss는 torch.nn.CrossEntropyLoss이고, 인자를 주지 않았으므로
        # ignore_index가 기본값인 -100으로 설정된다.
        # -> 즉 위쪽 "손실 함수에서 -100 레이블은 제외 예시" 섹션에서 직접 만들어 확인한
        #    nn.CrossEntropyLoss(ignore_index=-100)과 완전히 동일한 손실 함수가 모델 내부에서 사용된다.
        # -> 그 결과 레이블이 -100인 위치는 손실에 전혀 반영되지 않고, 해당 위치의 그레디언트는 정확히 0이 된다.
        #    (그레디언트가 0이면 그 위치 때문에 파라미터가 바뀌는 일이 없다 = 학습되지 않는다)
        # -> 정리하면, 첫 번째 서브워드가 아닌 위치와 [CLS]/[SEP]/[PAD] 위치는 학습 대상에서 빠지고,
        #    각 단어의 첫 번째 서브워드 위치만 학습된다. 원래 의도한 대로 동작하고 있다.
        #
        # [초보자가 흔히 하는 오해] "모델이 -100이라는 값을 예측하도록 학습되는 것인가?"
        # -> 아니다. 다음 두 가지 이유로 애초에 불가능하다.
        #    1) -100은 정답 클래스가 아니다. 출력층은 num_labels=29개의 로짓만 내보내므로
        #       예측값(argmax)은 항상 0~28이다. 모델이 -100을 출력할 경로 자체가 존재하지 않는다.
        #    2) 그 위치는 손실에서 제외되어 그레디언트가 0이므로, 무언가를 "배우는" 일이 일어나지 않는다.
        # -> 그러면 그 위치의 예측값은 어떻게 되는가? 아무 태그나 나올 수 있고, 신경 쓸 필요가 없다.
        #    평가 단계의 sequences_to_tags가 레이블이 -100인 위치를 읽지도 않고 버리기 때문이다.
        #    (학습에서 제외한 위치 == 평가에서 제외하는 위치 이므로 기준이 서로 일치한다)
        #
        # [주의] -100이라는 숫자가 특별한 이유는 순전히 PyTorch의 기본값 약속 때문이다.
        # 만약 convert_examples_to_features의 pad_token_id_for_label을 -1 같은 값으로 바꾸면,
        # 모델이 알아서 무시해주지 않으므로 nn.CrossEntropyLoss(ignore_index=-1)로 손실을 직접 계산해야 한다.
        # (그대로 넘기면 -1은 유효한 클래스 범위(0~28) 밖의 값이라 에러가 발생한다)
        outputs = model(input_ids, attention_mask=attention_masks, token_type_ids=token_type_ids, labels=labels)
        loss = outputs.loss # 손실값 추출

        # 옵티마이저의 그레디언트를 초기화
        optimizer.zero_grad()
        # 역전파를 통해 그레디언트를 계산
        loss.backward()
        # 옵티마이저를 통해 모델 파라미터를 업데이트
        optimizer.step()

    # 한 에포크가 끝난후 , 모델을 평가하여 성능을 측정
    evaluate(model, test_loader, index_to_tag)

# - f1: 84.60
#              precision    recall  f1-score   support

#         AFW       0.67      0.59      0.62       394
#         ANM       0.81      0.64      0.71       701
#         CVL       0.83      0.82      0.82      5758
#         DAT       0.88      0.91      0.89      2521
#         EVT       0.74      0.73      0.73      1094
#         FLD       0.78      0.47      0.59       228
#         LOC       0.87      0.82      0.85      2126
#         MAT       0.14      0.08      0.11        12
#         NUM       0.89      0.93      0.91      5590
#         ORG       0.84      0.88      0.86      4086
#         PER       0.85      0.91      0.88      4426
#         PLT       0.29      0.06      0.10        34
#         TIM       0.81      0.90      0.85       314
#         TRM       0.77      0.70      0.74      1964

#   micro avg       0.85      0.85      0.85     29248
#   macro avg       0.73      0.67      0.69     29248
#weighted avg       0.84      0.85      0.84     29248

# - f1: 85.88
#              precision    recall  f1-score   support

#         AFW       0.65      0.60      0.62       394
#         ANM       0.79      0.71      0.75       701
#         CVL       0.83      0.85      0.84      5758
#         DAT       0.91      0.93      0.92      2521
#         EVT       0.77      0.74      0.75      1094
#         FLD       0.81      0.49      0.61       228
#         LOC       0.88      0.84      0.86      2126
#         MAT       0.19      0.33      0.24        12
#         NUM       0.91      0.93      0.92      5590
#         ORG       0.87      0.88      0.87      4086
#         PER       0.87      0.91      0.89      4426
#         PLT       0.30      0.09      0.14        34
#         TIM       0.82      0.93      0.87       314
#         TRM       0.77      0.72      0.74      1964

#   micro avg       0.86      0.86      0.86     29248
#   macro avg       0.74      0.71      0.72     29248
#weighted avg       0.85      0.86      0.86     29248

# - f1: 86.13
#              precision    recall  f1-score   support

#         AFW       0.62      0.61      0.62       394
#         ANM       0.78      0.76      0.77       701
#         CVL       0.83      0.86      0.84      5758
#         DAT       0.91      0.93      0.92      2521
#         EVT       0.77      0.75      0.76      1094
#         FLD       0.66      0.62      0.64       228
#         LOC       0.86      0.85      0.86      2126
#         MAT       0.17      0.33      0.23        12
#         NUM       0.91      0.92      0.92      5590
#         ORG       0.86      0.89      0.87      4086
#         PER       0.89      0.89      0.89      4426
#         PLT       0.53      0.24      0.33        34
#         TIM       0.83      0.92      0.87       314
#         TRM       0.80      0.72      0.76      1964

#   micro avg       0.86      0.86      0.86     29248
#   macro avg       0.74      0.74      0.73     29248
#weighted avg       0.86      0.86      0.86     29248

print("\n=============================================")

######################################################################
# 모델 예측 (학습하지 않은 임의의 문장에 대해 개체명 태깅 정보를 예측)

## 임의의 문장에 대한 예측을 위해 입력 문장 전처리를 진행하는 함수 정의
