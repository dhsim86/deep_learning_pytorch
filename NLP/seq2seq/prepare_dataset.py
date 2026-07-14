import re
import os
import unicodedata
import urllib3
import zipfile
import shutil
import numpy as np
import pandas as pd
import torch
from collections import Counter
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset

# 19만개의 데이터 중 33,000개의 샘플만 사용
num_samples = 33000

############################################################
# 전처리 함수 구현

## 프랑스어 정규화
def unicode_to_ascii(s):
  # 프랑스어 악센트(accent) 삭제
  # 예시 : 'déjà diné' -> deja dine
  return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

## 문장 정규화
def preprocess_sentence(sent):
  # 악센트 삭제 함수 호출
  sent = unicode_to_ascii(sent.lower())

  # 단어와 구두점 사이에 공백을 만듭니다.
  # Ex) "he is a boy." => "he is a boy ."
  sent = re.sub(r"([?.!,¿])", r" \1", sent)

  # (a-z, A-Z, ".", "?", "!", ",") 이들을 제외하고는 전부 공백으로 변환합니다.
  sent = re.sub(r"[^a-zA-Z!.?]+", r" ", sent)

  # 다수 개의 공백을 하나의 공백으로 치환
  sent = re.sub(r"\s+", " ", sent)
  return sent

## 영어 및 프랑스어 문장을 전처리 후, 인코더 / 디코더 입력용 데이터 준비
def load_preprocessed_data():
  encoder_input, decoder_input, decoder_target = [], [], []

  with open("fra.txt", "r") as lines:
    for i, line in enumerate(lines):
      
      # 영어 및 프랑스어 문장 로드 (source 데이터와 target 데이터 분리)
      src_line, tar_line, _ = line.strip().split('\t')

      # 영어, source 데이터 전처리
      src_line = [w for w in preprocess_sentence(src_line).split()]

      # 프랑스어, target 데이터 전처리
      tar_line = preprocess_sentence(tar_line)

      # tearcher forcing을 위해 훈련시 사용할 디코더의 입력 시퀀스와
      # 실제값, 레이블에 해당하는 출력 시퀀스를 분리하여 저장
      tar_line_in = [w for w in ("<sos> " + tar_line).split()] # 디코더 입력 시퀀스, <sos>를 앞에 더해준다.
      tar_line_out = [w for w in (tar_line + " <eos>").split()] # 디코더 출력 시퀀스, <eos>를 뒤에 더해준다.


      encoder_input.append(src_line)
      decoder_input.append(tar_line_in)
      decoder_target.append(tar_line_out)

      if i == num_samples - 1:
        break

  return encoder_input, decoder_input, decoder_target

## 전처리 테스트
en_sent = u"Have you had dinner?"
fr_sent = u"Avez-vous déjà diné?"

## 전처리 전 영어 문장 : Have you had dinner?
## 전처리 후 영어 문장 : have you had dinner ?
## 전처리 전 프랑스어 문장 : Avez-vous déjà diné?
## 전처리 후 프랑스어 문장 : avez vous deja dine ?
print('전처리 전 영어 문장 :', en_sent)
print('전처리 후 영어 문장 :',preprocess_sentence(en_sent))
print('전처리 전 프랑스어 문장 :', fr_sent)
print('전처리 후 프랑스어 문장 :', preprocess_sentence(fr_sent))

# 코퍼스 로드 후, 데이터셋 인코더의 입력, 디코더의 입력 및 레이블 상위 5개 샘플 출력
## 인코더의 입력 : [['go', '.'], ['go', '.'], ['go', '.'], ['go', '.'], ['hi', '.']]
## 디코더의 입력 : [['<sos>', 'va', '!'], ['<sos>', 'marche', '.'], ['<sos>', 'en', 'route', '!'], ['<sos>', 'bouge', '!'], ['<sos>', 'salut', '!']]
## 디코더의 레이블 : [['va', '!', '<eos>'], ['marche', '.', '<eos>'], ['en', 'route', '!', '<eos>'], ['bouge', '!', '<eos>'], ['salut', '!', '<eos>']]
sents_en_in, sents_fra_in, sents_fra_out = load_preprocessed_data()
print('인코더의 입력 :',sents_en_in[:5])
print('디코더의 입력 :',sents_fra_in[:5])
print('디코더의 레이블 :',sents_fra_out[:5])