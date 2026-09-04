# 다양한 데이터셋을 쉽게 로드하고 처리
## load_dataset 은 허깅페이스에 업로드 된 데이터셋을 불러오거나 로컬 파일을 로드하는 데 사용
## Dataset 은 개별 데이터셋 객체를 다룰 때 사용
from datasets import load_dataset, Dataset

import torch

from transformers import AutoModelForCausalLM, AutoTokenizer

# LoRA(Low‑Rank Adaptation) 튜닝을 사용할 때 필요한 설정값을 정의
from peft import LoraConfig

# 모델을 학습할 때 필요한 다양한 설정값을 정의하는 도구
## 학습률, 배치크기, 옵티마이저 등의 설정
from trl import SFTConfig

# 실제 학습을 수행하는 클래스
## 모델, 데이터셋, 학습 설정을 한 번에 입력하여 효율적인 학습을 진행
from trl import SFTTrainer

print("\n=============================================")

######################################################################
# 데이터 전처리

## 1. 허깅페이스 허브에서 데이터셋 로드 (금융 뉴스 데이터셋)
dataset = load_dataset("iamjoon/finance_news_summarizer", split="train")

## 2. 전체 데이터 크기만 출력
print("전체 데이터 크기:", len(dataset)) # 991

## 훈련/테스트 데이터셋으로 분할
test_ratio = 0.2
train_data = []
test_data = []

data_indices = list(range(len(dataset)))
test_size = int(len(data_indices) * test_ratio)
test_data = data_indices[:test_size]
train_data = data_indices[test_size:]

