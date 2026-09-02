# GPT-2를 이용한 챗봇 파인튜닝
import pandas as pd
from tqdm import tqdm

import torch
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer, PreTrainedTokenizerFast, GPT2LMHeadModel

######################################################################
# 모델과 토크나이저 로드 및 특수 토큰 확인
## 한국어 GPT 중 하나인 'skt/kogpt2-base-v2'를 사용.

## GPT2LMHeadModel = GPT2Model(Transformer 본체) + LMHead(맨 마지막 Linear 층)
model = GPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2')

## 토크나이저 로드시 텍스트 생성 시작 토큰(bos_token), 텍스트 생성 종료 토큰(eos_token) 등 지정 가능
tokenizer = PreTrainedTokenizerFast.from_pretrained(
    'skt/kogpt2-base-v2',
    bos_token='</s>', eos_token='</s>', unk_token='<unk>',
    pad_token='<pad>', mask_token='<mask>')


## 특수 토큰 확인

## <usr> 및 <sys> 특수 토큰은 GPT로 대화 모델을 만들 경우 유저의 발화(<usr>) 및 시스템의 응답(<sys>)를 구분하기 위한 용도
print("bos token ID: ", tokenizer.bos_token_id)
print("eos token ID: ", tokenizer.eos_token_id)
print("pad token ID: ", tokenizer.pad_token_id)
print('-' * 10)
print("1번 토큰: ", tokenizer.decode(1))
print("2번 토큰: ", tokenizer.decode(2)) # <usr>, 유저 발화를 위한 토큰
print("3번 토큰: ", tokenizer.decode(3))
print("4번 토큰: ", tokenizer.decode(4)) # <sys>, 시스템 응답을 위한 토큰

print("\n=============================================")

######################################################################
# 데이터셋 다운로드
# import urllib.request

# urllib.request.urlretrieve("https://raw.githubusercontent.com/songys/Chatbot_data/master/ChatbotData.csv", filename="ChatBotData.csv")
train_data = pd.read_csv('ChatBotData.csv')

## 질문과 쌍으로 이루어진 데이터셋 확인

#                 Q            A  label
# 0           12시 땡!   하루가 또 가네요.      0
# 1      1지망 학교 떨어졌어    위로해 드립니다.      0
# 2     3박4일 놀러가고 싶다  여행은 언제나 좋죠.      0
# 3  3박4일 정도 놀러가고 싶다  여행은 언제나 좋죠.      0
# 4          PPL 심하네   눈살이 찌푸려지죠.      0
print(train_data.head())
print('챗봇 샘플의 개수 :', len(train_data)) # 11823

print("\n=============================================")

######################################################################
# 데이터셋 전처리 및 데이터로더 준비

## 대화 데이터를 위한 사용자 정의 데이터셋 클래스 정의
## -> 사용자 정의 데이터셋을 정의하여 pytorch의 데이터로더에서 사용할 수 있도록 한다.
class ChatDataset(Dataset):
    def __init__(self, train_data, tokenizer):
        self.train_data = train_data  # 학습 데이터를 저장
        self.tokenizer = tokenizer  # 텍스트를 토큰으로 변환할 토크나이저 저장

    def __len__(self):
        return len(self.train_data)  # 데이터셋의 크기(샘플 수)를 반환

    # 특정 인덱스에 해당하는 데이터 샘플을 반환하는 역할
    def __getitem__(self, idx):
        question = self.train_data.Q.iloc[idx]  # 인덱스에 해당하는 질문 텍스트 가져오기
        answer = self.train_data.A.iloc[idx]  # 인덱스에 해당하는 답변 텍스트 가져오기

        bos_token = self.tokenizer.bos_token_id  # 문장의 시작을 나타내는 토큰 ID
        eos_token = self.tokenizer.eos_token_id  # 문장의 끝을 나타내는 토큰 ID

        # 질문과 답변을 하나의 문자열로 연결하여 토큰화, 정수 인코딩
        sent = self.tokenizer.encode('<usr>' + question + '<sys>' + answer, add_special_tokens=False)

        # 시작과 끝 토큰을 포함한 텐서를 반환
        return torch.tensor([bos_token] + sent + [eos_token], dtype=torch.long)

# 배치 내 각 시퀀스를 패딩하여 같은 길이로 맞추는 함수 정의
def collate_fn(batch):
    return torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=tokenizer.pad_token_id)

batch_size = 32  # 배치 크기 지정
chat_dataset = ChatDataset(train_data, tokenizer)  # 데이터셋 인스턴스 생성
data_loader = DataLoader(chat_dataset, batch_size=batch_size, collate_fn=collate_fn)  # 데이터로더 생성

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

## 모델을 선택한 디바이스로 이동
## -> 입력 텐서만 디바이스로 옮기고 모델이 CPU에 남아 있으면 device mismatch 에러가 발생한다.
model.to(device)

## 학습 파라미터 및 옵티마이저 설정
epochs = 3
steps = len(train_data) // batch_size + 1

# eps: 수치적 안정성을 위해 사용되는 작은 값, 나눗셈 연산에서 분모가 0이 되는 것을 방지하는 역할
optimizer = AdamW(model.parameters(), lr=3e-5, eps=1e-08) 

######################################################################
# 모델 학습 진행

## 주어진 에포크 수만큼 학습 루프를 반복
for epoch in range(epochs):
    epoch_loss = 0  # 에포크 손실 초기화

    # 데이터 로더에서 배치를 하나씩 가져와서 학습을 진행
    for step, batch in tqdm(enumerate(data_loader), total=steps):
        # 배치를 선택한 디바이스로 이동
        batch = batch.to(device)
        # 레이블을 배치와 동일하게 설정 (입력을 그대로 레이블로 사용)
        labels = batch.clone()

        # 모델에 입력을 주고, 출력과 손실값을 계산
        result = model(input_ids=batch, labels=labels)

        loss = result.loss  # 계산된 손실값
        batch_loss = loss.mean()  # 배치 손실 계산

        optimizer.zero_grad() # 옵티마이저의 기울기 초기화
        batch_loss.backward() # 손실값에 대해 역전파를 통해 기울기 계산
        optimizer.step() # 옵티마이저를 통해 가중치 업데이트

        # 에포크 손실에 이번 배치의 손실값을 추가
        epoch_loss += batch_loss.item() / steps

    # 현재 에포크가 끝난 후 평균 손실값 출력
    print('[Epoch: {:>4}] cost = {:>.9}'.format(epoch + 1, epoch_loss))

print("\n=============================================")

######################################################################
# 챗봇 동작 테스트

## '<usr>'는 사용자 입력을 '<sys>'는 시스템 응답을 나타내는 태그로 감싸서 대화 형태로 변환
text = '오늘도 좋은 하루!'
sent = '<usr>' + text + '<sys>'

# 문장의 시작을 알리는 bos_token_id와 토큰화 된 문장을 이어 붙이고 정수 인코딩.
input_ids = [tokenizer.bos_token_id] + tokenizer.encode(sent)
input_ids = torch.tensor([input_ids], dtype=torch.long).to(device)

# 모델을 사용해 주어진 입력에 대한 응답을 생성 (최대 50개의 토큰, 조기 종료 조건 설정)
## "</s><usr>사용자의 질문<sys>챗봇의 답변</s>" 형태를 학습하였으므로, 모델은 <sys> 뒤에 답변을 생성한다.
output = model.generate(input_ids, max_length=50, early_stopping=True, eos_token_id=tokenizer.eos_token_id)
decoded_sentence = tokenizer.decode(output[0].tolist())

# </s><usr> 오늘도 좋은 하루!<sys> 좋은 하루네요.</s>
print(decoded_sentence)
# 좋은 하루네요.
print(decoded_sentence.split('<sys> ')[1].replace('</s>', ''))

## 임의의 입력에 대한 챗봇 답변 출력
def return_answer_by_chatbot(user_text):
    sent = '<usr>' + user_text + '<sys>'

    input_ids = [tokenizer.bos_token_id] + tokenizer.encode(sent, add_special_tokens=False)
    input_ids = torch.tensor([input_ids], dtype=torch.long).to(device)

    # 답변을 랜덤하게 생성하도록 do_sample=True, top_k 지정
    output = model.generate(input_ids, max_length=50, do_sample=True, top_k=2)
    
    sentence = tokenizer.decode(output[0].tolist())
    chatbot_response = sentence.split('<sys> ')[1].replace('</s>', '')

    return chatbot_response

# 반가워요!!
print(return_answer_by_chatbot('안녕! 반가워~'))
# 저는 짝녀입니다.
print(return_answer_by_chatbot('너는 누구야?'))
# 같이 놀자요.
print(return_answer_by_chatbot('너무 심심한데 나랑 놀자'))
# 영화 추천해달라고 해보세요.
print(return_answer_by_chatbot('영화 해리포터 재밌어?'))
# 직접 해보세요.
print(return_answer_by_chatbot('너 딥 러닝 잘해?'))

# 먼저 다가가보세요.
print(return_answer_by_chatbot('개발을 잘하려면 어떻게 해야돼?'))
# 그게 제일 힘들거 같아요.
print(return_answer_by_chatbot('스트레스를 해소하려면 어떻게 해야될까?'))
