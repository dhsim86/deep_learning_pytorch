import torch
import torch.nn.functional as F

torch.manual_seed(1)

##################
# 소프트맥스 테스트
z = torch.FloatTensor([1, 2, 3])

result = F.softmax(z, dim=0)
print("softmax 함수 결과:", result.tolist())
print("softmax 함수 결과의 합:", result.sum().item())  # 합은 1이어야 함

##################
# 비용함수 테스트

## 데이터 준비 (레이블이 5개인 경우, 가중치 통과, 소프트맥스 입력 직전 단계)
z = torch.rand(3, 5, requires_grad=True)  # (3, 5) 크기의 랜덤 텐서 생성

### 각 샘플에 대해 소프트맥스 함수 적용
result = F.softmax(z, dim=1) # dim=1은 두 번째 차원을 제거 후 소프트맥스 적용, 각 행에 대해 소프트맥스 적용
print("softmax 적용 결과:\n", result)

## 정답 준비
### 임의의 레이블 생성
y = torch.randint(5, (3,)).long() # 0~4 사이의 정수 레이블 3개 생성
print("임의의 레이블:", y.tolist())

### 각 레이블에 대해 원핫인코딩 생성
y_one_hot = torch.zeros_like(result)  # result와 동일한 크기(3, 5)의 0 텐서 생성
y_one_hot.scatter_(1, y.unsqueeze(1), 1)  # 각 레이블 위치에 1 삽입
# 두 번째 인자로 y.unsqueeze(1)을 사용하는 이유는 y의 크기가 (3,)이므로 (3,1)로 변환, 그 위치에 1을 넣기 위함

print("원핫인코딩 결과:\n", y_one_hot)

## 비용함수
cost = -(y_one_hot * torch.log(result)).sum(dim=1).mean()  # 크로스 엔트로피 비용함수 계산
print("크로스 엔트로피 비용함수 결과:", cost.item())

print("=============================================")

# torch 함수 사용
## F.softmax() + torch.log() = F.log_softmax()
z = torch.rand(3, 5, requires_grad=True)

softmax_result = F.softmax(z, dim=1)
print("softmax 함수 결과:\n", softmax_result)
print("softmax 함수 결과의 로그:\n", torch.log(softmax_result))

log_softmax_result = F.log_softmax(z, dim=1)
print("log_softmax 함수 결과:\n", log_softmax_result)

## F.log_softmax() + F.nll_loss() = F.cross_entropy()
cost = -(y_one_hot * torch.log(result)).sum(dim=1).mean()  # 크로스 엔트로피 비용함수 계산
print("크로스 엔트로피 비용함수 결과:", cost.item())

### nill_loss() 함수 사용
#### nill_loss는 Negative Log Likelihood Loss의 약자
#### nill_loss를 활용하면 원핫벡터를 쓸 필요없이 바로 실제값을 사용 가능
nll_loss_cost = F.nll_loss(F.log_softmax(z, dim=1), y)
print("nll_loss 함수 사용 비용함수 결과:", nll_loss_cost.item())

### cross_entropy() 함수 사용
#### cross_entropy 함수는 소프트맥스 함수까지 포함하고 있는 것에 주의
cross_entropy_cost = F.cross_entropy(z, y)
print("cross_entropy 함수 사용 비용함수 결과:", cross_entropy_cost.item())