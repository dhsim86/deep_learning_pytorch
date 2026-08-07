# BertForMaskedLM vs BertForNextSentencePrediction

> 같은 `klue/bert-base`를 로드하는데 왜 결과가 다를까?

## 목차

- [핵심 한 줄 요약](#핵심-한-줄-요약)
- [1. BERT의 구조: 몸통 + 머리](#1-bert의-구조-몸통--머리)
- [2. 직접 검증한 결과](#2-직접-검증한-결과)
- [3. 체크포인트 내부 뜯어보기](#3-체크포인트-내부-뜯어보기)
- [4. 초보자가 꼭 알아야 할 함정](#4-초보자가-꼭-알아야-할-함정)
- [5. 형제 클래스 총정리](#5-형제-클래스-총정리)
- [6. 마지막 비유](#6-마지막-비유)
- [참고 자료](#참고-자료)

---

## 핵심 한 줄 요약

**`klue/bert-base`는 "모델"이 아니라 "가중치 보따리"입니다.**
`BertForXXX` 클래스는 그 보따리에서 필요한 것만 꺼내 쓰고, 그 위에 **어떤 출력 층(head)을 붙일지** 결정하는 껍데기입니다. 몸통은 똑같고 머리만 갈아 끼우는 겁니다.

---

## 1. BERT의 구조: 몸통 + 머리

```
┌─────────────────────────────────────┐
│  입력: "축구는 정말 재미있는 [MASK]다."   │
└─────────────────┬───────────────────┘
                  ↓
┌─────────────────────────────────────┐
│   BertModel (몸통 / body)             │  ← 여기가 klue/bert-base의 본체
│   임베딩 + 12층 Transformer 인코더      │     약 1억 1천만 파라미터
└─────────────────┬───────────────────┘
                  ↓
        (batch, 토큰수, 768) 짜리 벡터
                  ↓
        ┌─────────┴─────────┐
        ↓                   ↓
┌───────────────┐   ┌─────────────────┐
│  MLM 머리      │   │   NSP 머리        │  ← 여기만 다름
│  768 → 32000  │   │   768 → 2        │
└───────────────┘   └─────────────────┘
   "이 자리에 올            "두 문장이
    단어는?"                이어지나?"
```

몸통이 하는 일은 **"문장을 이해해서 768차원 벡터로 바꾸는 것"** 하나뿐입니다.
그 벡터를 가지고 무엇을 할지는 머리가 정합니다.

---

## 2. 직접 검증한 결과

### 실험 A — 몸통 출력이 정말 같은가?

같은 문장을 두 모델에 넣고 **머리를 거치기 직전** 값을 비교했습니다.

```python
import torch
from transformers import BertForMaskedLM, BertForNextSentencePrediction, AutoTokenizer

tok = AutoTokenizer.from_pretrained('klue/bert-base')
mlm = BertForMaskedLM.from_pretrained('klue/bert-base').eval()
nsp = BertForNextSentencePrediction.from_pretrained('klue/bert-base').eval()

enc = tok('축구는 정말 재미있는 [MASK]다.', return_tensors='pt')

with torch.no_grad():
    h_mlm = mlm.bert(**enc).last_hidden_state   # 몸통만 통과
    h_nsp = nsp.bert(**enc).last_hidden_state

print(torch.equal(h_mlm, h_nsp))                # True
print((h_mlm - h_nsp).abs().max().item())       # 0.0
```

```
본체(encoder) 출력 shape : (1, 10, 768) (1, 10, 768)
본체 출력 완전 일치 여부  : True
최대 오차                 : 0.0
```

**오차 0.0, 비트 단위로 완전히 동일합니다.** 몸통은 같은 모델이라는 게 증명됐습니다.

### 실험 B — 머리를 통과한 뒤 출력은?

```
MLM 최종 logits shape : (1, 10, 32000)  → (배치, 토큰 개수, 어휘 사전 크기)
NSP 최종 logits shape : (1, 2)          → (배치, 2)

MLM 예측 top5: ['스포츠', '거', '경기', '축구', '놀이']
NSP 예측 확률: [[0.554, 0.446]]
```

|  | `BertForMaskedLM` | `BertForNextSentencePrediction` |
|---|---|---|
| 출력 크기 | `(배치, 토큰수, 32000)` | `(배치, 2)` |
| 의미 | 토큰마다 "32000개 단어 중 뭐일까" 점수 | "이어짐 / 안 이어짐" 두 가지 점수 |
| 머리 파라미터 수 | **624,128개** | **1,538개** (= 768×2 + 2) |
| 정답 라벨 | 가려진 단어의 ID | 0(이어짐) 또는 1(안 이어짐) |

NSP 머리는 그냥 **768→2 짜리 선형 층 하나**입니다. 놀랄 만큼 작죠.
반면 MLM 머리는 3만 2천 개 단어 전체에 점수를 매겨야 해서 훨씬 큽니다.

---

## 3. 체크포인트 내부 뜯어보기

`klue/bert-base`의 `model.safetensors` 안에 든 텐서 이름 중, 인코더 층을 뺀 나머지입니다.

```
bert.pooler.dense.weight                    [768, 768]   ← NSP가 쓰는 부품
bert.pooler.dense.bias                      [768]

cls.predictions.transform.dense.weight      [768, 768]   ← MLM 머리
cls.predictions.transform.LayerNorm.weight  [768]
cls.predictions.bias                        [32000]

cls.seq_relationship.weight                 [2, 768]     ← NSP 머리
cls.seq_relationship.bias                   [2]
```

> **중요:** `klue/bert-base`에는 MLM 머리와 NSP 머리가 **둘 다** 들어있습니다.
> KLUE-BERT가 원조 BERT처럼 MLM + NSP 두 과제로 동시에 사전학습됐기 때문입니다.

### 클래스별 로드 로그

**`BertForMaskedLM`으로 로드할 때**

```
cls.seq_relationship.weight | UNEXPECTED    ← NSP 머리는 안 쓰니 버림
cls.seq_relationship.bias   | UNEXPECTED
bert.pooler.dense.weight    | UNEXPECTED    ← pooler도 안 쓰니 버림
bert.pooler.dense.bias      | UNEXPECTED
```

**`BertForNextSentencePrediction`으로 로드할 때**

```
cls.predictions.bias                    | UNEXPECTED    ← MLM 머리는 안 쓰니 버림
cls.predictions.transform.dense.weight  | UNEXPECTED
cls.predictions.transform.LayerNorm.*   | UNEXPECTED
```

`UNEXPECTED`는 **"체크포인트엔 있는데 이 클래스는 쓸 데가 없어서 버렸다"**는 뜻입니다.
에러가 아니라 정상입니다.

> 참고로 `BertForMaskedLM`은 `pooler`를 **아예 만들지도 않습니다**.
> MLM은 `[CLS]` 요약 벡터가 필요 없거든요.
> 그래서 두 모델의 `bert.*` 파라미터 수가 정확히 pooler 크기(590,592개)만큼 차이 납니다.

---

## 4. 초보자가 꼭 알아야 할 함정

### 함정 1 — 다른 모델에선 NSP가 안 될 수 있다

`klue/bert-base`는 운 좋게 NSP 머리가 들어있지만,
**`bert-base-uncased` 같은 모델은 MLM 머리만 배포**되어 있습니다.
이런 체크포인트를 `BertForNextSentencePrediction`으로 로드하면:

```
Some weights of BertForNextSentencePrediction were not initialized ...
and are newly initialized: ['cls.seq_relationship.weight', 'cls.seq_relationship.bias']
You should probably TRAIN this model on a down-stream task ...
```

`newly initialized`(랜덤 초기화)라는 경고가 뜨고,
**그 상태로 예측하면 결과가 완전히 무의미한 난수**입니다.

| 경고 종류 | 의미 | 위험도 |
|---|---|---|
| `UNEXPECTED` / `not used` | 체크포인트에 남는 가중치가 있음 | 무시해도 됨 ✅ |
| `newly initialized` / `MISSING` | 머리가 **랜덤값**임 | 학습 없이 쓰면 안 됨 ❌ |

### 함정 2 — 파인튜닝용 클래스는 항상 랜덤 머리다

`BertForSequenceClassification`이 대표적입니다.
"뉴스 민감도 분류" 같은 과제는 사전학습에 없었으니,
**머리는 무조건 랜덤 초기화 → 반드시 직접 학습**해야 합니다.
이때 뜨는 `newly initialized` 경고는 "정상"입니다.

---

## 5. 형제 클래스 총정리

| 클래스 | 붙는 머리 | `klue/bert-base` 로드 시 머리 상태 |
|---|---|---|
| `BertModel` | 없음 (768 벡터만 반환) | — |
| `BertForMaskedLM` | MLM (768→32000) | ✅ 사전학습됨, 바로 사용 가능 |
| `BertForNextSentencePrediction` | NSP (768→2) | ✅ 사전학습됨, 바로 사용 가능 |
| `BertForPreTraining` | MLM + NSP **둘 다** | ✅ 둘 다 로드됨 |
| `BertForSequenceClassification` | 분류 (768→N) | ❌ 랜덤 → 파인튜닝 필수 |
| `BertForQuestionAnswering` | 시작/끝 위치 (768→2) | ❌ 랜덤 → 파인튜닝 필수 |

**MLM과 NSP를 한 번에 쓰고 싶다면** `BertForPreTraining`을 쓰면 됩니다.
두 머리를 다 붙여주므로 `UNEXPECTED` 경고 없이 깔끔하게 로드됩니다.

---

## 6. 마지막 비유

> `klue/bert-base` = **한국어를 통달한 사람의 뇌**
> `BertForXXX` = 그 사람에게 쥐여주는 **도구**
>
> - `BertForMaskedLM` → 빈칸 채우기 문제지를 줌
> - `BertForNextSentencePrediction` → OX 문제지를 줌
> - `BertForSequenceClassification` → 처음 보는 시험지를 줌 (연습 필요)
>
> 사람(뇌)은 똑같습니다. 손에 쥔 도구만 다를 뿐이고,
> 그래서 답의 **형태**가 달라지는 겁니다.

---

## 관련 예제 코드

- [`korean_masked_language_modeling.py`](./korean_masked_language_modeling.py) — MLM 예제
- [`korean_next_sentence_prediction.py`](./korean_next_sentence_prediction.py) — NSP 예제
- [`korean_news_sensitivity_classification.py`](./korean_news_sensitivity_classification.py) — 분류 파인튜닝 (함정 2 해당)

## 참고 자료

- [BERT — Hugging Face Transformers 공식 문서](https://huggingface.co/docs/transformers/model_doc/bert)
- [modeling_bert.py 소스코드 (BertOnlyMLMHead / BertOnlyNSPHead 정의)](https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/modeling_bert.py)
- [Why aren't all weights of BertForPreTraining initialized from the model checkpoint? — HF 포럼](https://discuss.huggingface.co/t/why-arent-all-weights-of-bertforpretraining-initialized-from-the-model-checkpoint/10509)
- [KLUE: Korean Language Understanding Evaluation (arXiv:2105.09680)](https://arxiv.org/pdf/2105.09680)
- [KLUE-benchmark/KLUE GitHub](https://github.com/KLUE-benchmark/KLUE)
