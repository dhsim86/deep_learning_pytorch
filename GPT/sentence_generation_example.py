import torch

from transformers import AutoTokenizer, PreTrainedTokenizerFast # AutoTokenizer 는 쓰지 않는다 (이유는 아래 주석)
from transformers import GPT2LMHeadModel # 허깅페이스에서 제공하는 GPT-2 기반의 언어 모델 클래스

## GPT2LMHeadModel = GPT2Model(Transformer 본체) + LMHead(맨 마지막 Linear 층)
## - GPT2Model 만 쓰면 각 토큰의 은닉 표현(hidden state, 768차원)까지만 얻는다.
## - LMHead가 그 768차원을 단어 집합 크기(51200)로 펼쳐서 "다음 토큰 점수"를 만들어 준다.
##   따라서 클래스 이름의 LMHead(Language Model Head)가 곧 "다음 토큰 예측기"를 의미한다.
##
## skt/kogpt2-base-v2 의 주요 설정값 (모델의 config.json)
##   vocab_size  = 51200   단어 집합 크기   -> 로짓 벡터의 길이가 된다
##   n_embd      = 768     은닉 표현 차원
##   n_layer     = 12      Transformer 블록 수
##   n_positions = 1024    한 번에 넣을 수 있는 최대 토큰 수
model = GPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2')

## 토크나이저는 반드시 PreTrainedTokenizerFast 를 써야 한다. AutoTokenizer 를 쓰면 오동작한다.
##
## ---------------------------------------------------------------------------
## [원인 1] skt/kogpt2-base-v2 리포에 tokenizer_config.json 이 없다
##
## AutoTokenizer 는 tokenizer_config.json 의 tokenizer_class 값을 보고 클래스를 정한다.
## 그런데 이 리포에는 config.json / tokenizer.json / 가중치 파일만 있고
## tokenizer_config.json, special_tokens_map.json, vocab.json, merges.txt 가 모두 없다.
## 그래서 클래스 힌트를 찾지 못하고 config.json 의 model_type 으로 폴백한다.
##
##   config.json 의 model_type: "gpt2"  ->  GPT2Tokenizer 선택
##
## KoGPT2 는 "신경망 구조만" GPT-2 이고 토크나이저는 SentencePiece 계열인데도
## "GPT-2 구조니까 OpenAI GPT-2 의 토크나이저겠지" 라고 잘못 추정해버리는 것이다.
##
## ---------------------------------------------------------------------------
## [원인 2] GPT2Tokenizer 가 tokenizer.json 의 파이프라인을 통째로 덮어쓴다
##
## GPT2Tokenizer.__init__ 은 tokenizer.json 을 그대로 복원하지 않는다.
## Tokenizer 객체를 새로 만들면서 ByteLevel 을 하드코딩한다.
## (transformers/models/gpt2/tokenization_gpt2.py)
##
##   self._tokenizer = Tokenizer(BPE(vocab=..., merges=..., fuse_unk=False))
##   self._tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(...)
##   self._tokenizer.decoder = decoders.ByteLevel()
##
## normalizer 는 설정하는 코드조차 없어서 None 이 된다.
## 결과적으로 tokenizer.json 에서 vocab/merges 만 살아남고 텍스트 처리 파이프라인은 사라진다.
## 즉 "단어장은 같은데 그 단어장을 찾아가는 길이 없어진" 상태다.
##
##   구성요소      | tokenizer.json 원본 (정상 동작) | AutoTokenizer (GPT2Tokenizer)
##   --------------+---------------------------------+------------------------------
##   normalizer    | NFKC -> BertNormalizer          | None
##   pre_tokenizer | Metaspace(replacement="▁")      | ByteLevel
##   decoder       | Metaspace(replacement="▁")      | ByteLevel
##   vocab/merges  | 51200 / 42185                   | 51200 / 42185 (동일)
##
## vocab 자체는 동일하다. '▁근육이' 의 id 는 양쪽 모두 33245 다.
## 문제는 원문에서 '▁근육이' 라는 토큰을 만들어내는 경로가 사라진 것이다.
##
## ---------------------------------------------------------------------------
## [증상] 한글이 UTF-8 바이트로 찢기고, 남은 조각은 조용히 삭제된다
##
## ByteLevel 은 텍스트를 UTF-8 바이트로 바꾼 뒤 각 바이트를 인쇄 가능한 문자로 매핑한다.
##
##   원문              : '근육이 커지기 위해서는'            (UTF-8 32바이트)
##   ByteLevel 통과 후 : 'ê·¼ìľ¡ìĿ´Ġì»¤ì§Ģê¸°ĠìľĦíķ´ìĦľëĬĶ'   (32자)
##
## 이 32자 중 KoGPT2 vocab 에 우연히 존재하는 것은 12자뿐이다.
## 그런데 GPT2Tokenizer 가 만든 BPE 는 unk_token 이 None 이다.
## 그래서 나머지 20자는 <unk> 로도 바뀌지 않고 에러도 없이 그냥 버려진다.
## 경고가 한 줄도 나오지 않기 때문에 발견하기 어렵다.
##
##   PreTrainedTokenizerFast -> ['▁근육이', '▁커', '지기', '▁위해서는']
##                              [33245, 10114, 12748, 11357]
##   AutoTokenizer           -> ['ê','·','ì','ì','ì','ì','ê','°','ì','í','ì','ë']
##                              [499, 473, 501, 501, 501, 501, 499, 470, 501, 502, 501, 500]
##
## 입력 id 가 무의미하므로 모델 출력도 무너진다.
##
##   top5     : [',', 'n', '’', '<unk>', ')']
##   generate : ����������,▁��nǐ�i� 이런▁식으로▁하면▁중국식▁발음인▁'훙'...
##
## 생성 결과에 ▁ 가 그대로 노출되는 것도 같은 원인이다.
## 디코더가 Metaspace 가 아니라 ByteLevel 이라 ▁ 를 공백으로 되돌리지 못한다.
##
## ---------------------------------------------------------------------------
## [부수 문제] 단어 집합 크기가 모델 임베딩보다 커진다
##
## GPT2Tokenizer 의 기본 특수 토큰 <|endoftext|> 는 KoGPT2 vocab 에 없으므로
## id 51200 번으로 새로 추가된다. 하지만 모델 임베딩은 51200행(id 0~51199)뿐이다.
##
##   len(AutoTokenizer)     = 51201   <- 모델 임베딩 51200 을 초과
##   tokenizer.eos_token_id = 51200   <- 모델이 처리할 수 없는 id
##   model(torch.tensor([[51200]]))  ->  IndexError: index out of range in self
##
## 이 예제는 generate() 가 model.config 의 eos_token_id=1 을 쓰므로 크래시까지는 가지 않는다.
## 그러나 tokenizer.eos_token_id 를 직접 넘기는 코드에서는 바로 터진다.
##
## ---------------------------------------------------------------------------
## [교훈] AutoTokenizer 는 tokenizer_config.json 이 있는 리포에서만 신뢰할 수 있다.
## 그 파일이 없고 tokenizer.json 만 있는 리포에서는 model_type 으로 잘못 추정한다.
## SKT 공식 저장소(github.com/SKT-AI/KoGPT2)도 AutoTokenizer 대신
## PreTrainedTokenizerFast 를 쓰라고 명시하고 있다.
## ---------------------------------------------------------------------------

## 아래 특수 토큰 인자는 SKT 공식 저장소의 사용법이다.
## - 이 예제(문장 1건)는 인자를 생략해도 동작한다. encode 결과가 완전히 동일하다.
## - 하지만 생략하면 bos/eos/unk/pad 가 모두 None 이 되어,
##   padding=True 로 배치 처리하는 순간 아래 오류가 난다.
##     ValueError: Asking to pad but the tokenizer does not have a padding token.
## - 인자를 넘겨도 len(tokenizer) 는 51200 그대로다.
##   다섯 토큰이 모두 이미 vocab 에 있어서 새로 추가되지 않기 때문이다. (pad_token_id = 3)
tokenizer = PreTrainedTokenizerFast.from_pretrained(
    'skt/kogpt2-base-v2',
    bos_token='</s>', eos_token='</s>', unk_token='<unk>',
    pad_token='<pad>', mask_token='<mask>')

######################################################################
# 입력 문장 다음 토큰 예측 확인

## GPT의 입력, GPT는 이를 시작 문자열로 다음 토큰을 예측
sent = '근육이 커지기 위해서는'

## 입력 문자열을 정수 시퀀스로 인코딩
input_ids = tokenizer.encode(sent, return_tensors='pt')
# tensor([[33245, 10114, 12748, 11357]])
print(input_ids)

## 위 출력의 shape 은 (1, 4)
## - 1 : 배치 크기. 문장을 하나만 넣었기 때문. 대괄호가 두 겹인 이유가 이것이다.
##       바깥 [] = "문장 목록", 안쪽 [] = 실제 토큰들.
##       PyTorch 모델은 항상 여러 건을 한꺼번에 처리하는 것을 전제로 하므로,
##       문장이 1개여도 "1개짜리 배치(batch)"로 취급한다. 문장 3개를 넣으면 (3, 4)가 된다.
## - 4 : 시퀀스 길이. '근육이 커지기 위해서는' 이 토큰 4개로 쪼개졌다는 뜻.
##       주의: 글자 수나 띄어쓰기 단위가 아니라 "토크나이저가 쪼갠 서브워드(subword) 개수"다.
##       같은 문장이라도 토크나이저가 다르면 이 숫자는 달라진다.
##
##       인덱스 | 토큰 ID | 대응 토큰(대략)
##       -------+---------+----------------
##         0    | 33245   | 근육
##         1    | 10114   | 이
##         2    | 12748   | 커지기
##         3    | 11357   | 위해서는

## 정수 시퀀스를 GPT로 입력하여 GPT가 문장을 생성하도록 진행
## - 주어진 문장으로부터 이어서 문장을 생성하도록 하는 것은 model.generate()를 사용
##
## generate()는 아래 "토큰 예측 과정 확인" 절의 작업을 반복 자동화한 것이다.
##   1) 로짓 계산 -> 다음 토큰 1개 선택
##   2) 선택한 토큰을 입력 뒤에 붙임 (길이 4 -> 5)
##   3) 다시 로짓 계산 -> (1, 5, 51200) -> 마지막 위치만 사용
##   4) max_length 에 도달할 때까지 2~3 반복
##
## - repetition_penalty : 이미 나온 토큰의 점수를 깎아 같은 말 반복을 억제
## - use_cache=True     : 위 반복 과정에서 이미 계산한 앞부분 토큰들의 연산을 재사용한다.
##                        매번 4개, 5개, 6개... 전체를 다시 계산하지 않으므로 훨씬 빠르다.
output = model.generate(input_ids,
                        max_length=128,
                        repetition_penalty=2.0,
                        use_cache=True)
output_ids = output.numpy().tolist()[0]

## "33245, 10114, 12748, 11357" 뒤에 여러 출력값이 생성된 것을 확인
# [33245, 10114, 12748, 11357, 23879, 39306, 9684, 7884, 10211, 15177, 26421, 387, 17339, 7889, 9908, 15768, 6903, 15386, 8146, 12923, 9228, 18651, 42600, 9564, 17764, 9033, 9199, 14441, 7335, 8704, 12557, 32030, 9510, 18595, 9025, 10571, 25741, 10599, 13229, 9508, 7965, 8425, 33102, 9122, 21240, 9801, 32106, 13579, 12442, 13235, 19430, 8022, 12972, 9566, 11178, 9554, 24873, 7198, 9391, 12486, 8711, 9346, 7071, 36736, 9693, 12006, 9038, 10279, 36122, 9960, 8405, 10826, 18988, 25998, 9292, 7671, 9465, 7489, 9277, 10137, 9677, 9248, 9912, 12834, 11488, 13417, 7407, 8428, 8137, 9430, 14222, 11356, 10061, 9885, 19265, 9377, 20305, 7991, 9178, 9648, 9133, 10021, 10138, 30315, 21833, 9362, 9301, 9685, 11584, 9447, 42129, 10124, 7532, 17932, 47123, 37544, 9355, 15632, 9124, 10536, 13530, 12204, 9184, 36152, 9673, 9788, 9029, 11764]
print(output_ids)

## 모델 출력값을 텍스트로 변환

# 근육이 커지기 위해서는 무엇보다 규칙적인 생활습관이 중요하다.
# 특히, 아침식사는 단백질과 비타민이 풍부한 과일과 채소를 많이 섭취하는 것이 좋다.
# 또한 하루 30분 이상 충분한 수면을 취하는 것도 도움이 된다.
# 아침 식사를 거르지 않고 규칙적으로 운동을 하면 혈액순환에 도움을 줄 뿐만 아니라 신진대사를 촉진해 체내 노폐물을 배출하고 혈압을 낮춰준다.
# 운동은 하루에 10분 정도만 하는 게 좋으며 운동 후에는 반드시 스트레칭을 통해 근육량을 늘리고 유연성을 높여야 한다.
# 운동 후 바로 잠자리에 드는 것은 피해야 하며 특히 아침에 일어나면 몸이 피곤해지기 때문에 무리하게 움직이면 오히려 역효과가 날 수도 있다.
# 운동을
print(tokenizer.decode(output_ids))

print("\n=============================================")

######################################################################
# 토큰 예측 과정 확인

## GPT는 다음 토큰 예측시, 단어 사전에 있는 수 많은 후보 토큰 중 가장 확률이 높은 토큰을 뽑아 예측

## 토큰 예측시 후보 5건 확인

## generate() 와 달리 model(input_ids)는 "한 번의 forward"만 수행한다.
## 즉, 문장을 이어서 생성하지 않고 예측에 쓰이는 원본 점수(로짓)를 그대로 돌려준다.
##
## 전체 데이터 흐름
##   '근육이 커지기 위해서는'
##           | tokenizer.encode
##           v
##   (1, 4)            정수 시퀀스
##           | model(input_ids) : 임베딩 -> Transformer 블록 12개
##           v
##   (1, 4, 768)       각 토큰의 은닉 표현(hidden state)
##           | LMHead : Linear(768 -> 51200)
##           v
##   (1, 4, 51200)     output.logits  <-- 아래에서 확인하는 값
##           | [0, -1] 슬라이싱
##           v
##   (51200,)          마지막 토큰 다음에 올 후보들의 점수
##           | topk(k=5)
##           v
##   ['무엇보다', '우선', '반드시', '피부', '무엇보다도']
output = model(input_ids)

## torch.Size([1, 4, 51200])
##             |  |  +--- 단어 집합(vocabulary) 크기 = "다음 토큰 후보 개수"
##             |  +------ 입력 토큰 개수(시퀀스 길이)
##             +--------- 배치 크기(한 번에 넣은 문장 수)
##
## 즉 "문장 1개 x 토큰 위치 4개 x 각 위치마다 51200개 후보 점수" 라는 뜻이다.
##
## [왜 마지막 위치 하나가 아니라 4개 위치 전부에 대한 값이 나오는가?]
## 우리가 원하는 것은 '위해서는' 다음에 올 토큰 하나뿐인데 4개가 나오는 이유는,
## GPT가 구조적으로 "모든 위치에서 그 위치 다음에 올 토큰"을 동시에 예측하기 때문이다.
##
##   로짓 위치    | 그 위치까지 GPT가 본 문맥 | 예측하려는 다음 토큰
##   -------------+---------------------------+---------------------
##   logits[0, 0] | 근육                      | 이
##   logits[0, 1] | 근육 이                   | 커지기
##   logits[0, 2] | 근육 이 커지기            | 위해서는
##   logits[0, 3] | 근육 이 커지기 위해서는   | <-- 우리가 원하는 것
##
## 앞의 3개는 정답이 이미 입력 안에 들어있는 예측이라 문장 생성 시에는 버린다.
## 낭비처럼 보이지만 학습할 때는 이 4개 예측을 단 한 번의 forward로 모두 채점할 수 있어
## 훨씬 효율적이다. 그래서 추론(inference) 시에도 같은 구조가 그대로 나온다.
print("logit shape: ", output.logits.shape)

## 위에서 설명한 대로 마지막 토큰 위치의 점수만 뽑아낸다.
##   [0]  -> 배치의 0번째 문장
##   [-1] -> 마지막 토큰 위치(인덱스 3, '위해서는')
## 결과 shape == torch.Size([51200]). 즉, 총 단어 집합 크기만큼의 차원을 가지는 벡터.
logits = output.logits[0, -1]

## [로짓(logit)이란?]
## 확률로 변환되기 전의 "원시 점수"다.
## - 범위 제한이 없다 (음수도 되고, 9.95 처럼 1을 넘어도 된다)
## - 전부 더해도 1이 되지 않는다
## - 다만 "값이 클수록 그 토큰일 가능성이 높다"는 순서 정보는 담고 있다
##
## 로짓 벡터의 각 칸이 하나의 토큰 ID에 대응한다.
##   logits = [ 0.13, -2.5, ...,  9.9551, ..., -1.2  ]
##  index       0     1           23879        51199
##                                  ^
##                        '무엇보다' 토큰의 점수
##
## 확률로 바꾸려면 softmax 를 씌운다.
##   probs = torch.softmax(logits, dim=-1)   # 51200개 값의 합 = 1.0
## 다만 아래처럼 순위만 필요할 때는 softmax 가 필요 없다.
## softmax 는 단조 증가 함수라서 값의 순서를 바꾸지 않기 때문이다.

## topk 결과의 indices 가 곧 "로짓 벡터에서 몇 번째 칸인지 = 토큰 ID" 다.
# torch.return_types.topk(
#   values=tensor([9.9551, 9.4701, 9.1222, 9.1038, 9.0559], grad_fn=<TopkBackward0>),
#   indices=tensor([23879, 12201, 11488, 14564, 20030]))
top5 = torch.topk(logits, k=5)
print("top5: ", top5)

## 토큰 ID를 다시 텍스트로 되돌려 사람이 읽을 수 있게 만든다.
tokens = [tokenizer.decode(token_id) for token_id in top5.indices.tolist()]

# ['무엇보다', '우선', '반드시', '피부', '무엇보다도']
print(tokens)

print("\n=============================================")

######################################################################
# 토큰 랜덤 예측

## 각 시점마다 top 30개의 단어 중에서 랜덤으로 선택하게하여 문장을 생성
import random

sent = '근육이 커지기 위해서는'
input_ids = tokenizer.encode(sent, return_tensors='pt')

while len(input_ids[0]) < 50:
    with torch.no_grad():
        output = model(input_ids)
    logits = output.logits[0, -1]
    top5 = torch.topk(logits, k=30)
    token_id = random.choice(top5.indices.tolist())
    input_ids = torch.cat([input_ids, torch.tensor([[token_id]])], dim=1)

## 실행 때마다 계속 랜덤하게 생성됨
# 근육이 커지기 위해서는 무엇보다 체내에 영양결성분석이 잘된 음식이 더 바람직합니다.
# 우린 하루 1컵 이상 식전에 단백질을 먹어야 할 만큼 체중의 양이 많습니다.
# 우유로 짠 짠 우유가 비만한 사람들을 위해서 오늘부터 우동이 판매가
print(tokenizer.decode(input_ids[0]))