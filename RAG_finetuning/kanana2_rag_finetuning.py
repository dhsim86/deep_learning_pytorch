import os
import sys

if sys.platform == "win32":
    _ALLOC_CONF = "garbage_collection_threshold:0.8"
else:
    _ALLOC_CONF = "expandable_segments:True,garbage_collection_threshold:0.8"

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", _ALLOC_CONF)

import gc

EMPTY_CACHE_WHEN_RESERVED_OVER = 0.8

def _free_gpu_cache_if_large():
    """PyTorch 캐시가 GPU 용량의 일정 비율을 넘었을 때만 OS 에 반납한다."""
    ## 매번 비우면 확실하지만, 반납한 메모리를 다시 받아오는 비용 때문에 학습이 느려집니다.
    ## 그래서 "실제로 캐시가 커졌을 때만" 비웁니다. 메모리가 여유로우면 아무 일도 하지 않습니다.
    if not EMPTY_CACHE_WHEN_RESERVED_OVER:
        return

    if torch.cuda.is_available():
        total = torch.cuda.mem_get_info()[1]                 # 이 GPU 의 전체 VRAM
        if torch.cuda.memory_reserved() <= EMPTY_CACHE_WHEN_RESERVED_OVER * total:
            return
        gc.collect()
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        total = torch.mps.recommended_max_memory()
        if torch.mps.driver_allocated_memory() <= EMPTY_CACHE_WHEN_RESERVED_OVER * total:
            return
        gc.collect()
        torch.mps.empty_cache()


from datasets import load_dataset, Dataset

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

print("\n=============================================")

######################################################################
# 데이터 전처리

# 1. 허깅페이스 허브에서 데이터셋 로드
dataset = load_dataset("iamjoon/klue-mrc-ko-rag-dataset", split="train")

## RAG 성능을 높이기 위해 검색 결과를 바탕으로 답변을 하되 출처를 남기도록 작성
## {{search_result}}에 실제 검색 결과로 대치시켜 학습 및 추론
# 2. 시스템 프롬프트 정의
system_message = """당신은 검색 결과를 바탕으로 질문에 답변해야 합니다.

다음의 지시사항을 따르십시오.
1. 질문과 검색 결과를 바탕으로 답변하십시오.
2. 검색 결과에 없는 내용을 답변하려고 하지 마십시오.
3. 질문에 대한 답이 검색 결과에 없다면 검색 결과에는 "해당 질문에 대한 내용이 없습니다."라고 답변하십시오.
4. 답변할 때 특정 문서를 참고하여 문장 또는 문단을 작성했다면 뒤에 출처는 이중 리스트로 해당 문서 번호를 남기십시오. 예를 들어서 특정 문장이나 문단을 1번 문서에서 인용했다면 뒤에 [[ref1]]이라고 기재하십시오.
5. 예를 들어서 특정 문장이나 문단을 1번 문서와 5번 문서에서 동시에 인용했다면 뒤에 [[ref1]], [[ref5]]이라고 기재하십시오.
6. 최대한 다수의 문서를 인용하여 답변하십시오.

검색 결과:
-----
{search_result}"""

# 3. 원본 데이터의 type별 분포 출력
## synthetic_question: 497
## mrc_question: 491
## paraphrased_question: 196
## mrc_question_with_1_to_4_negative: 296
## no_answer: 404
print("원본 데이터의 type 분포:")
for type_name in set(dataset['type']):
    print(f"{type_name}: {dataset['type'].count(type_name)}")

# 4. train/test 분할 비율 설정 (0.5면 5:5로 분할)
test_ratio = 0.2
train_data = []
test_data = []

# 5. type별로 train / test 데이터 분할
for type_name in set(dataset['type']):
    # 현재 type에 해당하는 데이터의 인덱스만 추출
    curr_type_data = [i for i in range(len(dataset)) if dataset[i]['type'] == type_name]

    # test_ratio에 따라 test 데이터 개수 계산
    test_size = int(len(curr_type_data) * test_ratio)

    # 현재 type의 데이터를 test_ratio 비율로 분할하여 추가
    test_data.extend(curr_type_data[:test_size])
    train_data.extend(curr_type_data[test_size:])

# 6. OpenAI 형식으로 데이터를 변환하는 함수 정의
def format_data(sample):
    # RAG 문서 검색 결과를 문서1, 문서2... 형태로 포매팅
    search_result = "\n-----\n".join([f"문서{idx + 1}: {result}" for idx, result in enumerate(sample["search_result"])])

    # OpenAI format으로 변환
    return {
        "messages": [
            {
                "role": "system",
                "content": system_message.format(search_result=search_result),
            },
            {
                "role": "user",
                "content": sample["question"],
            },
            {
                "role": "assistant",
                "content": sample["answer"]
            },
        ],
    }

# 7. 분할된 데이터를 OpenAI format으로 변환
train_dataset = [format_data(dataset[i]) for i in train_data]
test_dataset = [format_data(dataset[i]) for i in test_data]

# 8. 최종 데이터셋 크기 출력
## 전체 데이터 분할 결과: Train 1509개 , Test 375개
print(f"\n전체 데이터 분할 결과: Train {len(train_dataset)}개 , Test {len(test_dataset)}개")

# 9. 분할된 데이터의 type별 분포 출력
## synthetic_question: 398
## mrc_question_with_1_to_4_negative: 237
## paraphrased_question: 157
## no_answer: 324
## mrc_question: 393
print("\n학습 데이터의 type 분포:")
for type_name in set(dataset['type']):
    count = sum(1 for i in train_data if dataset[i]['type'] == type_name)
    print(f"{type_name}: {count}")

## synthetic_question: 99
## mrc_question_with_1_to_4_negative: 59
## paraphrased_question: 39
## no_answer: 80
## mrc_question: 98
print("\n테스트 데이터의 type 분포:")
for type_name in set(dataset['type']):
    count = sum(1 for i in test_data if dataset[i]['type'] == type_name)
    print(f"{type_name}: {count}")

# 데이터셋 샘플 확인

# [
#   {'role': 'system', 'content': '당신은 검색 결과를 바탕으로 질문에 답변해야 합니다.\n\n다음의 지시사항을 따르십시오.\n1. 질문과 검색 결과를 바탕으로 답변하십시오.\n2. 검색 결과에 없는 내용을 답변하려고 하지 마십시오.\n3. 질문에 대한 답이 검색 결과에 없다면 검색 결과에는 "해당 질문에 대한 내용이 없습니다."라고 답변하십시오.\n4. 답변할 때 특정 문서를 참고하여 문장 또는 문단을 작성했다면 뒤에 출처는 이중 리스트로 해당 문서 번호를 남기십시오. 예를 들어서 특정 문장이나 문단을 1번 문서에서 인용했다면 뒤에 [[ref1]]이라고 기재하십시오.\n5. 예를 들어서 특정 문장이나 문단을 1번 문서와 5번 문서에서 동시에 인용했다면 뒤에 [[ref1]], [[ref5]]이라고 기재하십시오.\n6. 최대한 다수의 문서를 인용하여 답변하십시오.\n\n
#     검색 결과:\n-----\n
#     문서1: 브레히트는 서독에서도 젊은 작가들의 큰 지표와 자극이 되었으나 사회의 변혁을 지향하고 \'오늘의 세계를 연극으로 재현하는\' 사회적인 자세뿐만 아니라 예술적인 시도도 크게 강조되었다. 물론 전후에 수입된 부조리 연극의 영향도 무시할 수 없다(힐데스하이머, 하이, 그라스 등). 대체로 낡은 감동적 연극이 이미 관객의 마음을 사로잡을 수 없게 되었다는 경향은 <차가운 빛> 이후 추크마이어의 부진을 보아도 알 수가 있다. 이 밖에 1950년대부터 60년대에 걸쳐서 서독에서는 아르젠, 즈르바노스, 아스모디, 비트링거, 히르비에, 뷴셰, 미헤르젠, 렌츠, 헤커, 만스페르트, 발트만, 마이어멜스 등이 등장했으나 뒤렌마트의 <귀부인 고향으로 돌아오다>나 프리시의 <안도라>에 미칠 만한 작품은 낳지 못하였다. 1963-1964년에 호흐후트의 <신(神)의 대리인>, 키프하르트의 <오펜하이머의 사건>, 바이스의 <마라의 박해와 암살> 등이 등장하면서부터 독일극(獨逸劇)은 또다시 국제적인 주목을 끌게 되었다. 그라스는 동서 독일의 분열에 메스를 댄 <천민(賤民)의 폭동연습>을 쓰고, 발저는 브레히트의 한계와 그의 극복을 탐구하면서 일련의 사회적인 희곡을 발표했다. 이들 작품은 모두 액추얼한 작가의 자세를 나타내고 있는데, 한편 프리시나 뒤렌마트는 희곡의 사회적인 유효성에 관한 의문이 브레히트 비판이라는 형태로 나타났고, 반대로 바이스는 명확한 정치적 코미트의 자세를 분명히 했다. 가장 젊은 세대로는 서독의 슈펠, 랑게 등이 기대된다.\n-----\n
#     문서2: 누구나 겪었을, 누구나 겪고 있을 법한 일을 풍자와 해학이 넘치는 문체로 현실감 있게 그려온 소설가 전성태 씨가 등단 20년을 맞아 새 소설집 두 번의 자화상(창비)을 내놓았다.소설집에 실린 12편의 작품은 인생의 다양한 풍경을 담았다. 불법체류자 신분으로 숨죽여 살다 고국으로 돌아가는 외국인 여성의 이야기를 다룬 ‘배웅’, 중공군과 인민군 병사들이 묻힌 적군 묘지를 돌보는 늙은 상점 주인이자 퇴역 군인의 노래 ‘성묘’, 북한에 납치됐다 돌아온 어부들과 실향민의 이야기인 ‘망향의 집’은 묵직한 울림을 준다.새터민이 정착해 사는 아파트 단지에서 일하는 경비원이 폐품 더미에서 북한 신문을 발견했을 때의 반응을 보면 당황스럽다가도 웃음이 터진다. “로, 동, 신, 문… 로동신문? 어디 조합 신문인가?”(‘로동신문’ 중) 또 다른 작품인 ‘밥그릇’을 읽으면 전국을 떠도는 골동품 수집상들을 골탕먹이는 촌로의 능청스러움에 빙그레 웃음 짓게 된다.책의 맨 앞에 실린 ‘소풍’과 마지막 작품인 ‘이야기를 돌려드리다’는 자전적 요소가 담긴 작품으로, 부모 세대의 치매를 다뤘다. 슬픈 현실 속에서도 주인공은 희망을 놓지 않는 모습을 보여준다.\n-----\n
#     문서3: 지구에서 억압받지 않고 웃길 수 있는 곳을 찾지 못해 우주까지 밀려온 찰리 채플린은 한국에서 만났던 만담가 신불출과 재회한다. 말없이 몸짓을 주고받던 두 사람은 우주 헬멧을 벗고 대화를 나눈다.“우, 우리가 웃기는 게 우, 우주의 위협이 되면 어쩌지?”(채플린) “벼, 별수 있나요? 또 다른 데로 가야지.”(신불출) “말하는 동안 몇십 초밖에 안 남았어요. 어쩌죠?”(신불출) “어쩌긴? 웃겨야지. 우주 전체를.”(채플린)극단 연희단거리패 제작으로 서울 혜화동 게릴라극장에서 공연 중인 연극 ‘레드 채플린’(오세혁 극작, 이윤주 연출)은 젊은 창작인들의 연극적 상상력과 재기발랄함이 돋보이는 사회 풍자 코미디다.연극은 채플린이 꿈속 여행에서 겪는 에피소드를 코믹하게 펼쳐낸다. 조지프 매카시 미국 상원의원이 “채플린이 공산주의자”라고 발표하는 실제 영상을 뒤로하고 채플린은 꿈을 꾼다. 채플린은 ‘모던 타임스’ ‘개의 일생’ ‘위대한 독재자’ 등 ‘빨간색’이란 혐의를 받는 영화 주인공 중 한 명을 매카시식으로 ‘빨갛다’고 지목해야 용서받을 수 있다는 얘기를 듣고는 침대를 비행기 삼아 아메리카를 떠난다. 일제 강점기 한반도로 날아온 그는 만담 중 ‘체제 위협’ 발언으로 순사들에게 두들겨 맞는 신불출에게 동병상련을 느낀다. 해방 직후 친일파가 득세하는 한국과 ‘천삽 뜨고 허리 펴기’ 운동이 시작된 1950년대 후반의 북한, 2013년 서울역 광장, 거지와 병사가 나오는 시공이 불분명한 어느 곳을 거쳐 우주에 다다른 채플린은 신불출과 함께 힘겹게 코미디를 펼친다. ‘낙인 찍기’로 대표되는 경직된 사회 체제와 코미디 같은 현실에 대한 날 선 비판과 풍자, 예술과 표현의 자유와 한계에 대한 고민이 극 속에 녹아 있다.무엇보다 ‘우리 시대 채플린’이 되고 싶어 하는 젊은 연극인의 열망과 패기를 느낄 수 있다. 공연은 내달 12일까지, 2만~3만원.\n-----\n
#     문서4: 뮤지컬 ‘라카지’더할 나위 없이 화려한 ‘쇼뮤지컬’의 진수를 만끽할 수 있는 무대다. 클럽 ‘라카지오폴’을 운영하는 중년 게이 부부의 아들이 극우파 보수 정치인의 딸과 결혼을 선언하면서 일어나는 에피소드를 그린다. 2012년 국내 초연에 이은 두 번째 한국어 라이선스 공연이다. 초연에 이어 주인공 앨빈 역을 맡은 정성화의 능청스런 연기와 풍부한 성량이 극에 활력을 불어넣는다. 다만 뮤지컬 팬이라면 올해 쉴 새 없이 이어진 ‘드래그 퀸(여장 남자 무희)’ 퍼포먼스에 피로감을 느낄 법도 하다. 내년 3월8일까지, 서울 역삼동 LG아트센터.연극 ‘슈만, 나의 영혼 나의 사랑’‘연극과 클래식 음악의 만남’을 주제로 기획된 ‘산울림 편지 콘서트’의 두 번째 무대. 낭만주의 음악가 슈만과 부인 클라라의 순애보적 사랑을 두 사람이 주고받은 편지를 낭독하는 형식으로 그린다. ‘헌정’ ‘시인의 사랑’ 등 가곡과 ‘트로이메라이’ ‘카니발’ 등 슈만이 남긴 소품들이 극의 진행에 맞춰 라이브로 연주된다. 극 후반 슈만의 죽음과 함께 리릭 테너 김현호의 맑은 목소리로 ‘헌정’이 다시 흐를 때 가슴이 뭉클해진다. 이호성 조윤미 김민철 등 출연. 오는 30일까지, 서울 서교동 산울림소극장. 시와 그림의 아름다운 조응시정(詩情)과 화의(畵意)가 조응하는 이색 전시회다. 원로 시인 김남조 씨의 미수(米壽·88세)를 기념해 마련된 ‘시가 있는 그림전’에는 대한민국 예술원 회원 민경갑 화백을 비롯해 박돈, 황주리, 전준엽, 이명숙, 이희중, 김선두, 정일 등 13명의 화가들이 김 시인의 다양한 시를 형상화한 그림 또는 조각 작품을 내보인다. 황주리 씨는 김 시인의 ‘편지’를 꽃잎 속에서 기타를 치거나 어깨동무를 하며 서로 끌어안는 등의 삽화 같은 풍경화를 걸었다. ‘퓨전 한국화가’ 전준엽 씨는 시인의 ‘내가 흐르는 강물에’를 푸른 강과 소나무를 배경으로 한 몽환적인 작품을 각각 내놓았다. 내년 1월10일까지, 서울 청담동 갤러리 서림. (02)515-3377 국제시장1950년 6·25전쟁부터 현재까지 격변의 시대를 관통하며 가족을 위해 희생한 우리 시대 아버지 덕수의 파란만장한 삶을 그렸다. 영화 사상 처음으로 그려진 흥남 철수 장면부터 파독 광부, 베트남전 파견 근로자를 거쳐 남북 이산가족 찾기로 이어지는 드라마가 관객의 눈물과 웃음을 이끌어낸다. 황정민, 김윤진, 오달수 출연. 윤제균 감독.\n-----\n
#     문서5: 인물 몇이 등장하여 매력 있는 대화하고 행동하면서 이야기 하나를 교묘한 극작술에 기초하여 묘사(描寫)한다고 가정하자. 웃고 울며, 기대에 숨을 죽이게 되고 무대상 인물과 함께 손에 땀을 쥐면서 관객의 전신경은 무대에 쏠리게 되고 관람하고서 크게 감명받으며, 때때로 그 감명이 뜻밖에 적고 이 연극은 무엇을 노리고 관객에게 무엇을 호소하려는지 알 수 없는 때도 있다. 그것은 주제가 명확하지 않기 때문이다. 주제란 그 작품의 모든 세부 효과를 단일한 방향을 향하여 결합시키는 \'붉은 실\'과 같다.날짜=2013-02-19 셰익스피어는 『로미오와 줄리엣』에서 봉건제도의 특규성이 있는 도덕에 도전하는 사랑을 묘사하였다. 아서 밀러는 『세일즈맨의 죽음』에서 미국 자본주의사회에서 인간성 왜곡과 파괴를 묘사하였다. 이 작가들은 그것을 인간을 향한 한없는 애정과 인간답지 못한 것을 대상으로 하는 심한 분노로 압축하여 한 묘사이다. 즉 \'붉은 실\'이란 작자의 통찰이고 현실과 대결하는 자세이며, 극작술의 기저(基底)에 있는 인생관과 세계관이다. 작자의 그 시점(視點)으로 모든 세부가 명확해지고 연계되며, 조립되어 비로소 주제는 작품 전체에 일관되어서 보는 자에게 확고한 감명을 준다.신극은 사회의 진실과 대면을 두려워하지 않는다. 오히려 모순이야말로 사회가 발전하는 원동력이고 사회는 변동하는 것, 역사법칙을 좇아 발전하고 전화(轉化)한다는 인식에 입각(立脚)하여 창작한다. 조금이라도 뜻있는 신극인은 그 신념에 입각하여 창작하는 활동에 참가하듯이 연극은 주제가 명확해야 해서 작자의 사상이 확고해야 한다.'},
#   {'role': 'user', 'content': '정치적 코미트를 담고 있는 작품'},
#   {'role': 'assistant', 'content': "정치적 코미트를 담고 있는 작품으로는 다음과 같은 예시들이 있습니다.\n\n1. **브레히트의 작품들**: 브레히트는 사회 변혁을 지향하며 '오늘의 세계를 연극으로 재현하는' 사회적인 자세를 강조한 작품들을 많이 남겼습니다. 그의 작품들은 사회적, 정치적 메시지를 담고 있으며, 젊은 작가들에게 큰 영향을 미쳤습니다[[ref1]].\n\n2. **뒤렌마트의 <귀부인 고향으로 돌아오다>**: 이 작품은 사회적 유효성에 대한 의문을 제기하며, 브레히트 비판이라는 형태로 나타났습니다[[ref1]].\n\n3. **프리시의 <안도라>**: 프리시 역시 사회적 유효성에 대한 의문을 제기하는 작품을 통해 정치적 메시지를 전달했습니다[[ref1]].\n\n4. **호흐후트의 <신(神)의 대리인>**: 이 작품은 1963-1964년에 등장하여 독일극이 국제적인 주목을 받게 만든 작품 중 하나로, 명확한 정치적 코미트를 담고 있습니다[[ref1]].\n\n5. **키프하르트의 <오펜하이머의 사건>**: 이 작품도 정치적 메시지를 담고 있으며, 독일극의 국제적 주목을 이끌어낸 작품 중 하나입니다[[ref1]].\n\n6. **바이스의 <마라의 박해와 암살>**: 바이스는 명확한 정치적 코미트를 분명히 한 작가로, 이 작품 역시 정치적 메시지를 담고 있습니다[[ref1]].\n\n7. **연극 '레드 채플린'**: 이 연극은 찰리 채플린이 꿈속 여행에서 겪는 에피소드를 통해 경직된 사회 체제와 코미디 같은 현실에 대한 날 선 비판과 풍자를 담고 있습니다. 예술과 표현의 자유와 한계에 대한 고민도 극 속에 녹아 있습니다[[ref3]].\n\n이들 작품은 모두 사회적, 정치적 메시지를 담고 있으며, 작가의 정치적 코미트를 분명히 하고 있습니다."}
# ]
print(train_dataset[345]["messages"])

## 데이터셋 타입 변환
# 리스트 형태에서 다시 Dataset 객체로 변경
print(type(train_dataset))
print(type(test_dataset))

train_dataset = Dataset.from_list(train_dataset)
test_dataset = Dataset.from_list(test_dataset)
print(type(train_dataset)) # <class 'datasets.arrow_dataset.Dataset'>
print(type(test_dataset))

print("\n=============================================")

# 데이터셋 저장
test_dataset.save_to_disk("rag_finetuning_test_dataset")

######################################################################
# 챗 템플릿 적용

## 사용할 모델 ID
model_id = "kakaocorp/kanana-2-1.3b-instruct"

## 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

## 모델 로드
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    attn_implementation="sdpa",
    trust_remote_code=True,
)

## 모델의 챗 템플릿 적용
### qwen의 챗 탬플릿 형식(ChatML, by LM Studio)
# <|im_start|>system
# 시스템 프롬프트<|im_end|>
# <|im_start|>user
# 유저 프롬프트<|im_end|>
# <|im_start|>assistant
# 모델 응답<|im_end|>

## kanana-2 도 같은 ChatML 계열이지만, assistant 턴 앞에 빈 추론 블록이 하나 더 붙는다.
## (thinking 모드를 끈 상태를 뜻하는 표식)
# <|im_start|>assistant
# <think>
#
# </think>
#
# 모델 응답<|im_end|>

text = tokenizer.apply_chat_template(train_dataset[0]["messages"], tokenize=False, add_generation_prompt=False)
# <|im_start|>system
# 당신은 검색 결과를 바탕으로 질문에 답변해야 합니다.
# 
# 다음의 지시사항을 따르십시오.
# 1. 질문과 검색 결과를 바탕으로 답변하십시오.
# 2. 검색 결과에 없는 내용을 답변하려고 하지 마십시오.
# 3. 질문에 대한 답이 검색 결과에 없다면 검색 결과에는 "해당 질문에 대한 내용이 없습니다."라고 답변하십시오.
# 4. 답변할 때 특정 문서를 참고하여 문장 또는 문단을 작성했다면 뒤에 출처는 이중 리스트로 해당 문서 번호를 남기십시오. 예를 들어서 특정 문장이나 문단을 1번 문서에서 인용했다면 뒤에 [[ref1]]이라고 기재하십시오.
# 5. 예를 들어서 특정 문장이나 문단을 1번 문서와 5번 문서에서 동시에 인용했다면 뒤에 [[ref1]], [[ref5]]이라고 기재하십시오.
# 6. 최대한 다수의 문서를 인용하여 답변하십시오.
# 
# 검색 결과:
# -----
# 문서1: 확증 편향은 일반적으로 내가 원하는 바대로 정보를 수용하고 판단한다는 뜻이다. 한자성어로는 아전인수(我田引水)라는 말이 있다. 확증 편향에 의한 아전인수식 사고는 스스로가 이러한 판단을 참이라고 믿는 다는 점에서 거짓임을 뻔히 알지만 남을 속이고자 하는 견강부회(牽强附會)와는 다른 점이 있다.
# 
# 정보처리이론에서는 확증 편향은 자기실현적 예언 현상인 행동적 확증과 연관짓는다. 자신이 갖고 있는 신념 때문에 결과적으로 그에 따라 행동하고 결과를 얻는다는 것이다. 
# 
# 심리학에서는 종종 정보의 선택적 수용과 거부 모델로 확증 편향을 설명한다. 어떠한 정보를 신뢰하고 어떤 정보는 불신하는 가에 따라 동일한 정보들이 주어지더라도 다른 결론을 내릴 수 있다. 이러한 현상은 현재 일어나고 있는 일 뿐만아니라 과거의 일에 대한 기억에도 영향을 미친다. 같은 사건을 겪었더라도 사람마다 그 의미를 다르게 해석할 수 있다. 1950년 영화 《라쇼몽》은 하나의 사건에 대해 서로 다른 기억을 갖고 있는 목격자 셋의 이야기를 들려주어 확증 편향에 따른 기억의 재해석 사례로 자주 언급된다.
# -----
# 문서2: 스탠퍼드의 심리학자 로버트 맥컨은 확증이 형성되는 과정을 "차가운" 인지와 "뜨거운" 동기 부여의 메커니즘으로 설명한다. 
# 
# 인지적 메커니즘은 사람들이 복잡한 문제를 다루는 능력에 한계가 있기 때문에 확증 편향이 생긴다고 설명하는 것이다. 모든 정보를 다 갖출 수는 없기 때문에 주어진 것만으로 일종의 지름길인 휴리스틱을 이용한다는 것이다. 사람들은 정보를 유형화 하여 유용성을 따지거나 두 증거의 차이점을 비교하거198–99 예상되는 결과를 미리 생각해 두고 거꾸로 맞추어 보면서 문제를 해결한다. 이런 방식의 문제 해결은 일어날 수 있는 모든 경우의 수를 다 검토할 수는 없지만, 세계관 전체에 걸린 문제가 아니라면 자신의 신념을 유지하면서 문제를 다룰 수 있게 된다20
# 
# 동기 부여 메커니즘은 믿음에 대한 욕구에 의해 작동한다197 사람들은 대개 부정적인 생각보다는 긍정적인 것을 선호하는 경향을 보인다. 이를 폴리애너 원리라고 한다. 어떤 주장의 결론이 논거를 충분히 갖추면 보다 진실하다고 신뢰받는 이유다. 심리학 실험에서 사람들은 자신이 부정하고자 하는 주장에 대해 보다 엄격한 증거를 요구하는 경향을 보였다. 자신의 기존 생각에 거스르지 않는 것은 "제가 이것을 믿어도 될까요?" 정도로 검토한다면 그렇지 않은 것은 "제가 이것을 꼭 믿어야 하나요?"라고 반응한다. 태도의 일관성은 바람직한 품성이지만, 이 역시 확증 편향의 원인이 되기도 한다. 새롭고 놀라운 정보를 접했을 때 이를 쉽게 받아들이지 못하는 걸림돌로 작용한다. 사회심리학자 지바 쿤다(Ziva Kunda)는 인지적 메카니즘과 동기 부여 메커니즘을 결합하여 편향을 만드는 것은 동기적 측면이지만 편향의 규모를 결정하는 것은 인지적 과정이라고 주장하였다198
# -----
# 문서3: 동일한 사건을 함께 경험한 사람이라도 기억은 서로 다를 수 있다. 인간의 장기 기억은 각자의 경험 속에서 주관적으로 중요한 것, 감정과 결합된 것 들이 강하게 기억되며 세세한 것 보다는 전체적으로 요약된 인상만이 남게 된다. 어떤 것은 쉽게 기억되고 되살릴 수 있고 어떤 것은 잊어버리거나 왜곡되는 것을 "선택적 기억" 또는 "편향적 기억"이라고 할 수 있다. 스키마 이론은 이미 기대하고 있는 것과 들어맞는 정보가 그렇지 않은 정보보다 더 잘 기억되며 되살리기도 쉽다고 설명한다 또는 놀랄만한 정보 역시 다른 정보 보다 잘 기억된다. 경험의 기억 역시 사람들이 기존에 갖고 있는 기대와 예측이 작용하는 것이다. 
# 
# 참가자들에게 한 여성의 성격 프로필을 제공한 실험이 있었다. 프로필에는 내성적 성격의 특징과 외향적 성격의 특징이 섞여 있었다 일정 시간이 지난 뒤 참가자를 두 그룹으로 나누어 한 그룹에는 이 여성의 직업이 도서관 사서라고 소개하고 다른 그룹에는 부동산 중계사라고 소개하였다. 사서라는 설명을 들은 그룹은 여성의 내성적 성격을 더 많이 기억해 냈고, 중계사라는 설명을 들은 그룹은 외향적 성격을 더 많이 기억했다. 참가자가 갖고 있는 직업에 대한 이미지가 기억을 되살리는 데 영향을 준 것이다. 한편, 이와 별개로 진행된 동일한 실험에서 참가자들은 자신의 성향과 들어맞는 프로필을 더 잘 기억했다. 내성적 성향의 사람들은 내성적 프로필을, 외향적 성향의 사람들은 외향적 프로필을 기억해 내는 것이 더 쉬웠다. 이 경우엔 프로필을 읽으면서 자신의 상태와 감정적으로 결합하여 기억한 것이라고 볼 수 있다. 
# 
# 감정적 요인 역시 기억의 편향에 관여한다. 예를 들어 우울증이나 불안장애를 겪는 사람들은 긍정적 경험에 대한 기억이 억제되고 그 기억을 되살리는 것에도 어려움을 느낀다. 부정적 요소를 인식하면 그것에 대한 집중이 과도하게 진행되어 다른 긍정적 요소를 파악하기 어렵다. O. J. 심슨 사건의 판결에 대한 참가자의 감정을 묻는 실험에 판결 1주 후 , 2개월 후, 1년 후로 나누어 진행된 응답 결과 참가자들은 판결에 대한 평가를 계속해서 바꾸었는데, 특히 시간이 지날수록 처음 사건을 회상하고 그 당시의 느낌을 떠올리기 보다는 응답할 당시의 감정 상태에 따라 의견을 결정하는 성향을 보였다. 14개월이 지난 후 참가자들 상당수는 이전의 응답과 별개로 현재의 판단에 따라 기억을 재구성 하였다. 
# 
# 과거에 경험한 일이라 할지라도 그것을 회상하는 일은 현재의 감정 상태에 관계되어 있다 배우자와 사별한 사람들을 대상으로 한 설문조사에서 많은 사람들이 사별 후 6개월까지 매우 큰 슬픔을 느낀다고 응답하며 5년이 지난 뒤에는 비교적 덤덤하게 받아들이게 된다. 배우자와 사별한 지 5년이 지난 사람에게 사별 후 6개월의 감정 상태를 기억해보라고 요구하면 과반 이상이 현재의 상태를 기준으로 당시도 덤덤하였다고 대답하였다. 현재의 감정 상태가 과거의 감정적 기억을 재구성하게 되는 것이다
# 
# 기억이 일정하게 유지 되지 않고 변형되는 현상은 목격 진술에서 심각한 문제를 일으킬 수 있다. 2008년 있었던 강화도 모녀 납치 살해 사건의 수사에서 목격자들의 진술이 일치하지 않아 수사가 난항을 겪은 사례가 있다. 이 경우도 목격자 각자의 편향적 기억이 원인으로 지목된다.
# -----
# 문서4: 네거티브 광고(Negative Marketing)는 금기시되는 소재를 활용하거나 사물의 부정적인 측면을 사용하는 광고이다. 부정광고, 네거티브 어필이라고도 한다.
# 
# 부정해서 강한 긍정을 나타내는 방식으로 터부시되는 소재의 활용이나 부정적인 이미지를 사용해 광고의 효과를 낸다. 하지만 너무 강한 표현이거나, 사람들에게 지나치게 부정적인 반응을 일으킬 수 있는 내용이면 역효과를 불러와 부정적인 브랜드 이미지가 형성 될 수 있다. 때문에 매우 신중하게 해야하는데 네거티브 광고를 하고자 하는 기업이나 회사에서는 법적인 부분이나 소비자들의 반응을 충분히 고려하고 준비해서 내보내야 한다.
# 네거티브 광고의 사례로는 가격이 비싼 제품의 경우, 비싸니 사지말라라는 식의 홍보를 통해 그 제품을 사면 하이클래스가 되는 듯한 기분을 느끼게 해주는 효과를 내어 매출이 급증한 사례가 있었다.
# 
# 홍보효과 뿐만 아니라 정치적으로 많이 사용되고 있는데, 유권자들이 부정적인 면이 강한 후보를 먼저 제외시키기 때문에 효과적이다. 미국의 대선에 많이 사용되어 이슈가 되었다. 최근에는 우리나라에서도 한나라당 경선때 '네거티브 선거전'이라고 하여 상대방의 약점이나 비리를 폭로하여 지지율을 떨어뜨리자는 전략으로 사용되었는데, 잘못쓰이면 정치공작으로 사용될 수 있어 신중히 사용해야 한다.
# -----
# 문서5: 확증편향(confirmation bias)이란, 선택적 사고의 일종이다. 사람은 자기 자신의 신념을 확실히 증명해주는 것들을 쉽게 찾거나, 발견하는 경향이 있으며, 그에 반대로 자기의 신념에 반대되는 것은 무시하거나, 덜 찾아보던가, 혹은 가치가 낮다고 생각하는 경향이 있다. 예를 들면, 보름달 저녁에는 회사에 사고가 많이 일어난다고 믿는 사람이 있다고 하자. 이 경우, 그 사람은 보름달 저녁에 일어났던 사고만 주목해 버리고, 보름달 이외의 기간에 일어났던 사고는 주의를 기울이지 않게 된다. 이러한 것이 반복되면, 보름달이 사고와 관계 있다는 신념은 부당하게 강화된다. 
# 
# 이처럼 처음에 가졌던 선입견이나 신념을 지지하거나 뒷받침하는 정보에 더 비중을 실어주게 되고, 이것에 반대되는 정보를 가볍게 보려는 경향은, 신념이나 선입견이 편견일 경우에는 더욱 현저하게 나타난다. 신념이 확실한 증거이거나 유효한 확증 실험에 의해 뒷받침된 경우라면, 신념에 맞지않는 정보에 더 무게를 둔다고 해서 잘못된 방법으로 길을 헤매지는 않을 것이다. 만약, 정말로 자신의 가설을 부정할 증거에 대하여 무시한다면, 합리와 맹목을 구분하는 마지노선을 넘어 버리게 된다.
# 
# 사람은 다시 한번 확인하는 정보, 즉 자기 자신의 의견에 유리하거나, 자기의견을 지지할 것 같은 정보를 지나치게 신뢰한다는 것은 많은 연구로부터 이미 밝혀져있다. 토마스 기로비치(Thomas Gilovich)는 “재확인 적인 정보에 지나치게 높은 점수를 주는 것은, 아마 인식론적으로 불리한 정보를 무시해 버리는 쪽이 편한하기 때문일 것이다”라고 말하고 있다. 정보가 얼마나 자기 자신의 의견을 뒷받침 하는 가를 생각하는 것이, 그러한 것이 얼마나 자기 의견을 반론 하는 가를 생각하는 것보다 훨씬 쉽다.
# 
# 예지몽(clairvoyant dream)에 대한 실험을 생각해 보자. 성공 사례는 종종 분명히 성공한 것으로 보이거나, 혹은 자료가 쉽게 성공으로 간주되어 기억되지만, 실패한 사례의 경우는 그것이 실패했다고 생각하는 것 조차도 지적노력이 들고, 그것이 중요하다고 생각까지 기억해야 한다. 자기 자신의 의견을 뒷바탕하며 지지하고 확증한 사항에 더 무게를 두는 경향은, 기억에도 영향을 준다. 자기 자신의 의견에 관하여 떠올리려 하려고 할 때, 자기 자신의 의견을 지지하는 정보가 더 쉽게 떠오른다.
# 
# 연구자는 자신의 가설을 확증하도록 실험을 설정하거나, 정보를 취하기 때문에 때로는 재확인 선호 편향에 빠진다. 그들은 가설에 반대되는 정보를 취급하려고 할 때, 자기 자신에게 불리한 것을 인정하려고 하지 않기 때문에 문제가 복잡해진다. 사회 과학자의 대부분 특히, 자의적으로 설정한 역사 시대에 관하여 출생순서와 ‘급진적 사고’와 같은 불분명한 변수 사이의 상관관계를 발견하려고 하는 사람도 확증편향에 빠진다.
# 
# 연구자는 자신의 의견과 반대되는 의견을 지지하는 연구자와 실험 계획을 공동으로 계획하면, 재확인 선호 편향을 피하든지 줄일 수 있을 것이다. 각각의 개인은 재확인 선호 편향에 관하여 항상 생각에 염두에 두고 반증하는 정보를 잘 살펴보지 않으면 안된다. 이러한 것이 자연스러운 것은 아니므로 일반인들은 재확인 선호편향에 빠지지 않을 수 없다.<|im_end|>
# <|im_start|>user
# 긍정적인 생각을 부정적인 생각보다 선호하는 경향을 나타내는 용어는?<|im_end|>
# <|im_start|>assistant
# <think>
# 
# </think>
# 
# 긍정적인 생각을 부정적인 생각보다 선호하는 경향을 나타내는 용어는 "폴리애너 원리"입니다. 이 용어는 사람들이 대개 부정적인 생각보다는 긍정적인 것을 선호하는 경향을 설명하는 데 사용됩니다. [[ref2]]<|im_end|>
print(text)

print("\n=============================================")

######################################################################
# 모델 학습 준비

## LoRA 튜닝 설정
peft_config = LoraConfig(
    lora_alpha=32,
    lora_dropout=0.1,
    r=8,
    bias="none",
    target_modules=["q_proj", "v_proj"],
    task_type="CAUSAL_LM",
)

## 파인튜닝 설정
args = SFTConfig(
    output_dir="kanana2-1.3b-rag-ko", # 저장될 디렉토리와 저장소 ID
    num_train_epochs=3, # 학습할 총 에포크 수
    per_device_train_batch_size=4, # GPU당 배치 크기
    gradient_accumulation_steps=2, # 그래디언트 누적 스텝수
    gradient_checkpointing=True, # 그레디언트 체크포인팅

    loss_type="chunked_nll", # 책과 다른 부분
    # 손실을 계산할 때 로짓을 통째로 만들지 말고 조각내서 계산하라는 지시

    optim="adamw_torch_fused", # 옵티마이저
    logging_steps=1, # 학습 상황 (loss, 학습률) 로그 기록 주기
    save_strategy="steps", # 모델 체크포인트를 어떤 기준으로 저장할지
    save_steps=50, # 모델 체크포인트를 몇 스텝마다 저장할지
    bf16=True, # 계산시 bfloat16 사용
    learning_rate=1e-4, # 학습률
    max_grad_norm=0.3, # 그래디언트 클리핑, 모델 업데이트할 때 그레디언트를 제한하여 너무 급격하게 변하지 않도록 한다.
    warmup_ratio=0.03, # 워밍업 비율
    lr_scheduler_type="constant", # 학습률 스케줄러 지정
    push_to_hub=False, # 허브 업로드 안함
    remove_unused_columns=False, # 데이터셋의 컬럼 유지
    dataset_kwargs={"skip_prepare_dataset": True}, # 자동 전처리 OFF
    report_to="none" # 학습 지표 전송 OFF
)

print("\n=============================================")

######################################################################
# 모델 학습을 위한 데이터 준비

# 학습 데이터의 길이를(패딩 포함) 이 값의 배수로 올림 (None 이면 끔)
PAD_TO_MULTIPLE_OF = 128

## 정수 인코딩
### 학습 데이터에 챗 템플릿 적용 후 정수 인코딩, 모델의 입력(input_ids)와 모델의 응답(labels) 분리

## 마지막 assistant 메시지만 loss에 포함하도록 모델 고유 챗 템플릿을 토큰화
def _as_token_list(encoded):
    """Transformers의 BatchEncoding/list 반환값에서 1차원 토큰 리스트를 꺼낸다."""
    if hasattr(encoded, "data"):
        encoded = encoded["input_ids"]
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    return list(encoded)

## 학습 데이터에서 샘플별 가장 긴 토큰 길이를 구하고, 이를 max_seq_length로 지정
token_lengths = []
for data in train_dataset:
    encoded = tokenizer.apply_chat_template(
        data["messages"],
        tokenize=True,
        add_generation_prompt=False, # 이미 assistant 응답이 들어 있으므로 생성 프롬프트를 붙이지 않는다
        truncation=True
    )
    token_lengths.append(len(_as_token_list(encoded)))

max_seq_length = max(token_lengths)
print("max_seq_length: ", max(token_lengths))

def tokenize_with_assistant_labels(messages):
    ## [이 함수가 쓰는 트릭]
    ## "정답이 어디서 시작하는지"를 문자열에서 찾는 대신, 같은 챗 템플릿을 두 번 적용해서 길이 차이로 알아냅니다.
    ##   (A) 전체(system+user+assistant)를 토큰화        -> 학습에 넣을 input_ids
    ##   (B) assistant를 뺀 뒤 add_generation_prompt=True -> "<|im_start|>assistant\n" 까지만 만들어진 프롬프트
    ## (B)는 (A)의 앞부분과 정확히 일치하므로, len(B)가 곧 정답이 시작되는 위치가 됩니다.
    ## 토큰 문자열을 직접 찾는(find) 방식보다 안전합니다. 모델마다 다른 제어 토큰 형식을 신경 쓸 필요가 없기 때문입니다.
    if not messages or messages[-1]["role"] != "assistant":
        raise ValueError("messages의 마지막 항목은 학습할 assistant 메시지여야 합니다.")

    ## (A) 전체 대화를 토큰화 (이게 모델의 입력이 된다)
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False, # 이미 assistant 응답이 들어 있으므로 생성 프롬프트를 붙이지 않는다
        truncation=True,
        max_length=max_seq_length,
    )
    input_ids = _as_token_list(encoded)

    ## (B) assistant를 뺀 프롬프트만 토큰화해서 "정답 시작 위치"를 구한다
    assistant_prefix_encoded = tokenizer.apply_chat_template(
        messages[:-1],
        tokenize=True,
        add_generation_prompt=True, # "<|im_start|>assistant\n" 을 끝에 붙여 (A)의 앞부분과 형태를 일치시킨다
    )
    assistant_prefix_ids = _as_token_list(assistant_prefix_encoded)

    ## 프롬프트 구간은 -100으로 덮어 loss 계산에서 제외하고, assistant 응답 구간만 정답으로 남긴다
    assistant_start = min(len(assistant_prefix_ids), len(input_ids))
    labels = [-100] * assistant_start + input_ids[assistant_start:]

    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids), # 실제 토큰은 전부 1. 패딩(0)은 아래 collate_fn에서 붙는다
        "labels": labels,
    }


## 모델 학습시 데이터 전처리 진행하는 함수
def collate_fn(batch):
    new_batch = {
        "input_ids": [],
        "attention_mask": [],
        "labels": []
    }

    for example in batch:
        messages = example["messages"]

        tokenized = tokenize_with_assistant_labels(messages)
        new_batch["input_ids"].append(tokenized["input_ids"])
        new_batch["attention_mask"].append(tokenized["attention_mask"])
        new_batch["labels"].append(tokenized["labels"])

    # 패딩 처리
    ## 배치 안의 샘플들은 길이가 제각각인데, 텐서는 직사각형이어야 하므로 가장 긴 샘플에 맞춰 뒤를 채웁니다.
    ## 세 배열을 서로 다른 값으로 채우는 이유를 구분해서 알아두면 좋습니다.
    ##   input_ids      -> 패딩 토큰 id  : 자리를 채우기 위한 의미 없는 토큰
    ##   attention_mask -> 0            : "이 위치는 무시하라"는 표시 (어텐션 계산에서 제외)
    ##   labels         -> -100         : "이 위치는 채점하지 말라"는 표시 (loss 계산에서 제외)
    ##                                    -100은 PyTorch CrossEntropyLoss의 ignore_index 기본값입니다.
    ##
    ## !! [알아두면 좋은 점] tokenizer.pad_token_id가 None인 모델이 꽤 많습니다.
    ## 사전학습 전용 모델(base 모델)은 패딩을 쓸 일이 없어 pad_token이 정의되지 않은 경우가 많고,
    ## 그러면 아래 extend에 None이 들어가 torch.tensor()에서 TypeError가 납니다.
    ## 모델을 바꿀 때는 학습 전에 아래처럼 한 줄 방어해 두는 습관을 들이면 좋습니다.
    ##   if tokenizer.pad_token_id is None:
    ##       tokenizer.pad_token = tokenizer.eos_token
    ## (지금 쓰는 Konan-LLM-OND의 pad_token_id는 172728(<|endoftext|>)로 설정되어 있음이므로 이 스크립트에서는 문제가 없습니다)
    max_length = max(len(ids) for ids in new_batch["input_ids"])

    ## [메모리 대책] 패딩 길이를 PAD_TO_MULTIPLE_OF 의 배수로 올림
    ## 예: 1,088 토큰짜리 배치라면 1,152로 올려서 패딩합니다.
    ## 이렇게 하면 매 스텝 만들어지는 텐서 모양이 몇 종류로 고정되어, PyTorch가 이전 스텝에 쓰던
    ## 메모리 덩어리를 그대로 재활용할 수 있습니다. max_seq_length는 절대 넘지 않게 한 번 더 막습니다.
    if PAD_TO_MULTIPLE_OF:
        max_length = min(
            -(-max_length // PAD_TO_MULTIPLE_OF) * PAD_TO_MULTIPLE_OF,  # 올림 나눗셈
            max_seq_length,
        )

    ## [메모리 대책] 이번 배치를 계산하기 전에, 캐시가 너무 커졌으면 먼저 반납합니다.
    ## 여기(collate_fn)에 둔 이유는 "forward 가 큰 메모리를 요구하기 직전"이 청소하기 가장 좋은
    ## 시점이기 때문입니다. 낡은 덩어리를 반납해야 그 자리를 이번 배치가 쓸 수 있습니다.
    _free_gpu_cache_if_large()

    for i in range(len(new_batch["input_ids"])):
        pad_len = max_length - len(new_batch["input_ids"][i])
        new_batch["input_ids"][i].extend([tokenizer.pad_token_id] * pad_len) # input_ids에 패딩 토큰 추가
        new_batch["attention_mask"][i].extend([0] * pad_len) # attention_mask에 패딩 부분은 0으로 채움
        new_batch["labels"][i].extend([-100] * pad_len) # labels에 있는 모델 응답 뒤에 패딩 크기만큼 -100으로 채움

    for k in new_batch:
        new_batch[k] = torch.tensor(new_batch[k])

    return new_batch

## 전처리 샘플 테스트
example = train_dataset[0]
batch = collate_fn([example])

print("\n처리된 배치 데이터:")
print("입력 ID 형태:", batch["input_ids"].shape) # torch.Size([1, 2944])
print("어텐션 마스크 형태:", batch["attention_mask"].shape) # torch.Size([1, 2944])
print("레이블 형태:", batch["labels"].shape) # torch.Size([1, 2944])

# [128009, 10594, 198, 51176, 11203, 18243, 21362, 45144, 41950, ...
print('input_ids: ')
print(batch["input_ids"][0].tolist())

# input_ids 디코딩 결과:
# <|im_start|>system
# 당신은 검색 결과를 바탕으로 질문에 답변해야 합니다.
# 
# 다음의 지시사항을 따르십시오.
# 1. 질문과 검색 결과를 바탕으로 답변하십시오.
# 2. 검색 결과에 없는 내용을 답변하려고 하지 마십시오.
# 3. 질문에 대한 답이 검색 결과에 없다면 검색 결과에는 "해당 질문에 대한 내용이 없습니다."라고 답변하십시오.
# 4. 답변할 때 특정 문서를 참고하여 문장 또는 문단을 작성했다면 뒤에 출처는 이중 리스트로 해당 문서 번호를 남기십시오. 예를 들어서 특정 문장이나 문단을 1번 문서에서 인용했다면 뒤에 [[ref1]]이라고 기재하십시오.
# # 5. 예를 들어서 특정 문장이나 문단을 1번 문서와 5번 문서에서 동시에 인용했다면 뒤에 [[ref1]], [[ref5]]이라고 기재하십시오.
# 6. 최대한 다수의 문서를 인용하여 답변하십시오.

# 검색 결과:
# -----
# 문서1: 윤리, 종교에 딸린 기호 ...
# <|im_start|>user
# 비트겐슈타인의 철학적 사상에 영향을 준 인물들은 누구인가요?<|im_end|>
# <|im_start|>assistant
# <think>
# 
# </think>
# 
# 비트겐슈타인의 철학적 사상에 ...
# 사상에도 영향을 받았습니다 [[ref4]].<|im_end|>
decoded_text = tokenizer.decode(
    batch["input_ids"][0].tolist(),
    skip_special_tokens=False,
    clean_up_tokenization_spaces=False
)
print("\ninput_ids 디코딩 결과:")
print(decoded_text)

## 레이블에서 -100인 위치는 학습에서 제외된다. (BERT/practice/named_entity_recognition.py 참조)
# [-100, -100, -100, -100, ..., 13094, 122953, 109862, 13094, 63171, 21121, 116039, 65621, 116492, 80052, 3238, 92, 128009]
print('레이블에 대한 정수 인코딩 결과:')
print(batch["labels"][0].tolist())

label_ids = [token_id for token_id in batch["labels"][0].tolist() if token_id !=-100]
decoded_labels = tokenizer.decode(
    label_ids,
    skip_special_tokens=False,
    clean_up_tokenization_spaces=False
)

# labels 디코딩 결과 (-100 제외):
# 비트겐슈타인의 철학적 사상에 영향을 준 인물들은 다음과 같습니다:
# ... 사상에도 영향을 받았습니다 [[ref4]].<|im_end|>
print("\nlabels 디코딩 결과 (-100 제외):")
print(decoded_labels)

## 어텐션 마스크 확인

# 0번과 1번 데이터의 길이 확인
example0 = train_dataset[0]
example1 = train_dataset[1]

# 개별 길이 확인 (토큰화후)
tokenized0 = tokenizer.apply_chat_template(
    example0["messages"],
    tokenize=True,
    add_generation_prompt=False, # 이미 assistant 응답이 들어 있으므로 생성 프롬프트를 붙이지 않는다
    truncation=True
)
tokenized1 = tokenizer.apply_chat_template(
    example1["messages"],
    tokenize=True,
    add_generation_prompt=False, # 이미 assistant 응답이 들어 있으므로 생성 프롬프트를 붙이지 않는다
    truncation=True
)
print(f"0번 데이터 길이: {len(tokenized0['input_ids'])}") # 1930
print(f"1번 데이터 길이: {len(tokenized1['input_ids'])}") # 2601

batch = collate_fn([example0, example1])
print("\n배치 처리 후:")
print(f"입력 ID 형태: {batch['input_ids'].shape}") # torch.Size([2, 2688])
print(f"어텐션 마스크 형태: {batch['attention_mask'].shape}") # torch.Size([2, 2688])

# 길이가 짧은 샘플의 길이가 긴 샘플 or 긴 샘플과 패딩(메모리 고려)을 고려한 길이에 맞춰진다. (어텐션 마스크 0이 채워진다.)
max_length_in_batch = max(len(tokenized0['input_ids']), len(tokenized1['input_ids']))
print(f"\n배치내 최대 길이: {max_length_in_batch}") # 2601
print(f"0번 샘플 어텐션 마스크 1의 개수: {batch['attention_mask'][0].sum().item()}") # 1930
print(f"0번 샘플 어텐션 마스크 0의 개수: {(batch['attention_mask'][0] == 0).sum().item()}") # 758
print(f"1번 샘플 어텐션 마스크 1의 개수: {batch['attention_mask'][1].sum().item()}") # 2601
print(f"1번 샘플 어텐션 마스크 0의 개수: {(batch['attention_mask'][1] == 0).sum().item()}") # 87

print("\n=============================================")

######################################################################
# 모델 학습

trainer = SFTTrainer(
    model=model, # 아직 LoRA가 붙지 않은 맨 bfloat16 모델
    args=args, # SFTConfig
    train_dataset=train_dataset,
    data_collator=collate_fn,
    peft_config=peft_config
)

trainer.train() #  모델이 자동으로 output_dir에 저장
trainer.save_model() # 최종 모델(어댑터)을 저장

print("\n=============================================")

######################################################################
# 평가 준비 (테스트 데이터)

# prompt_lst : 시스템 + 유저 프롬프트 + 생성 프롬프트(모델이 이어서 쓸 시작점)  -> 모델 입력
# label_lst  : 정답 assistant 응답                                             -> 비교 기준

ASSISTANT_HEADER = "<|im_start|>assistant\n<think>\n\n</think>\n\n"
RESPONSE_END = "<|im_end|>" # kanana-2 의 턴 종료 토큰 (= tokenizer.eos_token, id 128010)

prompt_lst = []
label_lst = []

for messages in test_dataset["messages"]:
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

    ## assistant 헤더를 경계로 두 조각으로 나눈다 (assistant 턴이 1개뿐이므로 항상 정확히 2조각)
    before_assistant, after_assistant = text.split(ASSISTANT_HEADER)

    ## 입력: 시스템 + 유저 프롬프트 + 생성 프롬프트
    ##       split 하면 경계 문자열 자체는 사라지므로 다시 붙여줘야 한다
    prompt_lst.append(before_assistant + ASSISTANT_HEADER)

    ## 정답: 모델 응답 본문만 (뒤에 붙은 <|im_end|> 와 줄바꿈은 잘라낸다)
    label_lst.append(after_assistant.split(RESPONSE_END)[0])

######################################################################
# 추론 함수 정의
from transformers import pipeline

## 생성을 멈출 토큰 id
## kanana-2 는 tokenizer.eos_token 이 <|im_end|> 이고 generation_config 의 eos_token_id 도 128010 이라
## 사실 생략해도 멈추지만, 모델을 바꿨을 때 "끝없이 생성되는" 사고를 막기 위해 명시
eos_token = tokenizer(RESPONSE_END, add_special_tokens=False)["input_ids"][0] # 128010

## 추론 메서드 정의
def test_inference(pipe, prompt):
    outputs = pipe(
        prompt,
        max_new_tokens=8192,                 # 정답 응답이 길어도 잘리지 않을 만큼
        eos_token_id=eos_token,              # 이 토큰이 나오면 생성 중단
        pad_token_id=tokenizer.pad_token_id, # 지정 안 하면 경고가 뜬다 (배치 1이라 실제 패딩은 없음)
        do_sample=False,                     # 그리디 디코딩. 매번 같은 결과가 나와야 두 모델을 비교할 수 있다
        add_special_tokens=False,            # [중요] kanana2_finance_news_analysis.py의 추론 메서드 부분 참조
        return_full_text=False,              # 프롬프트를 뺀 "새로 생성된 텍스트"만 받는다
    )
    return outputs[0]["generated_text"].strip()

######################################################################
# 베이스 모델 vs 파인튜닝 모델 비교

## 먼저 학습에 쓴 모델을 메모리에서 내린다.
## LoRA가 붙은 학습용 모델이 그대로 남아 있으면, 아래에서 추론용 모델을 또 올리며 메모리를 두 배로 쓴다.
import gc

del trainer, model
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

print("\n=============================================")
print("베이스 모델 추론 (파인튜닝 전)")

## 학습에 쓴 것과 같은 설정으로 원본 모델을 다시 불러온다
base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    attn_implementation="sdpa",
    trust_remote_code=True,
)

pipe = pipeline("text-generation", model=base_model, tokenizer=tokenizer)

for prompt, label in zip(prompt_lst[10:15], label_lst[10:15]):
    print(f" response:\n{test_inference(pipe, prompt)}")
    print(f" label:\n{label}")
    print("-"*50)

print("\n=============================================")
print("파인튜닝 모델 추론 (LoRA 어댑터 부착)")

from peft import PeftModel

peft_model_id = "kanana2-1.3b-rag-ko"
fine_tuned_model = PeftModel.from_pretrained(base_model, peft_model_id)
pipe = pipeline("text-generation", model=fine_tuned_model, tokenizer=tokenizer)

for prompt, label in zip(prompt_lst[10:15], label_lst[10:15]):
    print(f" response:\n{test_inference(pipe, prompt)}")
    print(f" label:\n{label}")
    print("-"*50)