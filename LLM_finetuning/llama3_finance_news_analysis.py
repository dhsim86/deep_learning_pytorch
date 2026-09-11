# LLama 3를 이용한 금융 뉴스 분석 모델 파인튜닝

# 다양한 데이터셋을 쉽게 로드하고 처리
## load_dataset은 허깅페이스에 업로드된 데이터셋을 불러오거나 로컬 파일을 로드하는데 사용
## Dataset은 개별 데이터셋 객체를 다룰 때 사용
from datasets import load_dataset, Dataset

import torch

# AutoModelForCausalLM: 인과 언어 모델(Causal Language Model) 아키텍처를 자동으로 감지하고 불러오는 클래스
## - AutoModelForCausalLM: 추상화된 자동 팩토리 클래스로, 모든 인과적 언어 모델을 지원
##   - 입력된 모델 체크포인트 경로("gpt2", "meta-llama/Llama-2-7b-hf" 등)의 설정 파일(config.json)을 자동으로 분석
##   - 그 후 내부적으로 알맞은 모델 클래스(예: GPT-2인 경우 GPT2LMHeadModel)를 호출하여 인스턴스를 반환
## - GPT2LMHeadModel는 GPT-2 모델 한정으로 하드코딩된 모델
##   - 설정 파일 분석 없이 무조건 GPT-2 아키텍처를 기반으로 모델을 로드
##   - LLaMA나 Mistral 모델을 넣으면 에러가 발생
from transformers import AutoModelForCausalLM, AutoTokenizer

# LoRA(Low‑Rank Adaptation) 튜닝을 사용할 때 필요한 설정값을 정의
from peft import LoraConfig

# 모델을 학습할 때 필요한 다양한 설정값을 정의하는 도구
## 학습률, 배치크기, 옵티마이저 등의 설정
from trl import SFTConfig

# 실제 학습을 수행하는 클래스
## 모델, 데이터셋, 학습 설정을 한 번에 입력하여 효율적인 학습을 진행
from trl import SFTTrainer

# QLoRA 학습
from transformers import BitsAndBytesConfig
from peft import get_peft_model, prepare_model_for_kbit_training

print("\n=============================================")

######################################################################
# 데이터 전처리

## 1. 허깅페이스 허브에서 데이터셋 로드 (금융 뉴스 데이터셋)
dataset = load_dataset("iamjoon/finance_news_summarizer", split="train")

## 2. 전체 데이터 크기만 출력
print("전체 데이터 크기:", len(dataset)) # 991

# {
#   'system_prompt': '당신은 주어진 뉴스로부터 종목에 영향을 주는 뉴스인지 판별하는 금융 뉴스 판별기입니다.\n두 가지 답변 케이스가 존재하며 무조건 파이썬의 dictionary 형식으로 작성하십시오.\n큰 따옴표 사이에 다른 따옴표들을 적으려고 시도하지 마십시오. 이는 dictionary 파싱을 실패하게 하는 원인이 됩니다. 따라서 주의하십시오.\n아래 dictionary에서 각 value는 지시사항에 해당합니다. 지사사항을 따라 적지마십시오. 해당 지시사항에 따라 적절한 value를 채워넣으십시오.\n해당사항이 없다면 빈 문자열 또는 빈 리스트로 적어야 합니다. 임의로 \'없음\' 등을 적어서는 안 됩니다.\n\n만약 해당 뉴스가 특정 종목(회사)이 언급되지 않거나, 특정 종목(회사)와 아무런 연관이 없는 뉴스일 경우에는 아래와 같이 작성합니다.\n\n답변:\n{"is_stock_related": False,\n"summary": "여기에는 해당 뉴스를 요약해서 요약문을 작성하십시오"}\n\n만약 해당 뉴스가 특정 종목(회사)들과 연관되었거나, 특정 종목(회사)과 아무런 연관이 없는 뉴스일 경우에는 아래와 같이 작성합니다.\n\n답변:\n{"is_stock_related": True,\n"positive_impact_stocks": ["파이썬 문자열 리스트의 형태로 이 뉴스가 긍정적인 영향을 줄것으로 추정되는 종목들의 이름을 작성하십시오. 약자로 적거나 별명으로 적지마십시오. 종목명으로 추정되는 한글명을 적으십시오. 뉴스로부터 추정할 수 있는 정확한 풀네임으로 적으십시오. 만약, 존재하지 않는다면 빈 리스트로 작성하십시오."],\n"reason_for_positive_impact": "위의 종목들이 해당 뉴스로부터 긍정적인 영향을 받을 것으로 추정한 이유를 여기에다가 작성하십시오",\n"positive_keywords": ["긍정적인 영향을 줄 것으로 추정되는 종목들이 존재했다면 여기에 긍정적인 영향을 주는데 근거가 되었던 주요한 명사 키워드들을 파이썬 문자열 리스트 형태로 작성하십시오. 기술명, 회사명 등을 모두 포함합니다. 복합 명사 또한 허용합니다. 없다면 빈 리스트로 작성합시오."],\n"negative_impact_stocks": ["파이썬 문자열 리스트의 형태로 이 뉴스가 긍정적인 영향을 줄것으로 추정되는 종목들을 작성하십시오. 약자로 적거나 별명으로 적지마십시오. 종목명으로 추정되는 한글명을 적으십시오. 뉴스로부터 추정할 수 있는 정확한 풀네임으로 적으십시오. 만약, 존재하지 않는다면 빈 리스트로 작성하십시오."],\n"reason_for_negative_impact": "위의 종목들이 해당 뉴스로부터 긍정적인 영향을 받을 것으로 추정한 이유를 여기에다가 작성하십시오",\n"negative_keywords": ["부정적인 영향을 줄 것으로 추정되는 종목들이 존재했다면 여기에 부정적인 영향을 주는데 근거가 되었던 주요한 명사 키워드들을 파이썬 문자열 리스트 형태로 작성하십시오. 기술명, 회사명 등을 모두 포함합니다. 복합 명사 또한 허용합니다. 없다면 빈 리스트로 작성합시오."],\n"summary": "여기에는 해당 뉴스를 요약해서 요약문을 작성하십시오"}', 
#   'user_prompt': '추경호 중기 수출지원 총력 무역금융 40조 확대\n앵커 정부가 올해 하반기 우리 경제의 버팀목인 수출 확대를 위해 총력을 기울이기로 했습니다. 특히 수출 중소기업의 물류난 해소를 위해 무역금융 규모를 40조 원 이상 확대하고 물류비 지원과 임시선박 투입 등을 추진하기로 했습니다. 류환홍 기자가 보도합니다. 기자 수출은 최고의 실적을 보였지만 수입액이 급증하면서 올해 상반기 우리나라 무역수지는 역대 최악인 103억 달러 적자를 기록했습니다. 정부가 수출확대에 총력을 기울이기로 한 것은 원자재 가격 상승 등 대외 리스크가 가중되는 상황에서 수출 증가세 지속이야말로 한국경제의 회복을 위한 열쇠라고 본 것입니다. 추경호 경제부총리 겸 기획재정부 장관 정부는 우리 경제의 성장엔진인 수출이 높은 증가세를 지속할 수 있도록 총력을 다하겠습니다. 우선 물류 부담 증가 원자재 가격 상승 등 가중되고 있는 대외 리스크에 대해 적극 대응하겠습니다. 특히 중소기업과 중견기업 수출 지원을 위해 무역금융 규모를 연초 목표보다 40조 원 늘린 301조 원까지 확대하고 물류비 부담을 줄이기 위한 대책도 마련했습니다. 이창양 산업통상자원부 장관 국제 해상운임이 안정될 때까지 월 4척 이상의 임시선박을 지속 투입하는 한편 중소기업 전용 선복 적재 용량 도 현재보다 주당 50TEU 늘려 공급하겠습니다. 하반기에 우리 기업들의 수출 기회를 늘리기 위해 2 500여 개 수출기업을 대상으로 해외 전시회 참가를 지원하는 등 마케팅 지원도 벌이기로 했습니다. 정부는 또 이달 중으로 반도체를 비롯한 첨단 산업 육성 전략을 마련해 수출 증가세를 뒷받침하고 에너지 소비를 줄이기 위한 효율화 방안을 마련해 무역수지 개선에 나서기로 했습니다. YTN 류환홍입니다.', 
#   'assistant': {'is_stock_related': True, 'negative_impact_stocks': [], 'negative_keywords': [], 'positive_impact_stocks': ['현대상선', '대한통운', '한진', '삼성전자', 'LG전자'], 'positive_keywords': ['무역금융', '수출 지원', '임시선박', '물류비 지원', '첨단 산업 육성', '반도체'], 'reason_for_negative_impact': '', 'reason_for_positive_impact': '정부의 수출 지원 확대와 무역금융 규모 증가가 물류 및 전자 관련 기업들의 수출 및 운영에 긍정적인 영향을 미칠 것으로 예상되기 때문이다.', 'summary': '한국 정부가 수출 확대를 위해 무역금융을 40조 원 이상 확대하고 수출 중소기업의 물류비를 지원하기로 했습니다. 이는 수출 중심의 한국 경제 회복을 위한 대책이며, 반도체와 같은 첨단 산업 육성 전략도 포함됩니다.'}
# }
print("원본 데이터셋의 첫 번째 샘플 확인:", dataset[0])

## 훈련/테스트 데이터셋으로 분할
test_ratio = 0.2
train_data = []
test_data = []

data_indices = list(range(len(dataset)))
test_size = int(len(data_indices) * test_ratio)
test_data = data_indices[:test_size]
train_data = data_indices[test_size:]

## OpenAI 형식으로 데이터를 변환하는 함수 정의
def format_data(sample):
    return {
        "messages": [
            {
            "role": "system",
            "content": sample["system_prompt"],
            },
            {
            "role": "user",
            "content": sample["user_prompt"],
            },
            {
            "role": "assistant",
            "content": str(sample["assistant"])
            },
        ],
    }

## 분할된 데이터를 OpenAI format으로 변환
train_dataset = [format_data(dataset[i]) for i in train_data]
test_dataset = [format_data(dataset[i]) for i in test_data]

## 데이터셋 크기 출력
print(f"\n전 체 데 이 터 분 할 결 과: Train {len(train_dataset)}개 , Test {len(test_dataset)}개") # Train 793개, Test 198개

## 데이터셋 샘플 확인
# [
#   {'role': 'system', 'content': '당신은 주어진 뉴스로부터 종목에 영향을 주는 뉴스인지 판별하는 금융 뉴스 판별기입니다.\n두 가지 답변 케이스가 존재하며 무조건 파이썬의 dictionary 형식으로 작성하십시오.\n큰 따옴표 사이에 다른 따옴표들을 적으려고 시도하지 마십시오. 이는 dictionary 파싱을 실패하게 하는 원인이 됩니다. 따라서 주의하십시오.\n아래 dictionary에서 각 value는 지시사항에 해당합니다. 지사사항을 따라 적지마십시오. 해당 지시사항에 따라 적절한 value를 채워넣으십시오.\n해당사항이 없다면 빈 문자열 또는 빈 리스트로 적어야 합니다. 임의로 \'없음\' 등을 적어서는 안 됩니다.\n\n만약 해당 뉴스가 특정 종목(회사)이 언급되지 않거나, 특정 종목(회사)와 아무런 연관이 없는 뉴스일 경우에는 아래와 같이 작성합니다.\n\n답변:\n{"is_stock_related": False,\n"summary": "여기에는 해당 뉴스를 요약해서 요약문을 작성하십시오"}\n\n만약 해당 뉴스가 특정 종목(회사)들과 연관되었거나, 특정 종목(회사)과 아무런 연관이 없는 뉴스일 경우에는 아래와 같이 작성합니다.\n\n답변:\n{"is_stock_related": True,\n"positive_impact_stocks": ["파이썬 문자열 리스트의 형태로 이 뉴스가 긍정적인 영향을 줄것으로 추정되는 종목들의 이름을 작성하십시오. 약자로 적거나 별명으로 적지마십시오. 종목명으로 추정되는 한글명을 적으십시오. 뉴스로부터 추정할 수 있는 정확한 풀네임으로 적으십시오. 만약, 존재하지 않는다면 빈 리스트로 작성하십시오."],\n"reason_for_positive_impact": "위의 종목들이 해당 뉴스로부터 긍정적인 영향을 받을 것으로 추정한 이유를 여기에다가 작성하십시오",\n"positive_keywords": ["긍정적인 영향을 줄 것으로 추정되는 종목들이 존재했다면 여기에 긍정적인 영향을 주는데 근거가 되었던 주요한 명사 키워드들을 파이썬 문자열 리스트 형태로 작성하십시오. 기술명, 회사명 등을 모두 포함합니다. 복합 명사 또한 허용합니다. 없다면 빈 리스트로 작성합시오."],\n"negative_impact_stocks": ["파이썬 문자열 리스트의 형태로 이 뉴스가 긍정적인 영향을 줄것으로 추정되는 종목들을 작성하십시오. 약자로 적거나 별명으로 적지마십시오. 종목명으로 추정되는 한글명을 적으십시오. 뉴스로부터 추정할 수 있는 정확한 풀네임으로 적으십시오. 만약, 존재하지 않는다면 빈 리스트로 작성하십시오."],\n"reason_for_negative_impact": "위의 종목들이 해당 뉴스로부터 긍정적인 영향을 받을 것으로 추정한 이유를 여기에다가 작성하십시오",\n"negative_keywords": ["부정적인 영향을 줄 것으로 추정되는 종목들이 존재했다면 여기에 부정적인 영향을 주는데 근거가 되었던 주요한 명사 키워드들을 파이썬 문자열 리스트 형태로 작성하십시오. 기술명, 회사명 등을 모두 포함합니다. 복합 명사 또한 허용합니다. 없다면 빈 리스트로 작성합시오."],\n"summary": "여기에는 해당 뉴스를 요약해서 요약문을 작성하십시오"}'}, 
#   {'role': 'user', 'content': '은행권 유동성 규제 단계적 정상화…예금금리 더 오를까\n7월부터 LCR 규제 단계적 정상화…현금 보유량 늘려야 시장금리 고공행진에 은행채 발행 부담 느낀 은행권 수신 금리 인상 가능성 서울 시내의 시중은행 ATM기기의 모습. 2021.11.29 뉴스1 © News1 이재명 기자 서울 뉴스1 서상혁 기자 금융당국의 금융규제 유연화 조치 종료 방침에 따라 은행권 유동성 규제인 유동성커버리지비율 LCR 이 단계적으로 정상화될 예정이다. 은행으로선 현금 보유량을 점차 늘려야하는 상황인데 은행채 조달 비용이 상승하고 있다는 점에서 예·적금 금리를 더 올릴 것이란 관측이 나온다. 3일 금융권에 따르면 금융당국의 금융규제 유연화 조치 정상화 계획 에 따라 7월부터 유동성커버리지비율 LCR 규제가 단계적으로 정상화될 예정이다. LCR이란 향후 1개월간 순현금유출액에 대한 고유동성자산의 비율로 쉽게 말하면 은행이 보유해야 할 현금의 수준을 정해두는 규제다. 금융위기 같은 상황에서 뱅크런 처럼 일시적으로 은행에서 뭉칫돈이 이탈할 때를 대비하기 위해서다. 은행업감독규정에 따르면 은행들은 통합 원화 외화 LCR을 100% 이상으로 유지해야 한다. 금융당국은 코로나19 확산에 따라 지난 2020년 4월부터 은행권 통합 LCR 규제비율은 100%에서 85%로 낮췄다. 팬데믹 시기 영업난을 겪고 있는 중소기업과 자영업자에 적극적으로 자금을 공급하라는 취지에서다. 당국의 정상화 계획에 따라 은행들은 7월부터 오는 9월까진 통합 LCR을 90% 10 12월은 92.5%까지 높여야 한다. 내년 7월부터는 규제 수준인 100%를 맞춰야 한다. 주요 은행의 통합 LCR 수치는 양호한 수준이다. 5월말 기준 신한·우리·하나 등 주요 은행의 통합 LCR 비율은 91.5 93.22%로 7월부터 맞춰야 할 규제 비율인 90%를 상회했다. 다만 앞으로 점진적으로 현금 보유량을 늘려야 한다는 점에서 은행권의 수신금리 인상 압력은 커질 것으로 전망된다. 통상 은행들은 LCR 규제를 맞추기 위해 은행채를 발행하는데 시장금리 상승으로 채권 발행 비용이 커진 만큼 정기예금 등으로 우회할 가능성이 높아졌기 때문이다. 시중은행 관계자는 시장금리가 빠르게 오르고 있어 현재로선 채권을 발행하는 것보다 예금으로 현금을 끌어모으는 게 비용 측면에서 더 유리하다는 판단 이라며 은행채의 경우 발행 시 수수료 명목으로 추가 비용이 나가는 반면 예금의 경우 고객을 유치하는 효과가 있다 고 설명했다. 또 다른 관계자는 한국은행의 기준금리 인상에 대응해 수신금리를 인상하는 것만으로도 LCR을 맞출 수 있을 것 이라면서도 여의치 않으면 정기예금 특판에 나설수도 있다 고 말했다. 금융투자협에 따르면 은행채 1년물 금리는 올 1월 3일 연 1.719%에서 지난 1일 연 3.583%로 올랐다. 미국 연방준비제도 Fed 가 인플레이션에 대응하기 위해 기준금리를 0.75%p 인상하는 자이언트스텝 을 단행하는 등 주요국이 긴축에 나서면서 시장금리가 급등한 영향이다. 금융감독원 금융상품통합비교공시에 따르면 1일 기준 국내 은행 정기예금 1년 기준 최고금리는 연 3.49%다. 한편 은행권 예대율 규제 완화 조치도 7월부로 해제된다. 그간 금융당국은 예대율 규제 비율인 100%에서 5%p 이내 위반에 대해선 제재를 하지 않았다. 예대율이란 은행의 예금잔액에 대한 대출잔액의 비율로 100%를 넘겨선 안 된다. 다만 은행권 가계대출이 6개월 연속으로 줄어들고 있어 규제 비율을 맞추는 데엔 무리가 없을 것으로 보인다. 올 1분기말 기준 4대 은행의 예대율은 96.7 98.8%로 규제 수준을 맞췄다.'}, 
#   {'role': 'assistant', 'content': "{'is_stock_related': True, 'negative_impact_stocks': [], 'negative_keywords': [], 'positive_impact_stocks': ['신한은행', '하나은행', '우리은행'], 'positive_keywords': ['유동성커버리지비율', '예금금리', '현금 보유량'], 'reason_for_negative_impact': '', 'reason_for_positive_impact': 'LCR 규제의 단계적 정상화로 인해 은행들은 현금 보유량을 늘려야 하며, 이로 인해 예금금리 인상 가능성이 증가한다. 이는 고객 유치가 용이해지면서 은행의 수익성 향상에 긍정적인 영향을 미칠 수 있다.', 'summary': '7월부터 단계적으로 정상화되는 LCR 규제에 따라 은행권은 현금 보유량을 늘려야 하며, 이로 인해 수신금리 인상 가능성이 높아진다. 주요 은행들의 통합 LCR 비율은 이미 규제 비율을 상회하고 있지만, 시장금리 상승으로 인해 채권 발행 비용이 증가하여 은행들은 예금으로 현금을 확보하려는 경향을 보일 수 있다.'}"}
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

######################################################################
# 챗 템플릿 적용 테스트
## LLM은 사전학습시 특정 챗 템플릿 형식에 맞추어 학습된 상태이므로, 파인 튜닝할 때도 이를 지켜야 한다.
## 토크나이저의 apply_chat_template 메서드로, OpenAI 형식으로 가공된 데이터를 특정 모델의 챗 템플릿으로 변환 가능

## 사용할 허깅페이스의 모델 ID
model_id = "NCSOFT/Llama-VARCO-8B-Instruct" # Meta-Llama-3.1-8B 모델을 한국어 성능에 특화되도록 추가학습된 모델

tokenizer = AutoTokenizer.from_pretrained(model_id)

## LLaMa를 위한 챗 템플릿 적용
## LLaMa의 챗 템플릿 형식
# <|begin_of_text|>
# <|start_header_id|>system<|end_header_id|>시스템 프롬프트<|eot_id|>
# <|start_header_id|>user<|end_header_id|>유저 프롬프트<|eot_id|>
# <|start_header_id|>assistant<|end_header_id|>거대 언어 모델이 해야하는 답변<|eot_id|>

## 인코딩 OFF / 생성 프롬프트 OFF
text = tokenizer.apply_chat_template(train_dataset[0]["messages"], tokenize=False, add_generation_prompt=False)
# <|begin_of_text|>
# <|start_header_id|>system<|end_header_id|>
#
# 당신은 주어진 뉴스로부터 종목에 영향을 주는 뉴스인지 판별하는 금융 뉴스 판별기입니다.
# 두 가지 답변 케이스가 존재하며 무조건 파이썬의 dictionary 형식으로 작성하십시오.
# 큰 따옴표 사이에 다른 따옴표들을 적으려고 시도하지 마십시오. 이는 dictionary 파싱을 실패하게 하는 원인이 됩니다. 따라서 주의하십시오.
# 아래 dictionary에서 각 value는 지시사항에 해당합니다. 지사사항을 따라 적지마십시오. 해당 지시사항에 따라 적절한 value를 채워넣으십시오.
# 해당사항이 없다면 빈 문자열 또는 빈 리스트로 적어야 합니다. 임의로 '없음' 등을 적어서는 안 됩니다.
#
# 만약 해당 뉴스가 특정 종목(회사)이 언급되지 않거나, 특정 종목(회사)와 아무런 연관이 없는 뉴스일 경우에는 아래와 같이 작성합니다.
#
# 답변:
# {"is_stock_related": False,
# "summary": "여기에는 해당 뉴스를 요약해서 요약문을 작성하십시오"}
# 
# 만약 해당 뉴스가 특정 종목(회사)들과 연관되었거나, 특정 종목(회사)과 아무런 연관이 없는 뉴스일 경우에는 아래와 같이 작성합니다.
# 
# 답변:
# {"is_stock_related": True,
# "positive_impact_stocks": ["파이썬 문자열 리스트의 형태로 이 뉴스가 긍정적인 영향을 줄것으로 추정되는 종목들의 이름을 작성하십시오. 약자로 적거나 별명으로 적지마십시오. 종목명으로 추정되는 한글명을 적으십시오. 뉴스로부터 추정할 수 있는 정확한 풀네임으로 적으십시오. 만약, 존재하지 않는다면 빈 리스트로 작성하십시오."],
# "reason_for_positive_impact": "위의 종목들이 해당 뉴스로부터 긍정적인 영향을 받을 것으로 추정한 이유를 여기에다가 작성하십시오",
# "positive_keywords": ["긍정적인 영향을 줄 것으로 추정되는 종목들이 존재했다면 여기에 긍정적인 영향을 주는데 근거가 되었던 주요한 명사 키워드들을 파이썬 문자열 리스트 형태로 작성하십시오. 기술명, 회사명 등을 모두 포함합니다. 복합 명사 또한 허용합니다. 없다면 빈 리스트로 작성합시오."],
# "negative_impact_stocks": ["파이썬 문자열 리스트의 형태로 이 뉴스가 긍정적인 영향을 줄것으로 추정되는 종목들을 작성하십시오. 약자로 적거나 별명으로 적지마십시오. 종목명으로 추정되는 한글명을 적으십시오. 뉴스로부터 추정할 수 있는 정확한 풀네임으로 적으십시오. 만약, 존재하지 않는다면 빈 리스트로 작성하십시오."],
# "reason_for_negative_impact": "위의 종목들이 해당 뉴스로부터 긍정적인 영향을 받을 것으로 추정한 이유를 여기에다가 작성하십시오",
# "negative_keywords": ["부정적인 영향을 줄 것으로 추정되는 종목들이 존재했다면 여기에 부정적인 영향을 주는데 근거가 되었던 주요한 명사 키워드들을 파이썬 문자열 리스트 형태로 작성하십시오. 기술명, 회사명 등을 모두 포함합니다. 복합 명사 또한 허용합니다. 없다면 빈 리스트로 작성합시오."],
# "summary": "여기에는 해당 뉴스를 요약해서 요약문을 작성하십시오"}
# <|eot_id|>
# <|start_header_id|>user<|end_header_id|>
#
# CEO포커스 임기 초년 나희승 코레일 사장 꼴찌 성적표 받자마자 열차 사고까지
# 나희승 한국철도공사 사장. 사진 한국철도공사 제공 지난해 공공기관 경영평가에서 한국철도공사 코레일 가 36개 공기업 가운데 유일하게 E등급 을 맞으며 꼴찌라는 불명예를 얻게 됐다. 이번 경영평가에서 국토교통부 산하 공공기관 가운데 성적이 낮은 곳은 E등급인 코레일뿐만이 아니었다. 코레일이 최하 점수를 받은 이유로는 지속적으로 발생한 안전사고가 지목된다. 코레일은 재난·안전관리 분야에서 최하등급을 받았다. 특히 나희승 56·사진 코레일 사장에게 이번 평가는 더욱 뼈아프게 다가올 수밖에 없다. 올 1월에 발생한 부산행 KTX 탈선 사고는 지난해 11월 취임한 나 사장의 임기 중에 발생했다. 나 사장이 책임에서 자유롭지 못하다는 뜻이다. 이런 상황에 지난 1일 수서고속철도 SRT 탈선 사고가 또 발생해 코레일의 안전관리 체계 자체에 문제가 있는 것이 아니냐는 우려가 나온다. 원희룡 국토교통부 장관은 코레일의 안전관리 체계의 근본적인 점검을 지시한 상태다. 나 사장 전임이던 손병석 전 사장의 경우 2020년 경영평가에서 경영관리 부문 E등급을 받자 스스로 자리를 내려놨다. 당시 코레일 전체 등급은 C등급이었음에도 손 사장은 책임을 지고 물러났다. 공공기관 경영평가 등급은 성과급 기준이 돼 당시 내부에서 불만의 목소리가 나오면서 압박도 커진 것으로 알려졌다. 일각에서는 나 사장이 친야권 인사로 분류돼 새 정부의 평가에 영향을 준 것이 아니냐는 의구심도 제기되는 상황이다. 이전 정부에서 여당이던 더불어민주당은 철도 핵심정책인 남북철도 등 업무를 수행할 적임자로 나 사장을 지목한 바 있다. 나 사장은 철도 연구자로 잘 알려졌다. 나 사장은 2019년부터 민주평화통일자문회의 경제협력분과위원회 상임위원을 맡는 등 민주당 측과 가까운 인사로 분류되는 게 사실이다. 임기가 2024년 11월까지 2년 이상 남은 나 사장이 이번 난관을 어떻게 극복할지 관심이 집중된다.
# <|eot_id|>
# <|start_header_id|>assistant<|end_header_id|>
# 
# {'is_stock_related': False, 'negative_impact_stocks': None, 'negative_keywords': None, 'positive_impact_stocks': None, 'positive_keywords': None, 'reason_for_negative_impact': None, 'reason_for_positive_impact': None, 'summary': '나희승 코레일 사장이 임기 초반에 안전사고와 관련하여 코레일이 공공기관 경영평가에서 최하등급을 받았다는 뉴스입니다. 나 사장의 안전 관리 체계에 대한 책임론이 대두되고 있으며, 이는 그의 정치적 배경과도 관련이 있다는 분석이 제기되고 있는 상황입니다.'}
# <|eot_id|>
# <|start_header_id|>assistant<|end_header_id|>
print(text)

print("\n=============================================")

######################################################################
# 모델 학습 준비

## LoRA 튜닝 설정
##
## [LoRA를 처음 보는 사람을 위한 30초 요약]
## 파인튜닝의 원래 방식은 모델의 모든 가중치 W를 직접 업데이트하는 것입니다(전체 파인튜닝).
## 하지만 8B 모델이면 W만 16GB이고, 옵티마이저 상태까지 더하면 80GB 이상이 필요해 개인 GPU로는 불가능합니다.
##
## LoRA는 "W는 그대로 얼려두고, 변화량 dW만 따로 학습하자"는 아이디어입니다.
## 여기에 "dW는 사실 그렇게 복잡한 정보가 아닐 것"이라는 가정을 더해, dW를 두 개의 얇은 행렬 곱으로 쪼갭니다.
##
##     원래     : y = W x                     (W는 예: 4096 x 4096 = 1,600만개)
##     LoRA 적용: y = W x + (alpha/r) * B(A x)  (A는 8 x 4096, B는 4096 x 8 => 합쳐서 6.5만개)
##
## 학습 대상이 1,600만개에서 6.5만개로 줄었지만(약 0.4%), 실무에서 전체 파인튜닝에 근접한 성능이 나옵니다.
## 그리고 학습이 끝나면 B*A를 W에 더해버릴 수 있어서(merge), 추론 속도는 원본과 완전히 동일합니다.
peft_config = LoraConfig(
    lora_alpha=32,      # LoRA의 alpha, 스케일링 계수 설정. LoRA 가중치의 모델 출력 영향도를 조정
                        ## 위 수식의 (alpha/r)이 곧 LoRA의 반영 강도입니다. 여기서는 32/8 = 4배로 증폭됩니다.
                        ## 관례적으로 alpha = 2*r 또는 4*r을 씁니다. r을 올릴 때 alpha도 같은 비율로 올려주면
                        ## 반영 강도가 유지되므로 학습률을 다시 튜닝하지 않아도 됩니다.
    lora_dropout=0.1,   # LoRA 적용시 드롭아웃 비율 설정. 학습 동안 10%의 뉴런을 랜덤하게 비활성화하여 과적합 방지
                        ## 이 드롭아웃은 원본 W가 아니라 LoRA 경로(A x)에만 적용됩니다.
                        ## 데이터가 793개로 적어 과적합 위험이 있는 상황이라 켜 두는 것이 좋습니다.
    r=8,                # LoRA 랭크, LoRA가 학습할 저차원 공간의 크기를 설정
                        ## 위 수식에서 A, B의 얇은 쪽 차원입니다. r이 클수록 표현력(=배울 수 있는 양)이 커지지만
                        ## 학습 파라미터와 메모리도 비례해 늘고 과적합 위험도 커집니다.
                        ## 8~16은 "말투/출력 형식을 익히는" 정도의 작업에 적합한 출발점이고,
                        ## 새로운 지식이나 복잡한 추론을 가르치려면 32~64를 검토합니다.
    bias="none",        # LoRA 적용시 편향 설정. none이면 편향이 LoRA에 의해 조정되지 않음. ["none", "all", "lora_only"]
    target_modules=["q_proj", "v_proj"], # LoRA를 적용할 레이어, 여기서는 Self Attention의 W^q, W^v 에 적용
                        ## [알아두면 좋은 점] q_proj/v_proj만 고르는 것은 원조 LoRA 논문의 최소 구성입니다.
                        ## 반면 QLoRA 논문은 "4비트 양자화로 잃은 정확도를 보상하려면 LoRA를 넓게 깔아야 한다"며
                        ## 모든 선형 레이어(q,k,v,o + gate,up,down)에 적용할 것을 권장합니다.
                        ## PEFT에서는 target_modules="all-linear" 한 줄로 그렇게 지정할 수 있습니다.
                        ## 이 스크립트는 두 분기(LoRA/QLoRA)가 같은 설정을 공유하도록 최소 구성을 유지했지만,
                        ## QLoRA로 실제 품질을 내야 한다면 "all-linear"로 바꿔 비교해 보는 것을 권합니다.
                        ## (그만큼 학습 파라미터와 메모리는 늘어납니다)
    task_type="CAUSAL_LM",  # LoRA가 적용되는 작업의 유형. CAUSAL_LM은 시퀀스 생성 작업 (Causal Language Modeling)
                        ## 이 값에 따라 PEFT가 모델을 감쌀 래퍼 클래스를 고릅니다(여기서는 PeftModelForCausalLM).
                        ## 분류 작업이면 "SEQ_CLS"처럼 다른 값을 넣어야 하며, 잘못 넣으면 학습은 되지만
                        ## 저장/불러오기 시점에 헤드가 맞지 않아 문제가 생깁니다.
)

# ==========================================================================================
# [QLoRA vs 일반 LoRA] 아래 if/else가 갈리는 이유
#
# QLoRA(Quantized LoRA)를 한 문장으로 요약하면
#   "원본 모델은 4비트로 압축해서 통째로 얼려두고, 그 위에 덧붙인 작은 LoRA 행렬만 학습한다"
# 입니다. Llama-VARCO-8B (80.3억 파라미터) 기준으로 모델을 GPU에 올리는 데 드는 메모리는 대략 이렇게 줄어듭니다.
#
#   bf16(16비트)으로 그냥 로드   : 8.03B x 2바이트   = 약 16.1GB   <- 24GB GPU에서도 학습까지는 빡빡함
#   4bit로 양자화해서 로드      : 압축 대상 약 7B x 0.5바이트 + 임베딩/lm_head 2.1GB = 약 5.6GB
#
# 그런데 이 4비트 양자화를 실제로 수행하는 bitsandbytes 라이브러리가 CUDA(NVIDIA) 전용입니다.
# 애플 실리콘의 MPS에서는 사용할 수 없으므로, 맥북에서는 양자화를 포기하고
# 원본을 bfloat16으로 그대로 올린 뒤 LoRA만 적용합니다. (= 일반 LoRA)
#
# 그래서 두 분기의 차이는 "양자화를 하느냐"만이 아니라, 그 결과로
#   - QLoRA 분기 : prepare_model_for_kbit_training + get_peft_model 을 내가 직접 호출한다
#   - LoRA 분기  : 아무것도 하지 않고, LoRA 적용을 맨 아래 SFTTrainer에게 맡긴다
# 로 "LoRA를 누가 붙이는가"까지 달라진다는 점입니다. (자세한 이유는 맨 아래 SFTTrainer 부분 주석 참고)
# ==========================================================================================
if torch.cuda.is_available():
    ## ------------------------------------------------------------------
    ## [1] QLoRA 양자화 설정
    ## ------------------------------------------------------------------
    ## BitsAndBytesConfig = "모델 가중치를 어떤 방식으로 압축해서 불러올지" 적어두는 설정 객체.
    ## 이 객체 자체는 아무 일도 하지 않고, 아래 from_pretrained에 넘겨지는 순간 실제로 적용됩니다.
    bnb_config = BitsAndBytesConfig(
        # [4비트 로드] 모델 가중치를 4비트로 압축해서 불러온다 (16비트 대비 1/4 크기)
        load_in_4bit=True,

        # [이중 양자화] 양자화 과정에서 생기는 부가 정보(quantization constant)까지 한 번 더 압축
        ## 4비트 양자화는 가중치를 64개씩 블록으로 묶고, 블록마다 "스케일 상수"를 따로 저장합니다.
        ## 파라미터가 수십억 개면 이 상수들도 무시할 수 없는 용량이 되는데, 그걸 또 압축해서
        ## 파라미터당 약 0.4비트를 추가로 절약합니다. 정확도 손실은 거의 없어서 QLoRA 기본 권장값입니다.
        bnb_4bit_use_double_quant=True,

        # [양자화 자료형] "nf4" = 4-bit NormalFloat, QLoRA 논문이 제안한 4비트 표현 방식
        ## 4비트로는 숫자를 16종류밖에 표현할 수 없습니다. 그 16칸을 어디에 배치할지가 핵심인데,
        ## nf4는 "신경망 가중치는 0을 중심으로 정규분포를 이룬다"는 성질을 이용해 값이 몰려 있는
        ## 0 근처에 칸을 촘촘하게 배치합니다. 균등하게 나누는 "fp4"보다 정확도 손실이 적습니다.
        bnb_4bit_quant_type="nf4",

        # [계산 자료형] 저장은 4비트지만, 실제 행렬 곱셈을 할 때는 이 자료형으로 되돌려(역양자화) 계산
        ## 즉 QLoRA는 "저장 = 4비트 / 계산 = bfloat16"의 이중 구조이며 4비트로 직접 계산하지 않습니다.
        ## 그래서 메모리는 크게 아끼지만, 매번 압축을 푸는 비용 때문에 속도는 오히려 조금 느려집니다.
        ## (bfloat16은 NVIDIA Ampere 세대(A100, RTX 30xx) 이상에서만 지원. 구형 GPU라면 torch.float16)
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    ## ------------------------------------------------------------------
    ## [2] 모델 로드 - quantization_config를 넘기면 "불러오는 순간" 4비트로 압축된다
    ## ------------------------------------------------------------------
    ## [quantization_config가 하는 일]
    ## from_pretrained가 체크포인트 파일을 읽어 들이면서, 모델 안의 nn.Linear 레이어들을
    ## bitsandbytes의 bnb.nn.Linear4bit 레이어로 교체하고 가중치를 4비트로 변환해 GPU에 올립니다.
    ##
    ## 핵심은 "16비트로 전부 불러온 뒤에 압축"이 아니라 "레이어 단위로 읽으면서 바로 압축"이라는 점입니다.
    ## 그래서 bf16으로는 못 올라가는 크기의 모델도 작은 GPU에 올릴 수 있습니다.
    ##
    ## 참고 1) 모든 레이어가 압축되는 건 아닙니다. 임베딩(nn.Embedding), LayerNorm, lm_head는
    ##         정확도에 민감해서 양자화 대상에서 제외되고 16비트로 남습니다.
    ## 참고 2) device_map을 따로 주지 않아도 됩니다. 4비트 모델은 CPU에 올릴 수 없기 때문에
    ##         transformers가 자동으로 device_map={"": 현재 GPU}로 채워 줍니다.
    ##         (transformers/quantizers/quantizer_bnb_4bit.py 의 update_device_map)
    ## 참고 3) 여기서 지정하는 dtype은 "양자화되지 않고 남는 레이어들"의 자료형입니다.
    ##         torch_dtype은 transformers 5.x에서 deprecated되어 dtype으로 이름이 바뀌었습니다.
    ##         (현재는 "`torch_dtype` is deprecated! Use `dtype` instead!" 경고만 뜨고 동작합니다)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, quantization_config=bnb_config)

    ## ------------------------------------------------------------------
    ## [3] 4비트 모델을 "학습이 가능한 상태"로 손질
    ## ------------------------------------------------------------------
    ## [prepare_model_for_kbit_training이 하는 일]  (k-bit = 4비트/8비트 같은 저비트를 뜻함)
    ## 4비트로 압축된 모델을 그대로 학습시키면, 학습이 아예 안 되거나 매우 불안정합니다.
    ## 이 함수가 그 문제들을 한 번에 정리해 줍니다. 실제로는 아래 4가지 작업을 합니다.
    ## (peft/utils/other.py 의 prepare_model_for_kbit_training 참고)
    ##
    ##   (1) 원본 파라미터 전체를 얼린다 (모든 param.requires_grad = False)
    ##       -> QLoRA의 전제가 "원본은 절대 건드리지 않는다"입니다. 4비트로 압축된 가중치는
    ##          미세한 그래디언트를 더해도 반올림에 묻혀 사라지므로 애초에 학습이 불가능합니다.
    ##
    ##   (2) 양자화되지 않고 남은 레이어(LayerNorm, 임베딩, lm_head)를 float32로 올린다
    ##       -> LayerNorm은 분산을 구하고 나눗셈을 하기 때문에 16비트에서는 값이 튀기 쉽습니다.
    ##          이 부분만 32비트로 계산하면 loss가 훨씬 안정적으로 수렴합니다.
    ##       -> 대가: 메모리를 약 2GB (임베딩 525M + lm_head 525M 이 16비트 -> 32비트) 더 씁니다. 안정성과의 교환입니다.
    ##
    ##   (3) 입력 임베딩의 출력에 requires_grad=True를 걸어준다 (enable_input_require_grads)
    ##       -> 초보자가 가장 이해하기 어렵지만, 빠뜨리면 "학습이 아예 안 되는" 핵심 처리입니다.
    ##          gradient checkpointing(중간 계산값을 버리고 역전파 때 재계산해 메모리를 아끼는 기법)은
    ##          "그래디언트가 필요한 입력"이 들어와야 그 구간을 역전파 대상으로 인식합니다.
    ##          그런데 QLoRA는 원본이 전부 얼려져 있어 임베딩 출력에 그래디언트 표시가 없고,
    ##          그러면 체크포인팅 구간이 통째로 건너뛰어져 LoRA 가중치까지 그래디언트가 도달하지 못합니다.
    ##          증상: loss가 전혀 줄지 않거나
    ##               "element 0 of tensors does not require grad and does not have a grad_fn" 에러
    ##
    ##   (4) gradient checkpointing을 켠다 (model.gradient_checkpointing_enable())
    ##       -> 아래 SFTConfig(gradient_checkpointing=True)와 중복이지만 두 번 켜도 문제는 없습니다.
    ##
    ## !! 순서 주의: 반드시 get_peft_model보다 "먼저" 호출해야 합니다.
    ##    순서를 바꾸면 위 (1)이 방금 붙인 LoRA 가중치까지 얼려버려서 학습 대상 파라미터가 0개가 됩니다.
    ##    (0개여도 에러 없이 학습이 "돌아가는 것처럼" 보이므로 알아채기 어렵습니다)
    model = prepare_model_for_kbit_training(model)

    ## [get_peft_model이 하는 일]
    ## 얼려진 원본 모델에 LoRA 어댑터를 덧붙여, PeftModel로 감싼 새 모델을 반환합니다.
    ## 위에서 만든 peft_config(LoraConfig)의 target_modules에 적힌 레이어를 찾아
    ## 원래 연산   y = W x   를
    ## 다음과 같이 y = W x + (lora_alpha / r) * B(A x)   로 바꿔치기합니다.
    ##   - W    : 4비트로 압축되어 얼려진 원본 가중치 (학습하지 않음)
    ##   - A, B : 새로 추가된 작고 얇은 행렬 (오직 이것만 학습함)
    ##
    ## 주의: 이 함수는 model을 제자리에서 고치는 게 아니라 "감싼 새 객체"를 돌려주므로
    ##       반드시 model = get_peft_model(...) 처럼 반환값을 다시 받아야 합니다.
    ##
    ## 학습 대상이 제대로 잡혔는지는 아래 한 줄로 꼭 확인해 보세요. 초보자가 가장 자주 틀리는 지점입니다.
    ##   model.print_trainable_parameters()
    ##   -> trainable params: 3,407,872 || all params: 8,033,669,120 || trainable%: 0.0424
    ##   -> 여기서 trainable params가 0이면 위 "순서 주의"를 어긴 것입니다.
    model = get_peft_model(model, peft_config)
else:
    ## ------------------------------------------------------------------
    ## [맥북(MPS) / CPU] 일반 LoRA 경로 - 양자화 없이 bfloat16 원본 + LoRA
    ## ------------------------------------------------------------------
    ## 맥북(mps)는 bitsandbytes를 지원하지 않음 (CUDA 전용 라이브러리)
    ##
    ## 여기서는 get_peft_model을 호출하지 않고 "LoRA가 아직 붙지 않은 맨 모델"만 만들어 둡니다.
    ## LoRA를 붙이는 일은 맨 아래 SFTTrainer(peft_config=peft_config)가 내부에서 대신 해 줍니다.
    ## 양자화를 하지 않았으므로 prepare_model_for_kbit_training도 필요 없습니다.
    ## (일반 LoRA는 원본이 16비트라 반올림에 그래디언트가 묻히는 문제가 없고,
    ##  LoRA + gradient checkpointing 조합에 필요한 enable_input_require_grads()는
    ##  TRL의 SFTTrainer가 알아서 호출해 줍니다)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16) 

## 데이터의 최대 길이 제한
max_seq_length=8192

## 파인튜닝 설정
## SFTConfig = SFT(Supervised Fine-Tuning, 지도 파인튜닝) 학습에 쓰이는 "설정값 모음집"
## - 허깅페이스 transformers의 TrainingArguments를 상속받은 클래스라서, 일반 학습 설정(학습률, 배치 크기 등)을
##   전부 그대로 쓸 수 있고 여기에 SFT 전용 옵션(max_length, packing, dataset_kwargs 등)이 추가되어 있음
## - 이 객체 자체는 "학습을 시키는 주체"가 아니라 그냥 설정 덩어리이고, 실제 학습은 SFTTrainer가 이 설정을 읽어서 수행함
##
## [먼저 알아두면 좋은 용어]
## - step(스텝)  : 모델 파라미터를 1번 업데이트하는 단위
## - epoch(에포크): 훈련 데이터 전체를 처음부터 끝까지 1번 다 훑는 단위
## - 이 스크립트 기준 계산: 훈련 데이터 793개, 실질 배치 4개(아래 설명) -> 1 에포크 ≈ 198 step, 3 에포크 ≈ 595 step
args = SFTConfig(
    # [저장 위치] 학습 결과물(LoRA 어댑터 가중치, 설정 파일, 학습 로그)이 저장될 로컬 디렉토리 이름
    ## 왜 필요? 학습이 중간에 끊기거나 서버가 죽어도 여기 저장된 체크포인트부터 이어서 학습/추론할 수 있음
    ## 참고: push_to_hub=True로 바꾸면 이 이름이 허깅페이스 허브의 저장소 ID로도 그대로 쓰임 (지금은 False라 로컬 저장만 됨)
    output_dir="llama3-8b-summarizer-ko",

    # [학습량] 훈련 데이터 전체를 몇 바퀴 반복해서 학습할지
    ## 왜 필요? 데이터를 한 번만 보면 모델이 패턴을 충분히 못 익힘(과소적합). 반대로 너무 많이 돌리면
    ##          훈련 데이터를 통째로 외워버려서 처음 보는 뉴스에는 오히려 성능이 떨어짐(과적합)
    ## 데이터가 793개로 적은 편이므로 2~3 정도가 무난한 출발점 (SFTConfig 기본값도 3)
    num_train_epochs=3,

    # [배치 크기] GPU 1장이 "한 번에" 동시에 처리하는 학습 샘플 개수
    ## 왜 필요? GPU 메모리(VRAM) 사용량을 결정하는 가장 큰 요인. 값이 크면 학습이 안정적이고 빠르지만 메모리가 터짐(OOM)
    ##          8B 모델 + 긴 뉴스 텍스트 조합이라 2 정도로 아주 작게 잡은 것 (OOM이 나면 1로 줄이면 됨)
    per_device_train_batch_size=1, # GPU 메모리 사용량이 너무 커서 1로 줄임

    # [그래디언트 누적] 파라미터를 바로 업데이트하지 않고, 몇 번의 미니배치 계산 결과를 모았다가 한 번에 업데이트할지
    ## 동작: 배치 2개를 계산 -> 그래디언트를 메모리에 누적 -> 또 배치 2개 계산해서 누적 -> 그제서야 파라미터 1회 업데이트
    ## 왜 필요? "작은 배치로 여러 번 쪼개서 계산 = 큰 배치 한 번"과 수학적으로 거의 같은 효과를 내면서 메모리는 아낄 수 있음
    ## => 실질 배치 크기(effective batch size) = per_device_train_batch_size(2) x gradient_accumulation_steps(2) x GPU 수(1) = 4
    gradient_accumulation_steps=4,

    # [메모리 절약 기법] 순전파(forward) 때 계산한 중간 결과값을 전부 저장하지 않고, 역전파(backward) 때 필요한 부분을 다시 계산
    ## 왜 필요? 딥러닝은 역전파에 쓰려고 중간 계산값을 전부 메모리에 들고 있는데, 8B급 모델에서는 이게 VRAM을 엄청나게 잡아먹음
    ##          이 옵션을 켜면 메모리를 크게(수십 %) 아끼는 대신 재계산 때문에 학습 속도가 대략 20~30% 느려짐
    ## 큰 모델을 소비자용/단일 GPU에서 학습할 때는 사실상 필수 옵션 (SFTConfig 기본값도 True)
    gradient_checkpointing=True,

    # [옵티마이저] 계산된 그래디언트를 가지고 실제로 파라미터를 어떻게 갱신할지 결정하는 알고리즘
    ## AdamW = 트랜스포머 학습의 사실상 표준. 파라미터마다 학습 속도를 자동 조절(Adam) + 가중치 감쇠를 분리 적용(W)
    ## "fused"는 여러 연산을 하나의 GPU 커널로 합쳐 실행하는 최적화 버전이라 같은 결과를 더 빠르게 계산함
    ## 참고: 이 맥북(M4 Pro / torch 2.12)의 MPS에서 fused AdamW가 정상 동작하는 것을 확인했으므로 그대로 두면 됨
    ## 그리고 LoRA는 학습 대상 파라미터가 340만개(전체의 0.04%)뿐이라, 옵티마이저 선택이 전체 속도에 미치는 영향은 거의 없음
    optim="adamw_torch_fused",

    # [로그 주기] 10 step마다 현재 loss, 학습률 등을 콘솔/로거에 출력
    ## 왜 필요? loss가 내려가고 있는지 눈으로 확인해야 학습이 정상인지 판단 가능. 너무 자주 찍으면 로그가 지저분해짐
    logging_steps=1,

    # [저장 전략] 체크포인트를 어떤 기준으로 저장할지. "steps"=일정 step마다, "epoch"=에포크 끝날 때마다, "no"=저장 안 함
    save_strategy="steps",

    # [저장 주기] save_strategy="steps"일 때, 몇 step마다 저장할지
    ## 이 설정 기준 총 595 step이므로 대략 11~12개의 체크포인트가 쌓임
    ## LoRA를 쓰면 8B 원본 모델이 아니라 작은 어댑터 가중치만 저장되므로 용량 부담은 크지 않음
    ## 참고: 디스크가 걱정되면 save_total_limit=2 처럼 최신 N개만 남기는 옵션을 추가하면 됨
    save_steps=50,

    # [혼합 정밀도] 계산을 32비트(float32) 대신 16비트 bfloat16으로 수행
    ## 왜 필요? 메모리 사용량이 대략 절반이 되고 계산도 훨씬 빨라짐
    ## float16 대신 bfloat16을 쓰는 이유: bfloat16은 표현 가능한 "범위"가 float32와 같아서 값이 넘치거나(overflow)
    ##                                   0으로 뭉개지는(underflow) 학습 불안정 문제가 훨씬 적음
    ## 주의: NVIDIA Ampere 세대(A100, RTX 30xx) 이상에서만 지원됨. 구형 GPU라면 fp16=True를 쓸 것
    bf16=True,

    # [학습률] 한 번의 업데이트에서 파라미터를 얼마나 크게 움직일지 결정하는, 가장 중요한 하이퍼파라미터
    ## 너무 크면 최적점을 지나쳐 loss가 튀거나 발산하고, 너무 작으면 학습이 거의 진행되지 않음
    ## 1e-4는 전체 파인튜닝의 통상값(2e-5)보다 약 5~10배 큰 값인데, 이는 LoRA이기 때문
    ##   -> LoRA는 원본 모델은 얼려두고 아주 작은 추가 행렬만 학습하므로, 같은 효과를 내려면 더 과감하게 움직여야 함
    learning_rate=1e-4,

    # [그래디언트 클리핑] 그래디언트 벡터의 크기(L2 norm)가 0.3을 넘으면 0.3으로 강제로 줄임
    ## 왜 필요? 어쩌다 이상한 샘플 하나 때문에 그래디언트가 폭발하면 파라미터가 크게 망가져 학습이 통째로 무너짐
    ##          이걸 막아주는 안전장치. 기본값 1.0보다 낮은 0.3은 "더 보수적으로, 안정성 위주로" 학습하겠다는 의미
    max_grad_norm=0.3,

    # [워밍업] 학습 초반에 학습률을 0부터 목표치까지 서서히 끌어올리는 구간의 비율 (전체 step의 3%)
    ## 왜 필요? 학습 시작 직후에는 파라미터가 아직 데이터에 적응하지 못한 상태라, 처음부터 큰 학습률을 쓰면 모델이 망가지기 쉬움
    ## !! 주의 1: 아래 lr_scheduler_type이 "constant"이면 transformers 내부에서 워밍업 인자를 아예 전달하지 않으므로
    ##            이 설정은 실제로 아무 효과가 없음. 워밍업을 쓰려면 lr_scheduler_type="constant_with_warmup"으로 바꿔야 함
    ## !! 주의 2: warmup_ratio는 현재 설치된 transformers 5.x에서 deprecated 경고가 뜸 (v5.2에서 제거 예정, warmup_steps 사용 권장)
    warmup_ratio=0.03,

    # [학습률 스케줄러] 학습이 진행됨에 따라 학습률을 어떻게 변화시킬지
    ## "constant"  : 처음부터 끝까지 1e-4로 고정
    ## "linear"    : 목표치에서 0까지 선형으로 감소 (기본값)
    ## "cosine"    : 코사인 곡선을 따라 부드럽게 감소
    ## 보통 후반부에 학습률을 줄여야 미세 조정이 잘 되지만, LoRA + 짧은 학습에서는 constant도 흔히 쓰임
    lr_scheduler_type="constant",

    # [허브 업로드] 학습 결과를 허깅페이스 허브에 자동 업로드할지 여부. False면 로컬에만 저장
    push_to_hub=False,

    # [컬럼 유지] 데이터셋에서 모델 forward()의 인자 이름과 매칭되지 않는 컬럼을 자동으로 버릴지 여부
    ## 기본값 True이면 우리 데이터의 "messages" 컬럼이 모델 입력 이름이 아니라는 이유로 통째로 삭제되어 학습이 불가능해짐
    ## 그래서 False로 꺼서 "messages"를 데이터 콜레이터(collator)까지 살려 보내는 것
    remove_unused_columns=False,

    # [자동 전처리 끄기] TRL이 알아서 해주는 데이터 준비 과정(챗 템플릿 적용 -> 토크나이즈 -> max_length로 자르기)을 건너뜀
    ## 왜 쓰나? 프롬프트 구성이나 loss 마스킹을 직접 제어하고 싶을 때 사용
    ## !! 주의: 이걸 켜면 토크나이즈되지 않은 원본 "messages"가 그대로 콜레이터로 넘어가므로,
    ##          SFTTrainer에 직접 만든 data_collator를 반드시 함께 넘겨줘야 함 (아니면 학습 시점에 에러)
    ##          TRL에 전처리를 맡길 거라면 이 두 줄(remove_unused_columns, dataset_kwargs)을 지우는 편이 더 간단함
    dataset_kwargs={"skip_prepare_dataset": True},

    # [실험 추적 도구] wandb, tensorboard 같은 로깅 도구로 학습 지표를 전송할지 지정
    ## !! 버그 주의: None을 넣으면 transformers 내부에서 리스트로 감싸져 [None]이 되고,
    ##              SFTTrainer 생성 시 "None is not supported" ValueError가 발생함
    ##              아무 도구도 쓰지 않으려면 반드시 문자열 "none" 을 넣어야 함 (report_to="none")
    report_to="none",

    # [설정하지 않은 옵션: max_length]  기본값 1024
    ## SFTConfig의 max_length는 원래 "학습 시 한 샘플의 최대 토큰 길이"이며 이보다 긴 문장은 뒤가 잘려나갑니다.
    ##
    ## !! 다만 이 스크립트에서는 max_length가 아무 효과가 없습니다.
    ##    바로 위에서 dataset_kwargs={"skip_prepare_dataset": True}로 TRL의 자동 전처리를 껐기 때문입니다.
    ##    TRL은 잘라내기(truncation)를 "전처리 단계"에서 수행하는데, 그 단계 자체를 건너뛰므로
    ##    max_length를 몇으로 주든 무시됩니다.
    ##    (trl/trainer/sft_trainer.py 주석: "When preparation is skipped (`skip_prepare_dataset=True`),
    ##     no truncation is applied and the dataset must already be truncated.")
    ##
    ## => 즉 이 스크립트에서 실제로 길이를 자르는 것은 위쪽에 정의한 max_seq_length 변수이고,
    ##    그 값이 collate_fn 안의 tokenizer(truncation=True, max_length=max_seq_length)로 전달됩니다.
    ##    길이를 조절하고 싶다면 여기가 아니라 max_seq_length를 수정해야 합니다.
    ##
    ## 참고) 이 데이터셋의 실제 토큰 길이 분포 (챗 템플릿 적용 후, 991개 기준)
    ##       중앙값 1563 / 평균 1635 / 최대 5907
    ##       1024 이하  57개 ( 5.8%)
    ##       2048 이하 839개 (84.7%)
    ##       4096 이하 988개 (99.7%)   <- 현재 max_seq_length=4096은 99.7%를 온전히 담음
    ##       (길이를 늘리면 GPU 메모리 사용량이 함께 늘어나므로 배치 크기와 함께 조절 필요)
    # max_length=max_seq_length
)

print("\n=============================================")

######################################################################
# 모델 학습을 위한 데이터 준비

## 정수 인코딩
### 학습 데이터에 챗 템플릿 적용 후 정수 인코딩, 모델의 입력(input_ids)와 모델의 응답(labels) 분리

### 모델은 입력에 따라 어떤 응답을 생성할지 학습하는 것이므로, 
### 학습을 위한 labels를 만들 때 필요없는 부분(시스템/유저 프롬프트 부분)은 -100으로 처리해야 한다.
### -> 실제 학습시 labels에 -100인 부분들은 생성을 위한 학습 대상에서 제외된다.

### 1. 챗 템플릿 적용 후
# <|begin_of_text|>
# <|start_header_id|>system<|end_header_id|>
# 당신은 친절한 AI 어시스턴트입니다.
# <|eot_id|>
# <|start_header_id|>user<|end_header_id|>
# 안녕하세요 , 오늘 날씨는 어떤가요?
# <|eot_id|>
# <|start_header_id|>assistant<|end_header_id|>
# 안녕하세요! 오늘 날씨는 맑고 화창합니다.
# <|eot_id|>

### 2. 토크나이즈하면 input_ids를 얻는다.
# input_ids = [
#    128000, # <|begin_of_text|>
#    128006, 9125, 128007, 198, # <|start_header_id|>system<|end_header_id|> (줄바꿈)
#    22173, 13, 126808, 49816, 33302, 23239, 18966, 13, # 당신은 친절한 AI 어시스턴트입니다.
#    128009, # <|eot_id|>
#    128006, 882, 128007, 198, # <|start_header_id|>user<|end_header_id|> (줄바꿈)
#    118145, 11, 24482, 1174, 107485, 102823, 64337, 30, # 안녕하세요, 오늘 날씨는 어떤가요?
#    128009, # <|eot_id|>
#    128006, 78191, 128007, 198, # <|start_header_id|>assistant<|end_header_id|> (줄바꿈)
#    118145, 0, 24482, 1174, 107485, 102823, 64337, 107823, 108562, 13, # 안녕하세요! 오늘 날씨는 맑고 화창합니다.
#    128009 # <|eot_id|>
# ]

### 3. 모델이 학습/생성할 필요가 없는 부분은 -100으로 처리한 labels를 만든다.
### -> -100으로 처리하는 이유: 손실함수(torch.nn.CrossEntropyLoss)는 ignore_index의 디폴트 값이 -100이다.
### -> 모델은 -100이 아닌 부분만 학습하여, 모델이 응답만 학습하도록 유도
# labels = [
#   -100, # <|begin_of_text|>
#   -100, -100, -100, -100, # <|start_header_id|>system<|end_header_id|> (줄바꿈)
#   -100, -100, -100, -100, -100, -100, -100, -100, # 당신은 친절한 AI 어시스턴트입니다.
#   -100, # <|eot_id|>
#   -100, -100, -100, -100, # <|start_header_id|>user<|end_header_id|> (줄바꿈)
#   -100, -100, -100, -100, -100, -100, -100, -100, # 안녕하세요 , 오늘 날씨는 어떤가요?
#   -100, # <|eot_id|>
#   -100, -100, -100, -100, # <|start_header_id|>assistant<|end_header_id|> (줄바꿈)
#   118145, 0, 24482, 1174, 107485, 102823, 64337, 107823, 108562, 13, # 안녕하세요! 오늘 날씨는 맑고 화창합니다.
#   128009 # <|eot_id|>
# ]

## 모델 학습시 데이터 전처리 진행하는 함수
def collate_fn(batch):
    new_batch = {
        "input_ids": [],
        "attention_mask": [],
        "labels": []
    }

    for example in batch:
        messages = example["messages"]

        # LLaMA 3 채팅 템플릿 적용 (시작 토큰 포함)
        ## !! [알아두면 좋은 점 1: 공식 템플릿과의 미묘한 차이]
        ## Llama 3의 공식 형식은 헤더 뒤에 줄바꿈이 "두 번" 들어갑니다.
        ##   공식      : <|start_header_id|>user<|end_header_id|>\n\n{내용}<|eot_id|>
        ##   아래 코드 : <|start_header_id|>user<|end_header_id|>\n{내용}<|eot_id|>     <- \n 이 한 번
        ## 실제 확인: tokenizer.apply_chat_template(...)를 출력해 보면 \n\n 으로 나옵니다.
        ##
        ## 학습이 실패하지는 않습니다(모델이 이 형식에 맞춰 학습되니까요). 문제는 추론 시점입니다.
        ## 추론할 때 보통 apply_chat_template()를 쓰는데, 그러면 \n\n 형식이 들어가서
        ## "학습한 형식 != 추론 형식"이 되고, 출력 품질이 미묘하게 떨어지는 원인이 됩니다.
        ##
        ## 그래서 제어 토큰을 직접 문자열로 조립하는 것은 초보자에게 권하지 않습니다.
        ## 모델을 바꿀 때마다 형식이 달라져 매번 틀릴 수 있기 때문입니다.
        ## 같은 폴더의 qwen3_finance_news_analysis.py 는 이 부분을
        ## tokenizer.apply_chat_template()로 바꿔서 이런 실수 가능성을 없앤 버전이므로 비교해 보세요.
        prompt = "<|begin_of_text|>"
        for msg in messages:
            role = msg["role"]
            content = msg["content"].strip()
            prompt += f"<|start_header_id|>{role}<|end_header_id|>\n{content}<|eot_id|>"

        # 마지막 assistant 메시지는 응답으로 간주하고 레이블에 포함
        text = prompt.strip()

        # 토큰화
        tokenized = tokenizer(
            text,
            truncation=True,
            max_length=max_seq_length,
            padding=False,
            return_tensors=None,
        )

        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]
        labels = [-100] * len(input_ids)

        # assistant 응답의 시작 위치 찾기
        assistant_header = "<|start_header_id|>assistant<|end_header_id|>\n"
        assistant_tokens = tokenizer.encode(assistant_header, add_special_tokens=False)
        eot_token = "<|eot_id|>"
        eot_tokens = tokenizer.encode(eot_token, add_special_tokens=False)

        # 레이블 범위 지정
        ## 앞에서 찾은 assistant 헤더 토큰열이 input_ids 안에서 처음 등장하는 위치를 찾고,
        ## 그 다음 토큰부터 <|eot_id|>까지를 "정답"으로 표시(labels에 실제 토큰 id를 채움)합니다.
        ## 나머지(시스템/유저 프롬프트)는 위에서 이미 -100으로 채워져 있어 loss 계산에서 제외됩니다.
        ##
        ## !! [알아두면 좋은 점 2: 잘림(truncation)이 발생하면 여기서 사고가 난다]
        ## 위 tokenizer(truncation=True, max_length=max_seq_length)는 문장을 "뒤에서부터" 자릅니다.
        ## 그런데 정답(assistant 응답)은 문장의 맨 뒤에 있으므로, 프롬프트만으로 max_seq_length를
        ## 다 채우는 긴 샘플에서는 정답이 잘려 나가고 아래 두 가지 중 하나가 발생합니다.
        ##
        ##   (a) assistant 헤더까지 통째로 잘린 경우
        ##       -> while 루프가 아무것도 찾지 못해 labels가 전부 -100으로 남습니다.
        ##          학습 대상 토큰이 0개인 샘플의 cross entropy는 nan이 되고, 그 스텝에서 가중치가 망가집니다.
        ##          (loss에 nan이 찍히기 시작하면 이후 학습은 전부 무의미해집니다)
        ##
        ##   (b) assistant 헤더는 남았지만 끝의 <|eot_id|>가 잘린 경우
        ##       -> 아래 안쪽 while이 <|eot_id|>를 못 찾고 end == len(input_ids)까지 증가한 뒤,
        ##          바로 다음 for j in range(end, end + len(eot_tokens)) 에서 labels[len(input_ids)]에
        ##          접근하며 IndexError가 발생합니다.
        ##
        ## 이 데이터셋의 토큰 길이 분포는 중앙값 1563 / 평균 1635 / 최대 5907 이고 max_seq_length=4096이므로,
        ## 991개 중 3개(0.3%)가 잘림 대상입니다. 확률은 낮지만 한 번만 걸려도 학습이 무너지므로,
        ## 학습 전에 길이가 max_seq_length를 넘는 샘플을 데이터셋에서 미리 걸러내는 것을 권합니다.
        i = 0
        while i <= len(input_ids) - len(assistant_tokens):
            if input_ids[i:i + len(assistant_tokens)] == assistant_tokens:

                # assistant 응답의 시작과 끝 부분 인덱스 찾기
                start = i + len(assistant_tokens)
                end = start

                while end <= len(input_ids) - len(eot_tokens):
                    if input_ids[end:end + len(eot_tokens)] == eot_tokens:
                        break
                    end += 1

                # labels에는 이미 -100으로 채워져 있는 상태
                for j in range(start, end): # 모델 응답 텍스트 (<|start_header_id|>assistant<|end_header_id|>\n 제외, <|eot_id|> 바로 앞까지)
                    labels[j] = input_ids[j]

                for j in range(end, end + len(eot_tokens)):
                    labels[j] = input_ids[j] # <|eot_id|> 토큰 append

                break
            i += 1

        new_batch["input_ids"].append(input_ids)
        new_batch["attention_mask"].append(attention_mask)
        new_batch["labels"].append(labels)

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
    ## (지금 쓰는 Llama-VARCO-8B-Instruct의 pad_token_id는 0('!')로 설정되어 있음이므로 이 스크립트에서는 문제가 없습니다)
    max_length = max(len(ids) for ids in new_batch["input_ids"])
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
print("입력 ID 형태:", batch["input_ids"].shape) # torch.Size([1, 1540])
print("어텐션 마스크 형태:", batch["attention_mask"].shape) # torch.Size([1, 1540])
print("레이블 형태:", batch["labels"].shape) # torch.Size([1, 1540])

# [128000, 128006, 9125, 128007, 198, 65895, 83628, 34804, 56773, 125441, ... 
print('input_ids: ')
print(batch["input_ids"][0].tolist())

# <|begin_of_text|><|start_header_id|>system<|end_header_id|>
# 당신은 주어진 뉴스로부터 종목에 영향을 주는 뉴스인지 판별하는 금융 뉴스 판별기입니다.
# 두 가지 답변 케이스가 존재하며 무조건 파이썬의 dictionary 형식으로 작성하십시오.
# 큰 따옴표 사이에 다른 따옴표들을 적으려고 시도하지 마십시오. 이는 dictionary 파싱을 실패하게 하는 원인이 됩니다. 따라서 주의하십시오.
# ...
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
# {'is_stock_related': False, 'negative_impact_stocks': None, ... 있는 상황입니다.'}<|eot_id|>
print("\nlabels 디코딩 결과 (-100 제외):")
print(decoded_labels)

## 어텐션 마스크 확인

# 0번과 1번 데이터의 길이 확인
example0 = train_dataset[0]
example1 = train_dataset[1]

# 개별 길이 확인 (토큰화후)
tokenized0 = tokenizer(
    #전체 처리 과정과 동일하게 전체 대화를 토큰화
    "<|begin_of_text|>" + "".join([f"<|start_header_id|>{msg['role']}<|end_header_id|>\n{msg['content'].strip()}<|eot_id|>" for msg in example0["messages"]]),
    truncation=True,
    max_length=max_seq_length,
    padding=False,
    return_tensors=None,
)
tokenized1 = tokenizer(
    #전체 처리 과정과 동일하게 전체 대화를 토큰화
    "<|begin_of_text|>" + "".join([f"<|start_header_id|>{msg['role']}<|end_header_id|>\n{msg['content'].strip()}<|eot_id|>" for msg in example1["messages"]]),
    truncation=True,
    max_length=max_seq_length,
    padding=False,
    return_tensors=None,
)
print(f"0번 데이터 길이: {len(tokenized0['input_ids'])}") # 1540
print(f"1번 데이터 길이: {len(tokenized1['input_ids'])}") # 1403

batch = collate_fn([example0, example1])
print("\n배치 처리 후:")
print(f"입력 ID 형태: {batch['input_ids'].shape}") # torch.Size([2, 1540])
print(f"어텐션 마스크 형태: {batch['attention_mask'].shape}") # torch.Size([2, 1540])

# 길이가 짧은 샘플이 길이가 긴 샘플에 맞춰진다. (어텐션 마스크 0이 채워진다.)
max_length_in_batch = max(len(tokenized0['input_ids']), len(tokenized1['input_ids']))
print(f"\n배치내 최대 길이: {max_length_in_batch}")
print(f"0번 샘플 어텐션 마스크 1의 개수: {batch['attention_mask'][0].sum().item()}") # 1540
print(f"0번 샘플 어텐션 마스크 0의 개수: {(batch['attention_mask'][0] == 0).sum().item()}") # 0
print(f"1번 샘플 어텐션 마스크 1의 개수: {batch['attention_mask'][1].sum().item()}") # 1403
print(f"1번 샘플 어텐션 마스크 0의 개수: {(batch['attention_mask'][1] == 0).sum().item()}") # 137

print("\n=============================================")

######################################################################
# 모델 학습

# ==========================================================================================
# [중요] QLoRA는 왜 SFTTrainer에 peft_config를 넘기지 않는가?
#
# 결론부터 말하면 "LoRA를 붙이는 주체가 둘 중 하나여야 하고, 둘이 겹치면 TRL이 에러를 낸다" 입니다.
#
# ── SFTTrainer가 peft_config를 받으면 무슨 일을 하는가 ──────────────────────────────
#   SFTTrainer는 peft_config를 받으면 "아직 LoRA가 안 붙은 맨 모델이 들어왔구나"라고 판단하고,
#   생성자 안에서 우리 대신 get_peft_model(model, peft_config)를 호출해 모델을 PeftModel로 감쌉니다.
#   즉 peft_config를 넘기는 것은 get_peft_model 호출을 TRL에게 위임하는 것과 같습니다.
#   (trl/trainer/sft_trainer.py 의 "# PEFT" 블록)
#
# ── 그런데 QLoRA 분기에서는 이미 우리가 직접 붙여 놨다 ─────────────────────────────
#   위쪽 CUDA 분기에서 model = get_peft_model(model, peft_config) 를 이미 호출했기 때문에,
#   지금 model 변수에 담긴 것은 평범한 모델이 아니라 이미 LoRA가 붙어 있는 PeftModel입니다.
#   여기에 peft_config까지 같이 넘기면 LoRA 위에 LoRA를 또 얹는 이중 적용이 되어버립니다.
#   그러면 학습된 어댑터가 어느 층에 속하는지 알 수 없고 저장/병합도 깨지므로,
#   TRL은 이 상황을 아예 막아두고 아래 에러를 냅니다. (사용자가 만난 그 에러입니다)
#
#     ValueError: You passed a `PeftModel` instance together with a `peft_config` to the trainer.
#                 Please first merge and unload the existing adapter, save the resulting base model,
#                 and then pass that base model along with the new `peft_config` to the trainer.
#
#     (해석: "이미 LoRA가 붙은 모델과 peft_config를 같이 줬다. 새 LoRA를 붙이고 싶다면
#            기존 어댑터를 원본에 병합(merge)하고 떼어낸(unload) 뒤 그 모델을 넘겨라")
#
# ── 정리: 둘 중 하나만 선택한다 ─────────────────────────────────────────────────
#   방식 A (QLoRA / 이 if 분기)
#       내가 직접 prepare_model_for_kbit_training -> get_peft_model 을 호출한다
#       => SFTTrainer에는 peft_config를 넘기지 않는다
#   방식 B (일반 LoRA / 아래 else 분기)
#       get_peft_model을 호출하지 않고 맨 모델만 만든다
#       => SFTTrainer에 peft_config를 넘겨서 TRL이 붙이게 한다
#
# ── QLoRA가 굳이 번거로운 방식 A를 쓰는 이유 ───────────────────────────────────
#   LoRA를 붙이기 "전에" 반드시 prepare_model_for_kbit_training이 끼어들어야 하는데
#   (그래야 원본 동결 / LayerNorm fp32 승격 / 입력 그래디언트 활성화가 이뤄짐),
#   SFTTrainer는 이 함수를 대신 호출해 주지 않습니다. 그래서 4비트 준비 과정을 우리가 직접
#   처리해야 하고, 그 과정에서 get_peft_model까지 직접 호출하게 되는 것입니다.
#
#   참고) 그 외의 QLoRA 부가 처리는 TRL이 알아서 해 줍니다. 이미 PeftModel을 넘겨받은 경우에도
#         - LoRA + gradient_checkpointing 조합에 필요한 enable_input_require_grads() 호출
#         - QLoRA 논문 권고에 따라 LoRA 가중치를 bfloat16으로 캐스팅
#         두 가지를 내부에서 수행하므로 우리가 추가로 손댈 것은 없습니다.
# ==========================================================================================
#
# [참고] 위쪽에서 데이터를 8:2로 나눠 test_dataset(198개)을 만들어 뒀지만 아래 SFTTrainer에는
#        전달하지 않아서 지금은 사용되지 않습니다. 학습 중에 "처음 보는 데이터에서의 성능"을
#        같이 보고 싶다면 두 SFTTrainer 호출에 아래를 추가하면 됩니다.
#          eval_dataset=test_dataset,
#        그리고 SFTConfig에 eval_strategy="steps", eval_steps=50 을 함께 넣어야 실제로 평가가 돌아갑니다.
#        (학습 loss만 보면 "외운 것"과 "배운 것"을 구분할 수 없습니다. 학습 loss는 내려가는데
#         평가 loss가 올라가기 시작하는 지점이 바로 과적합이 시작된 시점입니다.)
if torch.cuda.is_available():
    # QLoRA 적용할 경우 (peft_config를 넘기지 않는다 - 위 설명 참고)
    trainer = SFTTrainer(
        model=model, # 이미 4비트 양자화 + LoRA가 적용된 PeftModel
        args=args, # SFTConfig
        train_dataset=train_dataset,
        data_collator=collate_fn # 직접 만든 전처리 함수 (SFTConfig의 skip_prepare_dataset=True와 짝을 이룸)
        # peft_config=peft_config  <- 이 줄을 살리면 위에서 설명한 ValueError가 발생한다
    )
else:
    # 일반 LoRA 적용할 경우, peft_config를 명시적으로 넘김
    ## 위 else 분기에서 get_peft_model을 호출하지 않았으므로, LoRA를 붙이는 일을 SFTTrainer에게 맡긴다.
    ## SFTTrainer가 생성자 안에서 get_peft_model(model, peft_config)를 호출해 PeftModel로 감싼 뒤 학습을 시작한다.
    trainer = SFTTrainer(
        model=model, # 아직 LoRA가 붙지 않은 맨 bfloat16 모델
        args=args, # SFTConfig
        train_dataset=train_dataset,
        data_collator=collate_fn,
        peft_config=peft_config
    )

# 학습 시작
## push_to_hub=False이므로 실제로는 허브가 아니라 output_dir(로컬)에만 저장된다.
## save_strategy="steps" + save_steps=50 설정에 따라 50 step마다 output_dir/checkpoint-50, -100 ... 이 쌓인다.
trainer.train() # 모델이 자동으로 output_dir에 저장됨 (push_to_hub=True로 바꾸면 허브에도 업로드)

# 모델 저장
## [초보자 주의] LoRA/QLoRA로 학습한 경우 여기 저장되는 것은 "원본 모델 전체"가 아니라
## LoRA 어댑터 가중치(adapter_model.safetensors)와 설정(adapter_config.json)뿐입니다. 보통 수 MB~수십 MB.
## 그래서 나중에 추론할 때는 "원본 모델을 먼저 불러오고 그 위에 어댑터를 얹는" 2단계가 필요합니다.
##   from peft import PeftModel
##   base  = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16)
##   model = PeftModel.from_pretrained(base, "<output_dir>")
##   model = model.merge_and_unload()  # (선택) 어댑터를 원본에 합쳐 일반 모델처럼 만들기
##
## 참고) QLoRA로 학습한 어댑터를 4비트가 아닌 원본(bf16) 위에 얹어 추론하는 것도 가능합니다.
##       다만 학습은 4비트 원본을 기준으로 이뤄졌으므로 결과가 미세하게 달라질 수 있습니다.
trainer.save_model() # 최종 모델(어댑터)을 저장

print("\n=============================================")

######################################################################
# 맥북(Apple Silicon / MPS)에서 이 스크립트를 학습시킬 때의 설정 가이드
#
# 이 스크립트의 SFTConfig 기본값은 NVIDIA GPU 서버를 기준으로 잡힌 값입니다.
# 맥북에서 그대로 돌리면 일부 설정이 오히려 학습을 크게 느리게 만들기 때문에,
# 실제로 모델을 MPS에 올려 측정한 결과와 그에 따른 권장 설정을 아래에 정리합니다.
#
#   [측정 환경]
#     Apple M4 Pro (14코어) / 통합 메모리 48GB / macOS
#     torch 2.12.1, transformers 5.14.1, trl 1.12.0, peft 0.20.0
#     모델: NCSOFT/Llama-VARCO-8B-Instruct (8.03B) + LoRA(r=8, q_proj/v_proj)
#           -> 실제 학습되는 파라미터는 3,407,872개로 전체의 0.04%뿐
#
#
# ────────────────────────────────────────────────────────────────────
# 1. 지금 MPS로 학습되고 있는 게 맞는가?  ->  맞습니다
# ────────────────────────────────────────────────────────────────────
#
#   MPS(Metal Performance Shaders)는 애플 실리콘의 GPU를 PyTorch에서 사용하기 위한 백엔드로,
#   NVIDIA 환경에서의 "cuda"에 해당하는 이름입니다. 아래 3가지를 확인하면 됩니다.
#
#     accelerate.utils.get_max_memory()   ->  {'mps': 35981377536}   (약 33.5GB를 GPU 예산으로 인식)
#     next(model.parameters()).device     ->  mps:0                  (device_map="auto"가 MPS에 올림)
#     SFTConfig(...).device               ->  mps                    (Trainer도 MPS를 사용, n_gpu=1)
#
#   attention 구현체는 sdpa가 자동 선택되고, bf16=True도 정상 통과합니다.
#   즉 학습이 느린 이유는 "설정이 잘못돼서 CPU로 돌고 있어서"가 아니라,
#   8B 모델 학습 자체가 맥북에게 원래 매우 무거운 작업이기 때문입니다.
#
#
# ────────────────────────────────────────────────────────────────────
# 2. 실측 벤치마크
# ────────────────────────────────────────────────────────────────────
#
#   측정 방법: LoRA 학습 1 스텝(forward + backward + optimizer step)을 3회 반복 후 최소값
#   공통 조건: bf16, gradient_checkpointing=True, attention=sdpa
#
#     배치  시퀀스길이   미니배치 총토큰  |   1스텝 소요  |   처리량
#     ----  ----------   ---------------  |   ----------  |   ----------
#      1       2048           2,048       |     14.85초   |   138 tok/s     <- 가장 효율적
#      2       1024           2,048       |     15.24초   |   134 tok/s
#      2       2048           4,096       |     45.81초   |    89 tok/s     <- 처리량 급락
#      4       2048           8,192       |     93.82초   |    87 tok/s
#      1       4096           4,096       |     52.20초   |    79 tok/s
#
#   [핵심 원리]
#   성능을 결정하는 것은 배치 크기 자체가 아니라 "한 번의 forward에 들어가는 총 토큰 수"입니다.
#   총 토큰이 2048을 넘는 순간 통합 메모리 압박이 발생해 처리량이 138 -> 89 tok/s로 무너집니다.
#   (2048 이하에서는 bs=1이든 bs=2든 처리량이 ~135 tok/s로 사실상 동일합니다.)
#
#   맥북에는 별도의 VRAM이 없고 CPU와 GPU가 48GB를 나눠 쓰는 통합 메모리 구조라,
#   메모리를 아끼는 것이 곧 속도로 직결됩니다. NVIDIA GPU와 정반대의 결론이 나오는 이유입니다.
#
#   => 맥북에서의 최적화 목표: "미니배치 총 토큰 수를 2048 근처로 유지할 것"
#
#
# ────────────────────────────────────────────────────────────────────
# 3. 권장 설정 (이 스크립트에서 바꿔야 할 값)
# ────────────────────────────────────────────────────────────────────
#
#   (1) max_seq_length = 2048        # 현재 8192 -> 2048
#
#       이 스크립트는 dataset_kwargs={"skip_prepare_dataset": True}로 TRL의 자동 전처리를 껐기 때문에
#       SFTConfig의 max_length는 동작하지 않고, 위쪽에 정의된 max_seq_length 변수가
#       collate_fn 안의 tokenizer(truncation=True, max_length=max_seq_length)로 전달되어
#       실제 자르기를 담당합니다. 즉 맥북에서 길이를 조절하는 유일한 손잡이가 이 변수입니다.
#
#       현재 값 8192는 이 데이터셋의 최대 길이(5907 토큰)보다 크기 때문에 사실상 자르기가 없는 상태입니다.
#       그 결과 5000 토큰이 넘는 샘플이 그대로 들어와 위 표의 가장 느린 구간에서 학습됩니다.
#
#         이 데이터셋의 실제 토큰 길이 분포 (챗 템플릿 적용 후, 991개 기준)
#           중앙값 1563 / 평균 1635 / 최대 5907
#           1024 이하  57개 ( 5.8%)
#           2048 이하 839개 (84.7%)   <- 2048이면 대부분의 샘플을 온전히 담을 수 있음
#           4096 이하 988개 (99.7%)
#
#   (2) per_device_train_batch_size = 1     # 현재 2 -> 1
#       gradient_accumulation_steps  = 4    # 현재 2 -> 4
#
#       배치를 "줄이는" 것이 오히려 빨라지는 이유는 위 [핵심 원리] 때문입니다.
#       그리고 1 x 4 = 4 로 실질 배치 크기(effective batch size)는 기존 2 x 2 = 4 와 완전히 동일하므로,
#       학습 결과(수렴 양상)는 그대로 두고 속도만 얻는 손해 없는 변경입니다.
#
#       추가 이득: 이 데이터는 길이가 922~5907 토큰으로 제각각이라, 배치 크기가 2 이상이면
#       짧은 샘플이 긴 샘플 길이에 맞춰 패딩되며 그만큼의 계산이 통째로 버려집니다.
#       (위쪽 배치 처리 예제에서 1번 샘플에 패딩 137개가 붙는 것이 바로 이 현상입니다.)
#       배치 크기가 1이면 패딩이 아예 발생하지 않으므로 체감 이득은 아래 표보다 더 큽니다.
#
#     훈련 데이터 793개 x 3 에포크 기준 예상 소요 시간
#       현재 설정 (bs=2, accum=2, max_seq_length=8192)  ->  약 15시간
#       권장 설정 (bs=1, accum=4, max_seq_length=2048)  ->  약  7.5시간
#
#
# ────────────────────────────────────────────────────────────────────
# 4. 절대 바꾸면 안 되는 설정 (GPU 서버 튜토리얼과 정반대인 항목)
# ────────────────────────────────────────────────────────────────────
#
#   bf16=True 를 유지할 것
#     fp16으로 바꾸면 오히려 1.5배 느려집니다. (bs=1/seq2048 기준 14.85초 -> 21.86초)
#
#   gradient_checkpointing=True 를 유지할 것
#     GPU 서버에서는 이 옵션을 끄면 재계산이 사라져 빨라진다고 알려져 있지만, 맥북에서는 정반대입니다.
#       끈 경우 (bs=2, seq 1024)  ->  131.9초 (8.6배 느려짐. 메모리 압박으로 스왑 발생)
#       끈 경우 (bs=2, seq 2048)  ->  OOM 발생 (44GB 할당 후 실패)
#     메모리를 아끼는 것이 곧 속도인 통합 메모리 구조에서는 이 옵션이 사실상 필수입니다.
#
#   optim="adamw_torch_fused" 를 유지할 것
#     torch 2.12의 MPS에서 fused AdamW가 정상 동작하는 것을 확인했습니다.
#     어차피 학습 대상 파라미터가 340만개뿐이라 옵티마이저 선택은 전체 속도에 거의 영향이 없습니다.
#
#
# ────────────────────────────────────────────────────────────────────
# 5. MPS에서 원천적으로 불가능한 것 (튜토리얼에 나와도 따라 하면 안 되는 것)
# ────────────────────────────────────────────────────────────────────
#
#   QLoRA / 4bit·8bit 양자화
#     양자화를 담당하는 bitsandbytes 라이브러리가 CUDA 전용이라 애플 실리콘에서 사용할 수 없습니다.
#     따라서 "4bit로 모델을 불러와 메모리를 아낀다"는 방식은 맥북에서 선택지가 아닙니다.
#
#   FlashAttention-2
#     미지원입니다. (transformers.utils.is_flash_attn_2_available() == False)
#     현재 자동 선택되는 sdpa(Scaled Dot Product Attention)가 맥북에서 쓸 수 있는 최선입니다.
#
#
# ────────────────────────────────────────────────────────────────────
# 6. 학습 시간을 더 줄이는 방법 (효과가 큰 순서)
# ────────────────────────────────────────────────────────────────────
#
#   조치                                                        예상 소요 시간
#   ----------------------------------------------------------  --------------
#   현재 설정 (bs=2, accum=2, max_seq_length=8192, 3 에포크)         약 15시간
#   bs=1 + accum=4 + max_seq_length=2048 로 변경                     약 7.5시간
#   위 + num_train_epochs 를 3 -> 1 로 축소                          약 2.5시간
#   위 + 훈련 데이터를 200개로 축소 (파이프라인 검증 전용)           약  40분
#
#   [권장 작업 순서]
#   먼저 훈련 데이터 200개 / 1 에포크로 끝까지 한 번 돌려서 전체 파이프라인이 정상인지 확인한 뒤
#   본 학습에 들어가십시오. 특히 이 스크립트는 직접 만든 collate_fn에 의존하므로 학습 시작 직후
#   레이블 마스킹이나 패딩 관련 에러가 날 여지가 있는데, 7시간을 돌리고 실패하면 손해가 매우 큽니다.
#
#   [근본적인 해결책]
#   - 모델을 1B~3B급 한국어 모델로 교체하는 것이 가장 큰 효과를 냅니다.
#     8B는 맥북에서 학습하기에 무거운 크기이며, 위 최적화를 다 해도 시간 단위의 학습을 피할 수 없습니다.
#   - 8B를 반드시 써야 한다면 클라우드 GPU를 쓰는 편이 현실적입니다. (A100 기준 20~30분 수준)
######################################################################

######################################################################
# 평가 준비 (테스트 데이터)
#
# 맨 위에서 8:2로 나눠둔 test_dataset(198개)은 학습에 한 번도 쓰지 않은 데이터입니다.
# 이걸로 "파인튜닝 전(베이스) 모델"과 "파인튜닝 후 모델"의 출력을 나란히 비교합니다.
#
# 필요한 것은 두 가지입니다.
#   prompt_lst : 시스템 + 유저 프롬프트 + 생성 프롬프트(모델이 이어서 쓸 시작점)  -> 모델 입력
#   label_lst  : 정답 assistant 응답                                             -> 비교 기준
#
# ==========================================================================================
# [중요] 이 파일은 왜 apply_chat_template 을 쓰지 않고 프롬프트를 직접 조립하는가
#
# 추론 프롬프트는 "학습 때 쓴 형식"과 한 글자도 달라서는 안 됩니다.
# 그런데 위쪽 collate_fn 은 (알아두면 좋은 점 1 참고) 헤더 뒤 줄바꿈을 한 번만 넣는
# 비공식 형식으로 학습 데이터를 만들었습니다.
#
#   collate_fn(학습)                  : <|start_header_id|>assistant<|end_header_id|>\n{내용}<|eot_id|>
#   apply_chat_template(공식)         : <|start_header_id|>assistant<|end_header_id|>\n\n{내용}<|eot_id|>
#
# 여기서 apply_chat_template 을 쓰면 \n 이 하나 더 들어가 "학습한 형식 != 추론 형식" 이 되고,
# 파인튜닝 모델이 손해를 봅니다. 그래서 아래 build_chat_text 로 학습과 똑같이 조립합니다.
# (베이스 모델 입장에서는 반대로 \n 이 하나 부족한 셈이지만, 두 모델을 같은 프롬프트로
#  비교하는 것이 공정하고, 학습된 쪽 형식을 맞춰주는 것이 이 실험의 목적에 부합합니다)
#
# 애초에 학습 코드에서 apply_chat_template 을 썼다면 이 고민 자체가 없습니다.
# 같은 폴더의 qwen3 / kanana2 파일은 그렇게 바꾼 버전이므로 비교해 보세요.
#
# 또 하나, VARCO(Llama-3 계열)의 챗 템플릿은 add_generation_prompt=False 를 줘도
# 맨 끝에 <|start_header_id|>assistant<|end_header_id|>\n\n 를 무조건 한 번 더 붙입니다.
# (위쪽 "챗 템플릿 적용 테스트" 출력의 마지막 줄이 그것입니다)
# 그래서 apply_chat_template 결과를 assistant 헤더로 split 하면 조각이 2개가 아니라 3개로 나옵니다.
# 직접 조립하면 이런 모델별 함정도 함께 피할 수 있습니다.
# ==========================================================================================

ASSISTANT_HEADER = "<|start_header_id|>assistant<|end_header_id|>\n"
RESPONSE_END = "<|eot_id|>" # Llama-3 의 턴 종료 토큰 (id 128009)

## 위 collate_fn 과 완전히 동일한 규칙으로 챗 템플릿을 조립한다
def build_chat_text(messages):
    text = "<|begin_of_text|>"
    for msg in messages:
        text += f"<|start_header_id|>{msg['role']}<|end_header_id|>\n{msg['content'].strip()}<|eot_id|>"
    return text

prompt_lst = []
label_lst = []

for messages in test_dataset["messages"]:
    text = build_chat_text(messages)

    ## assistant 헤더를 경계로 두 조각으로 나눈다 (assistant 턴이 1개뿐이므로 항상 정확히 2조각)
    before_assistant, after_assistant = text.split(ASSISTANT_HEADER)

    ## 입력: 시스템 + 유저 프롬프트 + 생성 프롬프트
    ##       split 하면 경계 문자열 자체는 사라지므로 다시 붙여줘야 한다
    prompt_lst.append(before_assistant + ASSISTANT_HEADER)

    ## 정답: 모델 응답 본문만 (뒤에 붙은 <|eot_id|> 는 잘라낸다)
    label_lst.append(after_assistant.split(RESPONSE_END)[0])

print("----prompt_lst[0]----")
print(prompt_lst[0])
print("----label_lst[0]----")
print(label_lst[0])

print("\n=============================================")

######################################################################
# 추론 함수 정의
from transformers import pipeline

## 생성을 멈출 토큰 id
##
## [중요] 이 모델에서는 eos_token_id 를 반드시 직접 넘겨야 합니다.
##        Llama-3 계열은 "텍스트의 끝"과 "대화 턴의 끝"을 다른 토큰으로 구분하는데,
##        VARCO 의 config.json 에는 전자만 등록돼 있습니다.
##          config.eos_token_id = 128001 (<|end_of_text|>)   <- generate 가 기본으로 보는 값
##          챗 템플릿의 턴 종료  = 128009 (<|eot_id|>)        <- 우리가 학습시킨 종료 토큰
##        그대로 두면 모델이 <|eot_id|> 를 뱉어도 멈추지 않고 max_new_tokens 까지
##        다음 턴을 혼자 지어내며 계속 생성합니다.
##        (tokenizer.eos_token 은 <|eot_id|> 로 올바르게 지정되어 있어 값이 서로 어긋나 있습니다)
eos_token = tokenizer(RESPONSE_END, add_special_tokens=False)["input_ids"][0] # 128009

## 추론 메서드 정의
def test_inference(pipe, prompt):
    outputs = pipe(
        prompt,
        max_new_tokens=1024,                 # 정답 응답이 길어도 잘리지 않을 만큼
        eos_token_id=eos_token,              # 이 토큰이 나오면 생성 중단 (위 설명 참고)
        pad_token_id=tokenizer.pad_token_id, # 지정 안 하면 경고가 뜬다 (배치 1이라 실제 패딩은 없음)
        do_sample=False,                     # 그리디 디코딩. 매번 같은 결과가 나와야 두 모델을 비교할 수 있다
        add_special_tokens=False,            # 프롬프트에 <|begin_of_text|> 가 이미 있으므로 BOS 를 또 붙이지 않게 한다
        return_full_text=False,              # 프롬프트를 뺀 "새로 생성된 텍스트"만 받는다
    )
    return outputs[0]["generated_text"].strip()

print("\n=============================================")

######################################################################
# 베이스 모델 vs 파인튜닝 모델 비교

## 먼저 학습에 쓴 모델을 메모리에서 내린다.
## [맥북에서는 필수] 이 모델은 8B(bf16으로 약 16GB)라서, 학습용 모델을 내리지 않고
## 추론용 모델을 또 올리면 32GB가 되어 통합 메모리가 스왑으로 넘어가거나 그대로 죽습니다.
import gc

del trainer, model
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
elif torch.backends.mps.is_available():
    torch.mps.empty_cache()

print("\n=============================================")
print("베이스 모델 추론 (파인튜닝 전)")

## 학습에 쓴 것과 같은 설정으로 원본 모델을 다시 불러온다
base_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)

## device를 지정하지 않으면 pipeline이 사용 가능한 가속기(MPS/CUDA)를 스스로 골라 모델을 옮긴다
pipe = pipeline("text-generation", model=base_model, tokenizer=tokenizer)

## 베이스 모델은 이 작업을 학습한 적이 없으므로 시스템 프롬프트의 지시를 어림짐작으로 따라갑니다.
## "dictionary 비슷한 것"은 나오지만 키 이름이 빠지거나, 지시사항 문구를 그대로 베껴 쓰거나,
## 근거 없는 종목이 채워지는 등 파싱이 실패하는 출력이 섞여 나오는 것이 정상입니다.
for prompt, label in zip(prompt_lst[10:15], label_lst[10:15]):
    print(f" response:\n{test_inference(pipe, prompt)}")
    print(f" label:\n{label}")
    print("-"*50)

print("\n=============================================")
print("파인튜닝 모델 추론 (LoRA 어댑터 부착)")

from peft import PeftModel

## trainer.save_model() 이 최종 어댑터를 저장한 위치 = SFTConfig(output_dir=...) 와 같다.
## 중간 체크포인트로 비교하고 싶으면 "llama3-8b-summarizer-ko/checkpoint-500" 처럼 지정하면 된다.
## (save_steps=50 이고 총 step 은 793 x 3 / 4 = 약 594 이므로 checkpoint-50 ~ -550 이 쌓인다)
peft_model_id = "llama3-8b-summarizer-ko"

## [주의] PeftModel.from_pretrained 는 위에서 만든 base_model 안에 LoRA 층을 직접 끼워 넣는다.
##        즉 이 줄 이후의 base_model 은 더 이상 "베이스 모델"이 아니다.
##        그래서 베이스 모델 추론을 반드시 먼저 끝내야 한다.
##        대신 16GB 원본 가중치를 두 번 읽지 않으므로, 8B 모델에서는 이 방식이 사실상 필수다.
##
##        (메모리가 넉넉하다면 아래 두 줄로 베이스와 완전히 분리된 모델을 만들 수도 있다)
##          from peft import AutoPeftModelForCausalLM
##          fine_tuned_model = AutoPeftModelForCausalLM.from_pretrained(peft_model_id, torch_dtype=torch.bfloat16)
fine_tuned_model = PeftModel.from_pretrained(base_model, peft_model_id)
pipe = pipeline("text-generation", model=fine_tuned_model, tokenizer=tokenizer)

## 파인튜닝 모델은 학습 데이터의 형식(파이썬 dict 문자열, 키 8개)을 그대로 따라가는 것이 정상입니다.
## 종목명이나 요약 내용이 정답과 완전히 같지는 않아도, "형식이 깨지지 않는다"는 점이 가장 큰 차이입니다.
for prompt, label in zip(prompt_lst[10:15], label_lst[10:15]):
    print(f" response:\n{test_inference(pipe, prompt)}")
    print(f" label:\n{label}")
    print("-"*50)
