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

## 사용할 허깅페이스의 모델 ID
model_id = "NCSOFT/Llama-VARCO-8B-Instruct" # Meta-Llama-3.1-8B 모델을 한국어 성능에 특화되도록 추가학습된 모델

tokenizer = AutoTokenizer.from_pretrained(model_id)

## LLaMa를 위한 챗 템플릿 적용
## LLaMa의 챗 템플릿 형식
# <|begin_of_text|>
# <|start_header_id|>system<|end_header_id|>시스템 프롬프트<|eot_id|>
# <|start_header_id|>user<|end_header_id|>유저 프롬프트<|eot_id|>
# <|start_header_id|>assistant<|end_header_id|>거대 언어 모델이 해야하는 답변
# <|eot_id|>

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
peft_config = LoraConfig(
    lora_alpha=32,      # LoRA의 alpha, 스케일링 계수 설정. LoRA 가중치의 모델 출력 영향도를 조정
    lora_dropout=0.1,   # LoRA 적용시 드롭아웃 비율 설정. 학습 동안 10%의 뉴런을 랜덤하게 비활성화하여 과적합 방지
    r=8,                # LoRA 랭크, LoRA가 학습할 저차원 공간의 크기를 설정
    bias="none",        # LoRA 적용시 편향 설정. none이면 편향이 LoRA에 의해 조정되지 않음. ["none", "all", "lora_only"]
    target_modules=["q_proj", "v_proj"], # LoRA를 적용할 레이어, 여기서는 Self Attention의 W^q, W^v 에 적용
    task_type="CAUSAL_LM",  # LoRA가 적용되는 작업의 유형. CAUSAL_LM은 시퀀스 생성 작업 (Causal Language Modeling)
)

if torch.cuda.is_available():
    ## QLoRA 튜닝 설정
    ### BitsAndBytesConfig 클래스를 통해 양자화 설정 정의
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    ## 모델 및 토크나이저 로드
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, quantization_config=bnb_config)

    ## 모델을 4bit 학습을 위한 상태로 준비
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)
else:
    ## 맥북(mps)는 bitsandbytes를 지원하지 않음
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

    # !! 설정되지 않은 중요 옵션: max_length (기본값 1024)
    ## SFTConfig의 max_length는 학습 시 한 샘플의 최대 토큰 길이이며, 이보다 긴 문장은 뒤가 잘려나감
    ## 이 데이터셋을 실제로 토크나이즈해보면 길이 중앙값 1563, 평균 1635, 최대 5907 토큰으로,
    ## 991개 중 934개(94.2%)가 1024를 초과함 -> 기본값 그대로 두면 대부분의 샘플에서 정답(assistant 응답)이 통째로 잘림
    ## 학습을 실제로 돌리기 전에 max_length=2048(84.7% 커버) 또는 4096(99.7% 커버) 지정을 권장
    ## (단, 길이를 늘리면 GPU 메모리 사용량이 함께 늘어나므로 배치 크기와 함께 조절 필요)
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
if torch.cuda.is_available():
    # QLoRA 적용할 경우
    trainer = SFTTrainer(
        model=model,
        args=args, # SFTConfig
        train_dataset=train_dataset,
        data_collator=collate_fn
    )
else:
    # 일반 LoRA 적용할 경우, peft_config를 명시적으로 넘김
    trainer = SFTTrainer(
        model=model,
        args=args, # SFTConfig
        train_dataset=train_dataset,
        data_collator=collate_fn,
        peft_config=peft_config
    )

# 학습 시작
trainer.train() # 모델이 자동으로 허브와 output_dir에 저장됨

# 모델 저장
#trainer.save_model() # 최종 모델을 저장

print("\n=============================================")

######################################################################
# 모델 학습

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
