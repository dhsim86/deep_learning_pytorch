# kanana2 1.3B 모델 파인튜닝 결과 비교

## 베이스 모델

```json
 response:
{
    "is_stock_related": False,
    "summary": "글로벌 비즈 가트너 올해 전세계 스마트폰 판매량 7% 감소 전망"
}
 label:
{
    'is_stock_related': True, 
    'negative_impact_stocks': ['애플', '엔비디아', 'TSMC', '텐센트', '바이트댄스'], 
    'negative_keywords': ['스마트폰 판매 감소', '인력 감축', '비용 절감', '경제 둔화'], 
    'positive_impact_stocks': [], 
    'positive_keywords': [], 
    'reason_for_negative_impact': '가트너의 보고서에 따르면 스마트폰 판매량의 감소는 애플과 같은 제조업체뿐만 아니라 엔비디아와 TSMC 같은 반도체 업체에게 부정적인 영향을 미칠 것으로 보입니다. 또한, 텐센트와 바이트댄스가 인력을 추가 감원할 계획을 발표함에 따라 이들의 주가에 부정적인 영향이 예상됩니다.', 
    'reason_for_positive_impact': '', 
    'summary': '가트너는 올해 전세계 스마트폰 판매량이 7% 감소할 것이라고 전망하며, 애플과 반도체 업체들이 영향을 받을 것으로 보입니다. 한편, 텐센트와 바이트댄스는 하반기 대규모 감원을 계획하고 있어, 글로벌 빅테크 기업들이 경제 둔화와 비용 절감 압박에 직면하고 있습니다.'
}
--------------------------------------------------
 response:
{
    "is_stock_related": True,
    "summary": "야놀자와 포커스미디어가 동네가게 오래함께 캠페인을 진행하여 지역 소상공인 발굴 및 맞춤형 광고 지원, 총 14억 원 광고 제작 및 송출 비용 전액 부담, 서울시 노원구 동작구 등 2개월 간 광고 송출 예정. 야놀자 측은 소상공인 매출 증대 및 지역사회 상생에 기여 기대.",
    "positive_impact_stocks": ["야놀자", "포커스미디어"],
    "reason_for_positive_impact": "지역 소상공인 인지도를 높이고 매출 증대에 기여하기 위해 캠페인 진행",
    "positive_keywords": ["소상공인", "광고", "지역경제", "매출 증대", "상생"],
    "negative_impact_stocks": [],
    "reason_for_negative_impact": "",
    "negative_keywords": [],
    "summary": "야놀자와 포커스미디어가 동네가게 오래함께 캠페인을 통해 지역 소상공인 발굴 및 맞춤형 광고 지원, 총 14억 원 광고 제작 및 송출 비용 전액 부담, 서울시 노원구 동작구 등 2개월 간 광고 송출 예정. 야놀자 측은 소상공인 매출 증대 및 지역사회 상생에 기여 기대."
}
 label:
{
    'is_stock_related': True, 
    'negative_impact_stocks': [], 
    'negative_keywords': [], 
    'positive_impact_stocks': ['야놀자'], 
    'positive_keywords': ['야놀자', '포커스미디어', '소상공인 지원', '광고 캠페인', '지역경제 활성화'], 
    'reason_for_negative_impact': '', 
    'reason_for_positive_impact': '야놀자가 포커스미디어와 협력하여 지역 소상공인을 지원하는 캠페인을 진행함으로써 지역사회와의 상생 및 경제 활성화에 기여하며, 이는 브랜드 이미지와 매출 성장에 긍정적인 영향을 미칠 수 있다.', 
    'summary': "야놀자가 포커스미디어와 함께 지역 소상공인을 지원하기 위한 '동네가게 오래함께' 캠페인을 진행하여 소상공인들의 인지도와 매출 증가에 기여할 예정이다. 캠페인은 서울시 노원구 동작구를 시작으로 점차 대상 범위를 확대할 계획이다."
}
--------------------------------------------------
 response:
{
    "is_stock_related": True,
    "summary": "삼성바이오로직스가 미국 제약기업 MSD와 2768억2938만원 규모의 의약품 위탁생산 공급계약을 체결했다.",
    "positive_impact_stocks": ["삼성바이오로직스"],
    "reason_for_positive_impact": "이번 계약은 삼성바이오로직스의 글로벌 위탁생산 역량 강화 및 매출 확대에 긍정적인 영향을 미칠 것으로 예상된다.",
    "positive_keywords": ["위탁생산", "글로벌", "매출", "성장"],
    "negative_impact_stocks": [],
    "reason_for_negative_impact": "",
    "negative_keywords": [],
    "is_stock_related": True,
    "summary": "삼성바이오로직스가 미국 제약기업 MSD와 2768억2938만원 규모의 의약품 위탁생산 공급계약을 체결했다."
}
 label:
{
    'is_stock_related': True, 
    'negative_impact_stocks': [], 
    'negative_keywords': [], 
    'positive_impact_stocks': ['삼성바이오로직스'], 
    'positive_keywords': ['위탁생산계약', 'MSD', '매출 증대'], 
    'reason_for_negative_impact': '', 
    'reason_for_positive_impact': '삼성바이오로직스가 미국 제약기업인 MSD와 2768억원 규모의 위탁생산계약을 체결하였으므로, 이는 회사의 매출 증대에 긍정적 영향을 미칠 수 있다.', 
    'summary': '삼성바이오로직스가 미국 제약기업 MSD와 2768억원 규모의 의약품 위탁생산 계약을 체결하였으며, 이는 회사의 매출 대비 상당한 규모로, 향후 매출 증가가 기대된다.'
}
--------------------------------------------------
 response:
{
    "is_stock_related": True,
    "summary": "피트니스 시장 더 커질 것 LG전자·SM엔터 홈트 앱 내놓는다",
    "positive_impact_stocks": ["LG전자", "SM엔터테인먼트"],
    "reason_for_positive_impact": "피트니스 시장 성장 기대, LG전자와 SM엔터테인먼트의 홈트 앱 출시로 디지털 피트니스 산업 활성화",
    "positive_keywords": ["피트니스 시장", "LG전자", "SM엔터테인먼트", "홈트 앱", "디지털 피트니스"],
    "negative_impact_stocks": [],
    "reason_for_negative_impact": "",
    "negative_keywords": [],
    "is_stock_related": True,
    "summary": "피트니스 시장 더 커질 것 LG전자·SM엔터 홈트 앱 내놓는다"}
 label:
{
    'is_stock_related': True, 
    'negative_impact_stocks': [], 
    'negative_keywords': [], 
    'positive_impact_stocks': ['LG전자', 'SM엔터테인먼트'], 
    'positive_keywords': ['피트니스 캔디', '디지털 피트니스', 'LG전자', 'SM엔터테인먼트', 'K POP', '메타버스'], 
    'reason_for_negative_impact': '', 
    'reason_for_positive_impact': "LG전자와 SM엔터테인먼트가 공동으로 피트니스 콘텐츠 브랜드 '피트니스 캔디'를 출시하며 디지털 피트니스 시장에 진출하여, 양사의 디지털 기술력과 K POP 콘텐츠를 바탕으로 새로운 수익 창출과 시장 기회를 얻게 될 가능성이 높기 때문입니다.", 
    'summary': "LG전자와 SM엔터테인먼트는 디지털 피트니스 콘텐츠 합작 브랜드 '피트니스 캔디'를 발표하며, 피트니스 시장의 확장을 기대하고 있습니다. 이 플랫폼은 개인 맞춤형 커뮤니티 서비스와 K POP 콘텐츠를 결합한 디지털 피트니스 트렌드를 지향합니다."
}
--------------------------------------------------
 response:
{
    "is_stock_related": true,
    "positive_impact_stocks": ["경동나비엔", "나비엔 청정환기시스템"],
    "reason_for_positive_impact": "장마철에도 창문을 열지 않고도 실내 공기질을 효과적으로 관리할 수 있어, 미세먼지 및 초미세먼지로부터 가족의 건강을 지킬 수 있다.",
    "positive_keywords": ["미세먼지", "초미세먼지", "공기청정기", "환기", "건강", "쾌적", "가족", "실내공기질"],
    "negative_impact_stocks": [],
    "reason_for_negative_impact": "",
    "summary": "창문을 열지 않고도 장마철에도 실내 공기질을 관리할 수 있는 경동나비엔 청정환기시스템이 미세먼지, 초미세먼지, 바이러스 등으로부터 가족의 건강을 지키는 데 효과적임을 입증했다."
}
 label:
{
    'is_stock_related': True, 
    'negative_impact_stocks': [], 
    'negative_keywords': [], 
    'positive_impact_stocks': ['경동나비엔'], 
    'positive_keywords': ['경동나비엔', '청정환기시스템', '미세먼지', 'UV LED 모듈', '공기청정'], 
    'reason_for_negative_impact': '', 
    'reason_for_positive_impact': '경동나비엔의 청정환기시스템이 미세먼지 문제와 장마철 환기 문제를 동시에 해결하는 혁신적인 솔루션으로 소개되면서, 해당 시스템에 대한 수요 증가가 예상된다.', 
    'summary': '경동나비엔의 청정환기시스템은 실내 공기질 개선 솔루션으로, 미세먼지 문제와 장마철 환기 어려움을 극복할 수 있는 제품으로 소개되었다. 이 시스템은 실내외의 공기를 깨끗하게 교환하며, 주방에서 발생하는 오염물질 확산도 방지한다.'
}
```

## 파인튜닝된 모델


```json
 response:
{
    'is_stock_related': True, 
    'negative_impact_stocks': ['애플', '엔비디아', 'TSMC'], 
    'negative_keywords': ['판매 감소', '수요 부진', '규제'], 
    'positive_impact_stocks': [], 
    'positive_keywords': [], 
    'reason_for_negative_impact': '글로벌 스마트폰 판매량이 7% 감소할 것으로 전망되며, 이는 애플과 같은 스마트폰 제조사 및 엔비디아, TSMC와 같은 반도체 업체에 부정적인 영향을 미칠 수 있습니다.', 
    'reason_for_positive_impact': '', 
    'summary': '글로벌 스마트폰 판매량이 7% 감소할 것으로 전망되며, 이는 애플과 같은 스마트폰 제조사 및 엔비디아, TSMC와 같은 반도체 업체에 부정적인 영향을 미칠 수 있습니다. 또한, EU가 가상자산 돈세탁을 막기 위한 규제 방안을 잠정 합의하여 관련 기업들에게 개인정보 보호 및 거래 규칙 준수를 요구할 수 있습니다.'
}
 label:
{
    'is_stock_related': True, 
    'negative_impact_stocks': ['애플', '엔비디아', 'TSMC', '텐센트', '바이트댄스'], 
    'negative_keywords': ['스마트폰 판매 감소', '인력 감축', '비용 절감', '경제 둔화'], 
    'positive_impact_stocks': [], 
    'positive_keywords': [], 
    'reason_for_negative_impact': '가트너의 보고서에 따르면 스마트폰 판매량의 감소는 애플과 같은 제조업체뿐만 아니라 엔비디아와 TSMC 같은 반도체 업체에게 부정적인 영향을 미칠 것으로 보입니다. 또한, 텐센트와 바이트댄스가 인력을 추가 감원할 계획을 발표함에 따라 이들의 주가에 부정적인 영향이 예상됩니다.', 
    'reason_for_positive_impact': '', 
    'summary': '가트너는 올해 전세계 스마트폰 판매량이 7% 감소할 것이라고 전망하며, 애플과 반도체 업체들이 영향을 받을 것으로 보입니다. 한편, 텐센트와 바이트댄스는 하반기 대규모 감원을 계획하고 있어, 글로벌 빅테크 기업들이 경제 둔화와 비용 절감 압박에 직면하고 있습니다.'
}
--------------------------------------------------
 response:
{
    'is_stock_related': True, 
    'negative_impact_stocks': [], 
    'negative_keywords': [], 
    'positive_impact_stocks': ['야놀자', '포커스미디어'], 
    'positive_keywords': ['야놀자', '포커스미디어', '동네가게 오래함께 캠페인', '광고 제작 및 송출 비용', '소상공인 발굴'], 'reason_for_negative_impact': '', 
    'reason_for_positive_impact': '야놀자와 포커스미디어는 지역 내 소상공인 지원을 위한 광고 제작 및 송출 비용을 전액 부담함으로써 지역 경제 활성화에 기여할 것으로 기대되며, 이는 두 회사의 긍정적인 평판과 브랜드 이미지 강화에 도움이 될 수 있다.', 
    'summary': '야놀자와 포커스미디어는 지역 내 우수 소상공인을 발굴하고 맞춤형 광고를 제작하여 지원하는 '동네가게 오래함께' 캠페인을 진행한다. 총 14억 원 규모의 광고 제작 및 송출 비용을 양사가 전액 부담하며, 첫 광고는 서울시 노원구 동작구에서 시작해 연말까지 확대될 예정이다.'
}
 label:
{
    'is_stock_related': True, 
    'negative_impact_stocks': [], 
    'negative_keywords': [], 
    'positive_impact_stocks': ['야놀자'], 
    'positive_keywords': ['야놀자', '포커스미디어', '소상공인 지원', '광고 캠페인', '지역경제 활성화'], 
    'reason_for_negative_impact': '', 
    'reason_for_positive_impact': '야놀자가 포커스미디어와 협력하여 지역 소상공인을 지원하는 캠페인을 진행함으로써 지역사회와의 상생 및 경제 활성화에 기여하며, 이는 브랜드 이미지와 매출 성장에 긍정적인 영향을 미칠 수 있다.', 
    'summary': "야놀자가 포커스미디어와 함께 지역 소상공인을 지원하기 위한 '동네가게 오래함께' 캠페인을 진행하여 소상공인들의 인지도와 매출 증가에 기여할 예정이다. 캠페인은 서울시 노원구 동작구를 시작으로 점차 대상 범위를 확대할 계획이다."
}
--------------------------------------------------
 response:
{
    'is_stock_related': True, 
    'negative_impact_stocks': [], 
    'negative_keywords': [], 
    'positive_impact_stocks': ['삼성바이오로직스'], 
    'positive_keywords': ['삼성바이오로직스', '위탁생산 계약', 'MSD', '공급계약'], 
    'reason_for_negative_impact': '', 
    'reason_for_positive_impact': '삼성바이오로직스가 미국 제약기업 MSD와 대규모 위탁생산 계약을 체결함으로써 안정적인 매출 기반을 확보하고, 향후 성장 가능성을 높일 수 있기 때문이다.', 
    'summary': '삼성바이오로직스가 미국 제약기업 MSD와 2768억2938만원 규모의 의약품 위탁생산 공급계약을 체결하여 안정적인 매출 기반을 확보하고 성장 가능성을 높였다.'
}
 label:
{
    'is_stock_related': True, 
    'negative_impact_stocks': [], 
    'negative_keywords': [], 
    'positive_impact_stocks': ['삼성바이오로직스'], 
    'positive_keywords': ['위탁생산계약', 'MSD', '매출 증대'], 
    'reason_for_negative_impact': '', 
    'reason_for_positive_impact': '삼성바이오로직스가 미국 제약기업인 MSD와 2768억원 규모의 위탁생산계약을 체결하였으므로, 이는 회사의 매출 증대에 긍정적 영향을 미칠 수 있다.', 
    'summary': '삼성바이오로직스가 미국 제약기업 MSD와 2768억원 규모의 의약품 위탁생산 계약을 체결하였으며, 이는 회사의 매출 대비 상당한 규모로, 향후 매출 증가가 기대된다.'
}
--------------------------------------------------
 response:
{
    'is_stock_related': True, 
    'negative_impact_stocks': [], 
    'negative_keywords': [], 
    'positive_impact_stocks': ['LG전자', 'SM엔터테인먼트'], 
    'positive_keywords': ['디지털 피트니스 콘텐츠', '합작 브랜드', 'K POP', '스마트 가전', '고객 데이터 경영'], 
    'reason_for_negative_impact': '', 
    'reason_for_positive_impact': 'LG전자와 SM엔터테인먼트가 합작하여 새로운 디지털 피트니스 콘텐츠 브랜드 '피트니스 캔디'를 발표하며, 이는 두 회사의 기술력과 브랜드 가치를 결합하여 새로운 성장 기회를 제공할 것으로 예상됩니다.', 
    'summary': 'LG전자와 SM엔터테인먼트가 합작하여 새로운 디지털 피트니스 콘텐츠 브랜드 '피트니스 캔디'를 발표하며, 이는 두 회사의 기술력과 브랜드 가치를 결합하여 새로운 성장 기회를 제공할 것으로 예상됩니다.'
}
 label:
{
    'is_stock_related': True, 
    'negative_impact_stocks': [], 
    'negative_keywords': [], 
    'positive_impact_stocks': ['LG전자', 'SM엔터테인먼트'], 
    'positive_keywords': ['피트니스 캔디', '디지털 피트니스', 'LG전자', 'SM엔터테인먼트', 'K POP', '메타버스'], 
    'reason_for_negative_impact': '', 
    'reason_for_positive_impact': "LG전자와 SM엔터테인먼트가 공동으로 피트니스 콘텐츠 브랜드 '피트니스 캔디'를 출시하며 디지털 피트니스 시장에 진출하여, 양사의 디지털 기술력과 K POP 콘텐츠를 바탕으로 새로운 수익 창출과 시장 기회를 얻게 될 가능성이 높기 때문입니다.", 
    'summary': "LG전자와 SM엔터테인먼트는 디지털 피트니스 콘텐츠 합작 브랜드 '피트니스 캔디'를 발표하며, 피트니스 시장의 확장을 기대하고 있습니다. 이 플랫폼은 개인 맞춤형 커뮤니티 서비스와 K POP 콘텐츠를 결합한 디지털 피트니스 트렌드를 지향합니다."
}
--------------------------------------------------
 response:
{
    'is_stock_related': True, 
    'positive_impact_stocks': ['경동나비엔'], 
    'positive_keywords': ['청정환기시스템', '공기청정기', '프리필터', '전자 집진필터', '탈취필터', '초미세 집진필터', 'UV LED 모듈', '나비엔 청정환기시스템 키친플러스'], 
    'negative_impact_stocks': [], 
    'negative_keywords': [], 
    'summary': '창문을 열지 않고도 실내 공기와 미세먼지를 정화할 수 있는 청정환기시스템이 주목받고 있으며, 특히 장마철에도 효과적으로 실내 공기를 관리할 수 있는 솔루션으로 주목받고 있다. 경동나비엔의 청정환기시스템은 다양한 필터와 UV LED 모듈을 통해 초미세먼지 및 부유 세균을 효과적으로 제거하며, 주방에서의 요리 매연 문제도 해결할 수 있는 솔루션을 제공한다.'
}
 label:
{
    'is_stock_related': True, 
    'negative_impact_stocks': [], 
    'negative_keywords': [], 
    'positive_impact_stocks': ['경동나비엔'], 
    'positive_keywords': ['경동나비엔', '청정환기시스템', '미세먼지', 'UV LED 모듈', '공기청정'], 
    'reason_for_negative_impact': '', 
    'reason_for_positive_impact': '경동나비엔의 청정환기시스템이 미세먼지 문제와 장마철 환기 문제를 동시에 해결하는 혁신적인 솔루션으로 소개되면서, 해당 시스템에 대한 수요 증가가 예상된다.', 
    'summary': '경동나비엔의 청정환기시스템은 실내 공기질 개선 솔루션으로, 미세먼지 문제와 장마철 환기 어려움을 극복할 수 있는 제품으로 소개되었다. 이 시스템은 실내외의 공기를 깨끗하게 교환하며, 주방에서 발생하는 오염물질 확산도 방지한다.'
}
```