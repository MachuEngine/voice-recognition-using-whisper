# voice-recognition-using-whisper

### 프로젝트 개요
이 프로젝트는 **Whisper 모델의 파인튜닝**을 통해 금융, 경제 도메인에 특화된 Speech To Text 모델을 구현하는 것을 목표로 합니다.

### 프로젝트 동기
사전 학습된 음성인식 모델은 범용적으로 사용되도록 만들어져 특정 도메인의 전문 용어를 정확하게 인식하는 데에 한계를 가지고 있습니다. 특히 경제 용어의 경우 영어와 혼용하여 사용되는 경우가 많아 경제 도메인에 특화된 음성인식 모델의 필요성을 느꼈습니다.

---

### 데이터셋
이 프로젝트는 AI HUB에서 제공하는 **'뉴스 대본 및 앵커 음성 데이터'**를 사용하였습니다.

#### 데이터셋 개요
* 언론에 보도된 뉴스기사, 각 분야(정치, 경제, 사회, 문화, 국제, 지역, 스포츠, IT과학)별 전직/현직 아나운서, 아나운서 교육생들이 뉴스를 보도하는 음성 데이터 1,132시간을 포함합니다.

| 구분 | 정치 | 경제 | 사회 | 문화 | 국제 | 지역 | 스포츠 | IT·과학 | 합계 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 구축 목표 비율% | 15% | 20% | 10% | 10% | 10% | 15% | 10% | 10% | 100% |

* URL: https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=71557

#### 데이터셋 구성

데이터셋은 아래와 같이 구성되어 있습니다.

```
data_root_path/
├── Training/
│ ├── 01_SourceData/TS/
│ │ ├── file_a.wav
│ │ └── ...
│ └── 02_LabeledData/TL/
│ ├── file_b.json
│ └── ...
│
└── Validation/
    ├── 01_SourceData/VS/
    │ ├── file_c.wav
    │ └── ...
    └── 02_LabeledData/VL/
        ├── file_d.json
        └── ...
```

그리고 레이블링 데이터는 아래와 같이 JSON 형식으로 오디오 데이터 정보를 포함하고 있습니다.

```json
{
  "script": {
    "id": "YTNEC057",
    "url": "http://www.ytn.co.kr/_ln/0102_201801091444153396",
    "title": "최종구 `코스닥 활성화 위해 상장요건 완화·펀드 조성`",
    "press": "YTN",
    "press_field": "경제",
    "press_date": "20180109",
    "index": 2,
    "text": "최 위원장은 오늘 코스닥 시장 활성화 현장간담회에서 이 같은 내용을 담은 코스닥 활성화 방안을 공개했습니다.",
    "sentence_type": "작문형",
    "keyword": "코스닥,상장 요건,코스닥 상장,코스닥 활성화,위원장,코스닥 기업,최종구 코스닥,최종구 코스닥 활성화,활성화,코스닥 시장 활성화"
  },
  "speaker": {
    "id": "SPK054",
    "age": "20대",
    "sex": "남성",
    "job": "아나운서준비생"
  },
  "file_information": {
    "audio_format": "44100 Hz 16bit PCM",
    "utterance_start": "0.445",
    "utterance_end": "7.467",
    "audio_duration": "7.929"
  }
}
```

#### 오디오 데이터 (WAV)

오디오 데이터는 JSON의 **speaker** 필드 내 **id** 값과 **script** 필드 내 **id** 값의 조합 이름의 디렉토리에 .wav 파일로 저장되어 있습니다.

* 예를 들어, 위 JSON 기준으로 `data['speaker']['id']`는 `SPK054`이고, `data['script']['id']`는 `YTNEC057`입니다.
* **speaker**의 **sex**에 따라 데이터파일명의 **F** 또는 **M**이 결정되고, **script**의 **index**에 따라 성별 뒤에 오는 세 자리 수가 결정됩니다.

최종적으로 위 JSON의 파일명은 `SPK054YTNEC057M002.json`이 되고, 이에 매칭되는 WAV 파일의 이름은 `SPK054YTNEC057M002.wav`가 됩니다.

이 규칙에 따라 JSON 파일에서 필요한 정보를 파싱하고 해당 정보를 통해 매칭되는 오디오 파일을 찾을 수 있습니다.

예시:
* `...\data\open_data\Validation\02_LabeledData\VL\SPK054\SPK054YTNEC057\SPK054YTNEC057M001.json`
* `...\data\open_data\Validation\02_LabeledData\VL\SPK054\SPK054YTNEC057\SPK054YTNEC057M002.json`
* ...

### 사전 학습된 모델, 프로세서 및 학습 도구
* **Hugging Face Transformers**: Hugging Face에서 제공하는 트랜스포머 라이브러리를 이용하여 Whisper 모델을 가져와 진행하였습니다.
```py
from transformers import WhisperProcessor, WhisperForConditionalGeneration
```
: Hugging Face에서 제공하는 라이브러리입니다. `WhisperProcessor`는 오디오와 텍스트 데이터를 모델이 이해할 수 있는 형태로 변환하고, `WhisperForConditionalGeneration`은 실제 음성 인식 모델입니다.
```py
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
```
: 모델 학습에 필요한 도구입니다. `Seq2SeqTrainingArguments`는 학습 설정을 정의하고, `Seq2SeqTrainer`는 실제 학습 과정을 관리합니다.

### 파인튜닝 과정

데이터셋은 여러 도메인들이 혼합된 상태여서, 레이블 데이터에서 **'경제' 도메인만 필터링**해야 합니다.

이후 매칭되는 오디오 데이터도 함께 가져오는 전처리 과정을 거쳐 학습할 데이터셋을 구성합니다. 경제 도메인만 필터링한 '오디오-스크립트' 데이터셋이 구성되면 해당 데이터로 학습을 진행합니다.

### 성능 비교 검토

파인튜닝 이전과 이후 경제 도메인에 있어서 특화된 성능 향상이 이뤄졌는지 확인하기 위해 비교 검토를 진행했습니다.

#### 파인튜닝 전 평가 성능 결과

```
사용 장치: cpu

--- 검증 데이터 로드 시작 ---
검증 데이터 오디오 경로: C:/Users/Admin/Desktop/github/voice-recognition-using-whisper/data/news_scripts_and_speech_data/open_data\Validation\01_SourceData\VS
검증 데이터 JSON 경로: C:/Users/Admin/Desktop/github/voice-recognition-using-whisper/data/news_scripts_and_speech_data/open_data\Validation\02_LabeledData\VL
총 1606개의 Validation용 경제 도메인 데이터를 찾았습니다.
                                                                path \
0  C:/Users/Admin/Desktop/github/voice-recognitio...
1  C:/Users/Admin/Desktop/github/voice-recognitio...
2  C:/Users/Admin/Desktop/github/voice-recognitio...
3  C:/Users/Admin/Desktop/github/voice-recognitio...
4  C:/Users/Admin/Desktop/github/voice-recognitio...

                                                                transcription
0  대부분 온라인 쇼핑몰에서 가공, 신선식품이나 일용 잡화를 판매할 때 단위 가격을 표...
1  이에 따라 소비자의 합리적인 선택을 위해 온라인 쇼핑몰에서도 오프라인 매장처럼 단위...
2  한국소비자원이 대형 마트 쇼핑몰 (3)/(세) 곳과 오픈 마켓 (8)/(여덟) 곳 ...
3  대형 마트 등 오프라인 매장은 판매 가격만으로는 가격 비교가 어려운 (84)/(여든...
4  소비자원이 쇼핑몰별로 (79~82)/(칠십 구 에서 팔십 이) 개 품목 각 (20)...

검증 데이터 전처리 중입니다...
필터링 후 검증 데이터셋 크기: 1606
검증 데이터 전처리 완료.

최종 모델을 평가합니다...
최종 WER: 42.24%
```

#### 파인튜닝 후 평가 성능 결과
```
(여기에 파인튜닝 후 결과가 들어갈 예정입니다.)
```