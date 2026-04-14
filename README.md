# 🚀 Beyond Base Models: Fine-Tuning KoGPT2 for Korean Movie Reviews
### 범용 모델의 한계를 넘어: KoGPT2 파인튜닝과 토크나이저 무결성 확보 사례 연구

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)
![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97-Hugging%20Face-yellow.svg)
![Transformers](https://img.shields.io/badge/Transformers-Latest-orange.svg)

---

## 📌 프로젝트 요약 (Project Overview)
본 프로젝트는 단순 범용 텍스트 생성 혹은 어색한 직역체 문장을 내뱉던 **기본 언어 모델의 한계를 극복**하고, **한국어 영화 리뷰 도메인에 특화된 텍스트 생성 파이프라인**을 구축한 사례입니다. SKT의 사전 학습 한국어 모델인 **KoGPT2(`skt/kogpt2-base-v2`)** 베이스라인에 NSMC 데이터를 파인튜닝하여, 기존 모델 대비 **자연스러운 구어체 표현력과 도메인 이해도**를 대폭 향상시켰습니다.

특히, 토크나이저(Tokenizer) 로드 방식 차이로 인해 발생하는 내부 인덱스 어긋남 현상(CUDA Assert Error 및 한글 인코딩 깨짐)의 원인을 분석하고, 이를 구조적으로 해결하는 트러블슈팅 과정을 중점적으로 다룹니다.

---

## 🎯 핵심 목표 (Motivation)
| 구분 | 세부 내용 |
| :--- | :--- |
| **성능 극대화** | 범용 지식만 가진 Base 모델의 어색한 직역체를 극복하고, 영화 리뷰 분야에서 자연스러운 구어체 생성을 구사하도록 모델 최적화 |
| **도메인 특화 파인튜닝** | KoGPT2 모델에 NSMC(네이버 영화 리뷰) 데이터를 주입하여 구어체 중심의 문맥을 생성하도록 Causal LM 재학습 |
| **Tokenizer 무결성 확보** | 한국어 자연어 처리 모델에서 빈번히 발생하는 토큰 인덱스 불일치 및 한글 깨짐 문제를 디버깅하고 안정적인 인코딩 규격 확립 |
| **디코딩 전략 최적화** | 생성 파라미터(Temperature, Top-p, Repetition Penalty) 튜닝을 통해 텍스트의 일관성 제어 및 반복 생성 억제 |

---

## 📂 프로젝트 구조 (Project Structure)
```text
📂 kogpt2-korean-finetuning
├── 📄 main_finetuning.py        # 모델 학습 및 토크나이저 최적화 스크립트 (핵심 로직)
├── 📄 app.py                    # 학습된 모델 시연용 Gradio 웹 인터페이스
├── 📄 requirements.txt          # 프로젝트 의존성 패키지 리스트
├── 📄 README.md                 # 프로젝트 개요 및 결과 보고서
├── 📄 training.log              # 학습 과정 및 지표 모니터링 로그
└── 📁 kogpt2-korean-finetuned/  # 파인튜닝 가중치 (대용량 파일, GitHub 제외 및 Hugging Face 연동)
```

---

## 🛠️ 주요 알고리즘 및 기술적 구현 (Technical Implementation)

### 1. Tokenization & Data Preprocessing
* **정밀한 토크나이징 제어**: 일반적인 `AutoTokenizer` 대신 `PreTrainedTokenizerFast`를 명시하여 기본 토크나이저 설정이 라이브러리의 범용 로직에 의해 변형되는 것을 차단했습니다.
* **Data Pipelining**: `datasets` 라이브러리를 활용하여 원본 NSMC 데이터를 필터링하고 병렬로 Tokenize 처리하여 Causal LM 학습 포맷을 구성했습니다.

### 2. Causal LM Fine-tuning Architecture
* **자원 효율적 학습 설정**: 원본 데이터를 3만 건(Train), 3천 건(Eval)으로 샘플링하고 `gradient_accumulation_steps=2`, Mixed Precision(`fp16=True`)을 적용하여 GPU 환경에서 학습 효율을 극대화했습니다.
* **도메인 적응 (Domain Adaptation)**: 뉴스/위키백과 위주의 기존 가중치를 영화 리뷰 특유의 감성과 인터넷 구어체 텍스트로 재배치하여 질적 성능을 차별화했습니다.

### 3. Generation Decoding Strategy
* **Repetition Penalty (1.2)**: 텍스트 생성 모델의 고질적인 구절 반복 문제를 제어.
* **Top-P (0.92)**: 확률 분포 하위 단어를 잘라내어 문맥 붕괴 예방.
* **Temperature (0.8)**: 안전성과 창의성 간의 밸런스를 채택하여 감성적 표현력 확보.

---

## 🚀 트러블슈팅: 토크나이저 무결성 확보 (Troubleshooting)

**[문제 현상] 한글 인코딩 깨짐 및 `CUDA device-side assert triggered` 에러**
KoGPT2 모델 파인튜닝 중 지속적인 CUDA 메모리 참조 에러가 발생하였으며, 간헐적으로 출력이 외계어로 붕괴되는 현상을 겪었습니다.

**[원인 분석]**
범용 클래스인 `AutoTokenizer`가 KoGPT2를 초기화할 때, 기본 어휘 사전 크기(`51,200`)를 넘어 패딩을 위한 새로운 토큰을 암시적으로 추가하여 임베딩 크기를 `51,201`로 변형시켰습니다. 이로 인해 모델 학습 시 전체 인덱스 맵핑이 1칸씩 밀리면서, 올바른 벡터가 출력되어도 잘못된 글자로 강제 디코드되는 현상이 발생했습니다.

**[해결 방안]**
파생 토큰 생성을 방지하기 위해 `PreTrainedTokenizerFast`를 명시적으로 선언하고, `bos_token`, `eos_token`, `pad_token`을 본래 KoGPT2 체계에 맞게 하드코딩했습니다. 불필요한 `model.resize_token_embeddings()` 과정을 해제하여 원래 모델의 구조(Vocab Size 51,200)에 완벽히 종속되도록 코드를 수정함으로써 무결점의 한국어 출력을 확보했습니다.

---

## 🔥 라이브 데모 (Live Demo)
> 토크나이저가 최적화된 파인튜닝 모델을 웹에서 직접 테스트해 볼 수 있습니다.

---

## 🤖 파인튜닝 결과물 대조 (Inference Comparison)

목적 지향적 학습을 거친 **KoGPT2-Finetuned 모델**과 일반 베이스 모델의 차이입니다.

* **Prompt**: `"이 영화의 결말은"`
* **After (Fine-tuned KoGPT2)**: *"이 영화의 결말은 정말 최고였다... 정말 최고의 작품이었다!!!ㅎㅎㅎ 강추!! ^_^; 너무 기대하고 본 영화였습니다! 강추~♥♡♡♡!!♥♡!♥.^ 아놔여 ♡♡ 강추! 강츄♥♡아저씨강츄예요 ㅠ 아놔요 ㅜ ㅜ♡♡ 강추하셔서 감사합니다이~♡♡♥♡ 강츄~ 강츄, 강츄강츄에 모두들 좋아합니다 강츄!!"*

**💡 결과 해석**: 리뷰 데이터 특유의 인터넷 속어, 이모티콘, 전형적인 구어체 문맥이 모델에 성공적으로 흡수되었으며, 인덱스 밀림에 의한 인코딩 에러가 완벽하게 통제된 것을 확인할 수 있습니다.

---

## 💡 회고 (Retrospective)
단순히 API를 호출하는 것을 넘어, 언어 모델의 근간이 되는 토크나이저의 내부 행렬 크기(Vocabulary Size) 통제와 인덱싱 과정을 깊이 있게 이해할 수 있는 실습이었습니다. 트러블슈팅 과정을 통해 데이터의 '바이트와 인덱스 로직'을 제어하는 방법을 습득했으며, 양질의 도메인 특화 데이터가 텍스트 생성 품질에 미치는 긍정적인 영향을 시각적으로 명확히 확인할 수 있었습니다.

---

## 🔥 **[LIVE DEMO] 방금 탄생한 가장 완벽한 한글 영화 리뷰 생성기를 직접 체험해보세요!**
[![Test in Spaces](https://huggingface.co/datasets/huggingface/badges/resolve/main/deploy-to-spaces-lg-dark.svg)](https://huggingface.co/spaces/AD-Styles/kogpt2-korean-finetuning)
> **☝️ 위 버튼을 클릭하시면, 어휘 사전 인덱스까지 완벽하게 최적화된 파인튜닝 모델이 실제 가동 중인 Hugging Face 웹 서버로 즉시 연결됩니다.**
> **URL**: [https://huggingface.co/spaces/AD-Styles/kogpt2-korean-finetuning](https://huggingface.co/spaces/AD-Styles/kogpt2-korean-finetuning)

---

## 🤖 파인튜닝 결과물 대조 (Inference Comparison)

파인튜닝 전, 일반적인 영어 기반의 GPT-2 모델과 우리가 한국어 영화 리뷰로 목적 지향적 학습을 거친 **KoGPT2-Finetuned 모델**의 답변 차이입니다:

#### Test Case 1.
*   **Prompt**: `"배우의 연기가"`
*   **Before (English Base GPT)**: *(무의미한 한자 및 외계어 나열, 인코딩 붕괴 상태 발생)*
*   **After (Fine-tuned KoGPT2)**: *"배우의 연기가 더 좋았습니다.. 특히 연기가 너무 잘 어울렸어요. 영화도 이쁘고.. 연기하는게 정말 좋아요!!! 그리고 ᄒᄒᄒ!!♥♥♥♡ 짱!!~!!"*

#### Test Case 2.
*   **Prompt**: `"이 영화의 결말은"`
*   **After (Fine-tuned KoGPT2)**: *"이 영화의 결말은 재밋게 봤는데..아니 그걸 영화라고??;;;그냥 볼거리.그리고 주인공과 같이 살수도 없는거다..나쁜 여자들이랑 함께하는 장면은 너무 실망스럽다 진짜...뭐여?; 아휴 개연성도 없고...;"*

**💡 결과 해석**: 리뷰 데이터 특유의 인터넷 속어(ᄒᄒᄒ), 이모티콘(♥♥♥, ;;;), 그리고 영화 리뷰 데이터의 전형적인 구어체 문맥이 모델에 완벽하게 흡수된 것을 확인할 수 있습니다. 인코딩 에러를 완벽하게 통제하였습니다.

---

## 💡 회고 (Retrospective)
이 실습을 통해 언어 모델의 근간이 되는 '토크나이저'가 얼마나 민감한 부품인지 체감할 수 있었습니다. 특히, 단순히 라이브러리의 경고 메시지를 숨기거나 API를 끌어다 쓰는 수준을 넘어 **내부의 행렬 크기(Vocabulary Size) 통제 구조와 Tokenizer 인덱싱 과정**을 깊숙이 파헤치고, 트러블슈팅의 핵심 원리를 관통해낸 경험은 일반적인 튜토리얼에서는 결코 얻을 수 없는 인사이트였습니다.
단순한 텍스트 훈련을 넘어서, 인공지능이 데이터를 다루는 '바이트와 인덱스 로직' 자체의 지배력을 기르게 된 뜻깊은 포트폴리오로 자리매김했습니다.

  무엇보다 가장 큰 성과는 **모델의 성능 변화를 시각적, 정성적으로 바로 체감할 수 있었다**는 점입니다. 단순 영문을 나열하거나 엉뚱한 뉴스 문맥을 읊던 Base 언어 모델이, 파인튜닝을 거친 후 영화에 대한 인간의 감정을 그대로 모방한 한국어 문장을 술술 뱉어내는 것을 보며 **'양질의 도메인 특화 데이터가 텍스트 생성 품질에 미치는 압도적 폭발력'**을 증명해낼 수 있었습니다. 
