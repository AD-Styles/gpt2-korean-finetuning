# 🚀 Beyond English GPT-2: Fine-Tuning KoGPT2 for Korean Movie Reviews
### 영문 GPT-2의 한계를 넘어: KoGPT2 파인튜닝과 토크나이저 무결성 확보 사례 연구

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)
![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97-Hugging%20Face-yellow.svg)
![Transformers](https://img.shields.io/badge/Transformers-Latest-orange.svg)

---

## 📌 프로젝트 요약 (Project Overview)
본 프로젝트는 언어의 장벽을 뛰어넘기 위한 실습으로, 단순 영문 텍스트 생성 혹은 어색한 직역체 문장을 내뱉던 **기본 언어 모델의 한계를 극복**하고 **명확한 한국어 도메인(영화 리뷰)에 특화된 텍스트 생성 파이프라인**을 구축한 기록입니다. SKT의 사전 학습 한국어 모델인 **KoGPT2(`skt/kogpt2-base-v2`)** 베이스라인에 NSMC 데이터를 파인튜닝하여, 기존 일반 GPT 모델 대비 **압도적으로 향상된 한국어 표현력과 도메인 이해도**를 이끌어내는 전 과정을 다룹니다.

특히, 토크나이저(Tokenizer) 로드 방식 차이로 인해 발생할 수 있는 내부 인덱스 어긋남 현상(CUDA Assert Error 및 한글 인코딩 깨짐 현상)의 치명적인 원인을 분석하고 이를 구조적으로 완벽히 해결하는 트러블슈팅 경험을 중점적으로 보여줍니다.

---

## 🎯 핵심 목표 (Motivation)
| 구분 | 세부 내용 |
| :--- | :--- |
| **성능 극대화 (Base vs Fine-tuned)** | 범용 지식만 가진 Base 모델의 어색한 직역체나 외계어 출력을 극복하고, 영화 리뷰 분야에서 압도적인 구어체 생성을 구사하도록 모델 지능을 업그레이드. |
| **도메인 특화 Fine-tuning** | 일반적인 한국어 모델(KoGPT2)에 NSMC(네이버 영화 리뷰) 데이터를 주입하여 구어체 중심의 자연스러운 영화 리뷰 문맥을 생성하도록 Causal LM을 재학습. |
| **Tokenizer 무결성 확보** | 한국어 자연어 처리 모델에서 빈번히 발생하는 토큰 인덱스 불일치 및 한글 깨짐 문제를 디버깅하고 완벽한 인코딩 규격 확립. |
| **디코딩 전략 최적화** | 생성 파라미터(Temperature, Top-p, Repetition Penalty 등) 튜닝을 통해 텍스트의 창의성과 일관성 제어, 반복 생성 문제 억제 등. |

---

## 📂 프로젝트 구조 (Project Structure)
```plaintext
📂 kogpt2-korean-finetuning
├── 📄 main_finetuning.py        # 모델 학습 및 토크나이저 최적화 스크립트 (핵심 로직)
├── 📄 app.py                    # 학습된 모델을 시연할 수 있는 Gradio 웹 인터페이스
├── 📄 requirements.txt          # 프로젝트 의존성 패키지 리스트
├── 📄 README.md                 # 프로젝트 개요 및 결과 보고서
├── 📄 training.log              # 학습 과정 및 지표 모니터링 로그
└── 📁 kogpt2-korean-finetuned/  # 파인튜닝 가중치 (대용량 파일로 GitHub 제외, 허깅페이스 연동)
```

---

## 🛠️ 주요 알고리즘 및 기술적 구현 (Technical Implementation)

### 1. Tokenization & Data Preprocessing
*   **정밀한 토크나이징 제어**: 일반적인 `AutoTokenizer` 대신 `PreTrainedTokenizerFast`를 명시하여 기본 토크나이저 설정이 라이브러리의 범용 로직에 의해 변형되는 것을 차단했습니다.
*   **Data Pipelining**: `datasets` 라이브러리를 활용하여 원본 NSMC 데이터를 필터링하고 병렬로 tokenize 처리, `labels = input_ids` 복제를 통해 Causal LM 학습 포맷 구성.

### 2. Causal LM Fine-tuning Architecture
*   **자원 효율적 학습 설정**: 15만 건의 원본 데이터를 3만 건(Train), 3천 건(Eval)으로 샘플링하고 `gradient_accumulation_steps=2`, Mixed Precision(`fp16=True`)을 적용하여 로컬 GPU 환경에서 속도 대비 최적의 품질을 유지.
*   **가중치 편향 구조 개편 (Weight Optimization)**: 기존 위키백과/뉴스 기사 위주로 편향되어 있던 원본 Base 모델의 가중치를 영화 리뷰 특유의 감성, 속어, 인터넷 구어체 텍스트로 강하게 재배치하여 뚜렷한 질적 수준의 성능 차별화 달성.
*   **Trainer API Integration**: Hugging Face `Trainer`와 `DataCollatorForLanguageModeling`을 활용한 모듈화된 자동 학습 파이프라인.

### 3. Generation Decoding Strategy Analysis
*   **Repetition Penalty (1.2)**: 텍스트 생성 모델의 고질적인 단어/구절 무한 반복 문제를 제어하기 위해 확률 페널티 부여.
*   **Top-P (Nucleus Sampling 0.9)**: 확률 분포 하위 10%의 연관성 없는 단어를 잘라내어 문맥이 파괴되는 것을 예방.
*   **Temperature (0.8)**: 안전함과 창의성 간의 밸런스가 맞는 값을 채택하여 영화 리뷰 특유의 감성적 표현력 확보.

---

## 🚀 결정적 트러블슈팅 사례 (Crucial Troubleshooting)

이 프로젝트에서 겪은 가장 고무적인 문제 해결 사례입니다:

> **[문제 현상] 한글 인코딩 파괴 및 `CUDA device-side assert triggered` 에러**
> KoGPT2 모델 파인튜닝 중 지속적인 CUDA 메모리 참조 에러가 발생하였으며, 간헐적으로 출력이 외계어(깨진 한글)로 붕괴되는 현상을 발견했습니다.
> 
> **[원인 분석 (Root Cause)] "단 1개의 토큰이 부른 참사"**
> 허깅페이스의 범용 클래스인 `AutoTokenizer`가 KoGPT2를 초기화하는 과정에서 기본 어휘 사전 크기인 `51,200`을 넘어 패딩 처리를 위해 새로운 토큰을 암시적으로 추가, 임베딩 크기를 `51,201`로 변형시켰습니다.
> 모델 학습 시 이 1개의 오프로드(Off-load)가 전체 인덱스 맵핑을 1칸씩 밀리게 만들었고, 모델은 정상적인 한국어 벡터를 내보냈으나 토크나이저가 이를 잘못된 글자로 강제 디코드하여 한글이 붕괴되는 원인이 되었습니다.
> 
> **[해결 방안 (The True Fix)]**
> 파생 토큰을 방지하기 위해 `PreTrainedTokenizerFast`를 구체적으로 명시 선언하고, `bos_token`, `eos_token`, `pad_token` 등을 본래 KoGPT2 체계에 맞게 하드코딩 방식으로 고정했습니다. 강제로 수행되던 불필요한 `model.resize_token_embeddings()`를 해제시켜 원래 모델 구조(Vocab 크기 51,200)에 100% 종속되도록 코드를 재구성한 후, 무결점의 한국어 출력을 확보했습니다.

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
