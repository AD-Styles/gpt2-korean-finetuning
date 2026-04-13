# 📚 Optimizing Korean Text Generation via GPT-2 Fine-tuning and Decoding Strategy Analysis
### GPT-2 한국어 파인튜닝과 디코딩 전략 분석을 통한 문장 생성 최적화

## 📌 프로젝트 요약 (Project Overview)
본 프로젝트는 영어 기반의 범용 언어 모델인 **GPT-2**를 활용하여 한국어 도메인(영화 리뷰)에 특화된 텍스트 생성 파이프라인을 구축한 실습 기록입니다. Hugging Face의 Transformers와 Datasets 라이브러리를 통해 데이터 전처리부터 모델 학습까지의 전 과정을 모듈화하여 구현했습니다. 특히 학습 효율을 위해 30,000건의 데이터를 활용한 최적화된 파인튜닝을 수행했으며, 출력 제어 메커니즘인 디코딩 전략을 실습에 적용했습니다.

---

## 🎯 핵심 목표 (Motivation)
| 구분 | 세부 내용 |
| :--- | :--- |
| **Tokenizer 효율 분석** | BPE 기반의 GPT-2 토크나이저가 한국어 처리 시 발생하는 토큰 팽창 문제를 진단하고 효율적인 시퀀스 처리 방안 모색 |
| **도메인 특화 Fine-tuning** | 사전 학습된 언어 모델에 NSMC(네이버 영화 리뷰) 데이터를 주입하여 문맥에 맞는 한국어 문장을 생성하는 Causal LM 학습 구현 |
| **디코딩 전략 최적화** | 생성 파라미터(Temperature, Top-p, Top-k) 실험을 통해 텍스트의 창의성과 일관성 사이의 Trade-off 관계 분석 및 최적값 도출 |

---

## 📂 프로젝트 구조 (Project Structure)
```plaintext
📂 gpt2-korean-finetuning
├── 📄 .gitignore                # Git 관리 제외 설정 파일
├── 📄 LICENSE                   # MIT 라이선스 파일
├── 📄 README.md                 # 프로젝트 개요 및 결과 보고서
├── 📄 main_finetuning.py        # 모듈화된 파인튜닝 스크립트 (핵심 로직)
├── 📄 requirements.txt          # 프로젝트 의존성 패키지 리스트
├── 📄 training.log              # 학습 과정 및 지표 실시간 기록 로그
└── 📁 gpt2-korean-finetuned/    # 파인튜닝이 완료된 모델 및 토크나이저 가중치
```

---

## 🛠️ 주요 알고리즘 및 기술적 구현 (Technical Implementation)

### 1. Tokenization & Data Preprocessing
| 구현 단계 | 활용 모듈 및 파라미터 | 기술적 포인트 |
| :--- | :--- | :--- |
| **Tokenizer 비교** | `AutoTokenizer` (gpt2) | 영어 중심 모델의 한국어 토큰화 비용 확인 및 `pad_token` 수동 설정 |
| **Dataset Clean-up** | `.filter()`, `.shuffle()` | 결측치 제거 및 실습 효율을 위한 Train(30k), Eval(3k) 데이터 샘플링 구축 |
| **Causal LM Format** | `labels = input_ids.copy()` | 다음 단어 예측 학습을 위해 입력값과 정답값을 동일하게 구성하는 전처리 적용 |

### 2. Causal LM Fine-tuning Architecture
| 프로세스 순서 | 활용 모듈 | 수행 내용 |
| :--- | :--- | :--- |
| **1. Model Loading** | `AutoModelForCausalLM` | 사전 학습된 GPT-2 가운치를 로드하고 파라미터 수 및 모델 크기 분석 |
| **2. Trainer Config** | `TrainingArguments` | `num_train_epochs=1`, `learning_rate=5e-5` 등 로컬 환경 최적화 설정 |
| **3. Optimization** | `fp16=True`, `grad_accum` | Mixed Precision과 Gradient Accumulation(4회)을 활용한 GPU 자원 최적화 |

### 3. Generation Decoding Strategy Analysis
| 파라미터 | 제어 목적 | 기술적 의미 설명 |
| :--- | :--- | :--- |
| **Temperature** | 확률 분포 조절 | 낮은 값은 보수적, 높은 값은 창의적인(탐색적) 답변 생성 유도 |
| **Top-p (Nucleus)** | 누적 확률 필터링 | 상위 P% 내의 단어 후보군만 고려하여 문맥적 일관성 유지 |
| **Penalty** | `repetition_penalty` | 동일 문구 반복을 억제하여 문장 품질 향상 (본 프로젝트 1.2 적용) |

---

## 🚀 트러블슈팅 및 성능 최적화 (Troubleshooting & Optimization)
*   **학습 데이터 불균형 및 속도 이슈**: 15만 건의 전체 데이터 학습 시 발생하는 과도한 시간 소요를 해결하기 위해 3만 건의 유의미한 샘플링을 수행하고 1 Epoch로 최적화했습니다.
*   **패딩 토큰 에러**: GPT-2 모델에 기본 `pad_token`이 없는 문제를 해결하기 위해 `eos_token`을 패딩 토큰으로 매핑하여 전처리 에러를 방지했습니다.
*   **생성 품질 저하**: 파인튜닝 후 나타나는 무의미한 반복 생성을 `no_repeat_ngram_size=3` 설정을 통해 제어했습니다.

---

## 📊 학습 개념의 직관적 해석 (Analogies)
| 핵심 개념 | 비유 (Analogy) | 기술적 의미 설명 |
| :--- | :--- | :--- |
| **Fine-tuning** | **전공 심화 학습** | 기초 상식(Pre-training)이 있는 AI에게 특정 분야(영화 리뷰)의 전문 지식을 교육 |
| **Temperature** | **성격 조절기** | 0에 가까우면 신중하고 보수적인 성격, 2에 가까우면 모험적이고 자유로운 성격 |
| **Tokenization** | **퍼즐 조각 내기** | 문장을 모델이 처리하기 가장 좋은 단위의 조각(토큰)으로 쪼개어 번호 부여 |

---

## 🤖 최종 생성 결과 (Inference Results)

### Before vs After Comparison
*   **Prompt**: "오늘 본 영화는"
*   **Before (Original GPT-2)**: "is a very important part of our life..." (영어 위주 생성)
*   **After (Fine-tuned)**: "오늘 본 영화는 정말 감동적이었습니다. 배우들의 연기가 인상 깊었네요." (한국어 도메인 적응 완료)

---

## 💡 회고록 (Retrospective)
  이번 프로젝트를 통해 사전 학습된 모델을 특정 언어와 도메인에 적응시키는 **전이 학습(Transfer Learning)**의 강력함을 실감했습니다. 특히 단순히 학습을 마치는 것이 아니라, 사용자 환경에 맞춰 데이터 규모를 유연하게 조정(Sampling)하고 최적의 학습 효율을 찾아내는 과정이 실무 현장에서 얼마나 중요한지 깨달았습니다.

  가장 흥미로웠던 지점은 생성 파라미터 조정에 따른 모델의 '태도 변화'였습니다. temperature 값을 통해 창의성과 일관성 사이의 균형점을 찾는 과정은 모델의 기술적 이해를 넘어 데이터 엔지니어로서의 감각을 기르는 데 큰 도움이 되었습니다. 이번 실습을 기반으로 향후에는 효율적인 학습 기법인 **LoRA(Low-Rank Adaptation)** 등을 추가로 연구하여 대규모 모델 최적화 역량을 확장하고자 합니다.

