"""
GPT-2 한국어 영화 리뷰 데이터셋 파인튜닝 (속도 최적화 버전)
Author: Antigravity (AI Assistant)
Date: 2026-04-13
Description: 네이버 영화 리뷰 데이터셋(NSMC)을 활용하여 GPT-2 모델을 한국어 문장 생성에 맞게 추가 학습합니다.
             포트폴리오용으로 적합하도록 3만 건 샘플링 및 1 Epoch 설정을 적용하여 실행 속도를 최적화했습니다.
"""

import logging
import os
import torch
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)

# ==========================================
# 1. 설정 (Configuration)
# ==========================================
class Config:
    """학습 파라미터 및 경로 설정을 관리하는 클래스입니다."""
    MODEL_NAME = "gpt2"
    OUTPUT_DIR = "./gpt2-korean-finetuned"
    
    # 데이터셋 경로 (NSMC 공개 URL)
    TRAIN_URL = "https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt"
    TEST_URL = "https://raw.githubusercontent.com/e9t/nsmc/master/ratings_test.txt"

    # 학습 하이퍼파라미터 (속도 및 효율 최적화)
    EPOCHS = 1                        # 학습 반복 횟수
    BATCH_SIZE = 8                    # GPU당 배치 크기
    GRADIENT_ACCUMULATION_STEPS = 4   # 그래디언트 누적 단계 (실효 배치 크기 = 32)
    LEARNING_RATE = 5e-5              # 학습률
    MAX_LENGTH = 128                  # 시퀀스 최대 길이
    
    # 포트폴리오용 속도 개선을 위한 샘플링 설정
    TRAIN_SIZE = 30000                # 학습 데이터 샘플 수
    EVAL_SIZE = 3000                  # 평가 데이터 샘플 수

    # 로깅 및 평가 주기
    LOGGING_STEPS = 50
    SAVE_STEPS = 200
    EVAL_STEPS = 200

# ==========================================
# 2. 로깅 설정 (Setup Logging)
# ==========================================
def setup_logging():
    """콘솔과 파일에 동시에 로그를 기록하도록 설정합니다."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("training.log", encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

# ==========================================
# 3. 데이터 로드 및 전처리 (Data Preparation)
# ==========================================
def load_and_preprocess_data(tokenizer):
    """
    NSMC 데이터셋을 로드하고, 샘플링 및 토큰화를 수행합니다.
    """
    logger.info("Step 1: 데이터셋(NSMC) 다운로드 중...")
    dataset = load_dataset("csv", delimiter="\t", data_files={
        "train": Config.TRAIN_URL,
        "test": Config.TEST_URL,
    })

    logger.info("Step 2: 데이터 정제 및 속도 최적화를 위한 샘플링 진행 중...")
    # 결측치 제거 및 빈 문장 필터링
    dataset = dataset.filter(lambda x: x["document"] is not None and len(x["document"].strip()) > 0)

    # 빠른 확인을 위한 셔플 및 선택 (Sampling)
    shuffled_train = dataset["train"].shuffle(seed=42).select(range(min(Config.TRAIN_SIZE, len(dataset["train"]))))
    shuffled_test = dataset["test"].shuffle(seed=42).select(range(min(Config.EVAL_SIZE, len(dataset["test"]))))
    
    tokenized_dataset = DatasetDict({
        "train": shuffled_train,
        "test": shuffled_test
    })

    logger.info(f"데이터셋 구성 완료 - 학습용: {len(tokenized_dataset['train']):,}, 테스트용: {len(tokenized_dataset['test']):,}")

    def tokenize_function(examples):
        """텍스트를 토큰 ID로 변환하고 레이블을 생성합니다."""
        outputs = tokenizer(
            examples["document"],
            truncation=True,
            max_length=Config.MAX_LENGTH,
            padding="max_length",
        )
        # Causal LM 학습을 위해 레이블은 입력값(input_ids)을 그대로 복사합니다.
        outputs["labels"] = outputs["input_ids"].copy()
        return outputs

    logger.info("Step 3: 데이터 토큰화 진행 중 (Map 수행)...")
    tokenized_datasets = tokenized_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=tokenized_dataset["train"].column_names
    )

    return tokenized_datasets

# ==========================================
# 4. 모델 학습 (Model Training)
# ==========================================
def train_gpt2_model(tokenized_datasets, tokenizer):
    """
    GPT-2 모델을 초기화하고 Trainer API를 통해 학습을 진행합니다.
    """
    logger.info(f"Step 4: 기본 모델 로드 중: {Config.MODEL_NAME}")
    model = AutoModelForCausalLM.from_pretrained(Config.MODEL_NAME)

    # GPT-2는 기본 패딩 토큰이 없으므로 EOS(문장 종료) 토큰을 매핑합니다.
    model.config.pad_token_id = model.config.eos_token_id

    # 학습 인자 설정 (TrainingArguments)
    training_args = TrainingArguments(
        output_dir=Config.OUTPUT_DIR,
        num_train_epochs=Config.EPOCHS,
        per_device_train_batch_size=Config.BATCH_SIZE,
        per_device_eval_batch_size=Config.BATCH_SIZE,
        gradient_accumulation_steps=Config.GRADIENT_ACCUMULATION_STEPS,
        learning_rate=Config.LEARNING_RATE,
        weight_decay=0.01,                 # 가중치 감쇠 (과적합 방지)
        warmup_steps=100,                  # 웜업 단계 설정
        logging_steps=Config.LOGGING_STEPS,
        eval_strategy="steps",
        eval_steps=Config.EVAL_STEPS,
        save_strategy="steps",
        save_steps=Config.SAVE_STEPS,
        save_total_limit=2,                # 최근 체크포인트 2개만 유지
        fp16=torch.cuda.is_available(),    # GPU 가속 시 16비트 정밀도 사용
        report_to="none",                  # 외부 로깅 서비스 비활성화
        load_best_model_at_end=True,       # 평가 결과가 가장 좋은 모델 로드
        metric_for_best_model="eval_loss", # 평가 손실을 기준으로 최적 모델 선택
    )

    # 언어 모델 학습용 데이터 관리자 설정 (LM 전용)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        data_collator=data_collator,
    )

    logger.info("Step 5: 본격적인 파인튜닝(추가 학습) 시작...")
    trainer.train()

    logger.info(f"Step 6: 학습 완료된 모델을 '{Config.OUTPUT_DIR}'에 저장 중...")
    trainer.save_model(Config.OUTPUT_DIR)
    tokenizer.save_pretrained(Config.OUTPUT_DIR)

    return model

# ==========================================
# 5. 결과 검증 (Result Verification)
# ==========================================
def run_post_training_inference(model, tokenizer):
    """
    학습된 모델을 사용하여 문장 생성 테스트를 수행합니다.
    """
    logger.info("Step 7: 학습 완료 모델 성능 테스트 (문장 생성)...")
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # 테스트를 위한 샘플 프롬프트
    test_prompts = ["오늘 본 영화는", "이 영화의 결말은", "배우의 연기가"]
    print("\n" + "="*50 + "\nGPT-2 한국어 파인튜닝 결과 검증 테스트\n" + "="*50)

    for prompt in test_prompts:
        # 프롬프트를 토큰화하여 장치로 이동
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            # 텍스트 생성 파라미터 최적화
            output = model.generate(
                input_ids, 
                max_length=100, 
                do_sample=True, 
                temperature=0.8, 
                top_p=0.9, 
                repetition_penalty=1.2, 
                no_repeat_ngram_size=3
            )
        # 생성된 토큰 ID를 다시 텍스트로 변환
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        print(f"프롬프트: {prompt}\n생성결과: {generated_text}\n" + "-"*30)

# ==========================================
# 메인 실행부 (Execution)
# ==========================================
if __name__ == "__main__":
    logger.info("프로세스 초기화 중 (최적화 모드)...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"현재 실행 장치: {device.upper()}")

    # 1. 토크나이저 초기화
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
    # 패딩 토큰 설정 (데이터 전처리 전 수행 필수)
    tokenizer.pad_token = tokenizer.eos_token

    # 2. 데이터 워크플로우 실행
    tokenized_datasets = load_and_preprocess_data(tokenizer)
    
    # 3. 모델 학습 워크플로우 실행
    model = train_gpt2_model(tokenized_datasets, tokenizer)
    
    # 4. 최종 결과 검증 및 샘플 출력
    run_post_training_inference(model, tokenizer)

    logger.info("모든 프로세스가 성공적으로 완료되었습니다.")