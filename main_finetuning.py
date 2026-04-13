"""
KoGPT2 기반 한국어 영화 리뷰 데이터셋 파인튜닝 (성능 최적화 및 품질 개선 버전)
Author: Antigravity (AI Assistant)
Date: 2026-04-13
"""

import logging
import os
import sys
import torch
from datasets import load_dataset, DatasetDict
from transformers import (
    PreTrainedTokenizerFast,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)

# Windows 콘솔 인코딩 대응 (UTF-8 고정)
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        # sys.stdout.reconfigure might not be available in all Python environments
        pass

# ==========================================
# 1. 설정 (Configuration)
# ==========================================
class Config:
    MODEL_NAME = "skt/kogpt2-base-v2"
    OUTPUT_DIR = "./kogpt2-korean-finetuned"
    
    TRAIN_URL = "https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt"
    TEST_URL = "https://raw.githubusercontent.com/e9t/nsmc/master/ratings_test.txt"

    EPOCHS = 1                        # 안정성을 위해 1 Epoch 재시도 (품질 우선)
    BATCH_SIZE = 16                   
    GRADIENT_ACCUMULATION_STEPS = 2
    LEARNING_RATE = 5e-5
    MAX_LENGTH = 128
    
    TRAIN_SIZE = 30000                # 최적화된 샘플링 
    EVAL_SIZE = 3000

    LOGGING_STEPS = 50
    SAVE_STEPS = 500
    EVAL_STEPS = 500

# ==========================================
# 2. 로딩 및 전처리 (Preparation)
# ==========================================
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("training.log", encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

def load_and_preprocess_data(tokenizer):
    logger.info("Step 1: 데이터 로딩 중...")
    try:
        dataset = load_dataset("csv", delimiter="\t", data_files={"train": Config.TRAIN_URL, "test": Config.TEST_URL})
    except Exception as e:
        logger.error(f"데이터 로드 실패: {e}")
        raise

    logger.info("Step 2: 데이터 정제(5자 미만 제거) 및 샘플링 중...")
    dataset = dataset.filter(lambda x: x["document"] is not None and len(x["document"].strip()) > 5)
    
    train_subset = dataset["train"].shuffle(seed=42).select(range(min(Config.TRAIN_SIZE, len(dataset["train"]))))
    test_subset = dataset["test"].shuffle(seed=42).select(range(min(Config.EVAL_SIZE, len(dataset["test"]))))
    
    tokenized_dataset = DatasetDict({"train": train_subset, "test": test_subset})

    def tokenize_function(examples):
        outputs = tokenizer(examples["document"], truncation=True, max_length=Config.MAX_LENGTH, padding="max_length")
        outputs["labels"] = outputs["input_ids"].copy()
        return outputs

    logger.info("Step 3: 토크나이징 진행 중...")
    return tokenized_dataset.map(tokenize_function, batched=True, remove_columns=train_subset.column_names)

# ==========================================
# 3. 모델 학습 (Training)
# ==========================================
def train_model(tokenized_datasets, tokenizer):
    logger.info(f"Step 4: 모델 로드 및 임베딩 확인: {Config.MODEL_NAME}")
    model = AutoModelForCausalLM.from_pretrained(Config.MODEL_NAME)
    
    # PreTrainedTokenizerFast를 사용하여 어휘 사전 크기 51,200을 완벽하게 유지합니다.
    # 추가 토큰이 없으므로 resize_token_embeddings 호출이 필요 없습니다.
    logger.info(f"  - 토크나이저 어휘 크기: {len(tokenizer)}")
    logger.info(f"  - 모델 최종 vocab_size: {model.config.vocab_size}")

    training_args = TrainingArguments(
        output_dir=Config.OUTPUT_DIR,
        num_train_epochs=Config.EPOCHS,
        per_device_train_batch_size=Config.BATCH_SIZE,
        per_device_eval_batch_size=Config.BATCH_SIZE,
        gradient_accumulation_steps=Config.GRADIENT_ACCUMULATION_STEPS,
        learning_rate=Config.LEARNING_RATE,
        weight_decay=0.01,
        warmup_steps=100,
        logging_steps=Config.LOGGING_STEPS,
        eval_strategy="steps",
        eval_steps=Config.EVAL_STEPS,
        save_strategy="steps",
        save_steps=Config.SAVE_STEPS,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        report_to="none",
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    logger.info("Step 5: 학습 시작...")
    trainer.train()

    logger.info(f"Step 6: 모델 저장 도중: {Config.OUTPUT_DIR}")
    trainer.save_model(Config.OUTPUT_DIR)
    tokenizer.save_pretrained(Config.OUTPUT_DIR)
    return model

# ==========================================
# 4. 추론 테스트 (Inference)
# ==========================================
def run_inference(model, tokenizer):
    logger.info("Step 7: 최종 추론 테스트...")
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    prompts = ["오늘 본 영화는", "이 영화의 결말은", "배우의 연기가"]
    print("\n" + "="*50 + "\n[KoGPT2 파인튜닝 결과]\n" + "="*50)

    for prompt in prompts:
        # KoGPT2 고유의 BOS 토큰 처리
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_length=64,
                do_sample=True,
                temperature=0.8,
                top_p=0.9,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        # 생성된 텍스트 안전하게 인코딩 처리하여 출력
        try:
            generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
            print(f"프롬프트: {prompt}\n생성결과: {generated_text}\n" + "-"*30)
        except Exception as e:
            logger.error(f"생성 텍스트 출력 에러: {e}")

if __name__ == "__main__":
    logger.info("KoGPT2 최종 안정화 모델 구축 중...")
    
    # KoGPT2 전용 토크나이저 로드 (special tokens 명시적 지정으로 51200 규격 유지)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(
        Config.MODEL_NAME, 
        bos_token='</s>', 
        eos_token='</s>', 
        unk_token='<unk>', 
        pad_token='<pad>', 
        mask_token='<mask>'
    )

    
    tokenized_datasets = load_and_preprocess_data(tokenizer)
    model = train_model(tokenized_datasets, tokenizer)
    run_inference(model, tokenizer)
    
    logger.info("모든 프로세스가 성공적으로 완료되었습니다.")
