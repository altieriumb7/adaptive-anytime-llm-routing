import argparse
import json
from typing import Dict, Any, List

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
)

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from src.train.tokenize_sft import tokenize_one, DataCollatorForCausalLM


def load_sft_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", required=True, help="e.g., Qwen/Qwen2.5-Math-1.5B-Instruct")
    ap.add_argument("--train_jsonl", required=True)
    ap.add_argument("--output_dir", required=True)

    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=16)
    ap.add_argument("--max_steps", type=int, default=-1)

    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)

    ap.add_argument("--max_train_samples", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        # reasonable default for Qwen/Llama-style blocks:
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    rows = load_sft_jsonl(args.train_jsonl)
    if args.max_train_samples > 0:
        rows = rows[: args.max_train_samples]

    def gen():
        for r in rows:
            toks = tokenize_one(tokenizer, r["messages"], r["response"])
            toks["uid"] = r["uid"]
            toks["budget_t"] = r["budget_t"]
            yield toks

    ds = Dataset.from_generator(gen)

    collator = DataCollatorForCausalLM(tokenizer)

    train_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
max_steps=(args.max_steps if (args.max_steps is not None and args.max_steps > 0) else -1),
        logging_steps=20,
        save_steps=200,
        save_total_limit=2,
        bf16=torch.cuda.is_available(),
        fp16=False,
        report_to="none",
        seed=args.seed,
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.0,
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=ds,
        data_collator=collator,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved adapter + tokenizer to: {args.output_dir}")


if __name__ == "__main__":
    main()
