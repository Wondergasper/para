"""
train_lora.py
-------------
Phase 4: LoRA Fine-Tuning with PEFT + TRL SFTTrainer

What it does:
    Loads the base DeepSeek-Coder model in 4-bit quantization.
    Attaches a LoRA adapter (only ~0.1% of parameters are trainable).
    Trains on your verified corpus.jsonl for 3 epochs.
    Saves only the adapter weights (not the full model) to ./apg-adapter-v1/

Requirements:
    pip install transformers peft trl datasets accelerate bitsandbytes torch

Hardware:
    Minimum: 16GB VRAM GPU (RTX 3090, A100, etc.)
    Budget:  Google Colab free T4 (15GB VRAM) — set batch_size=1

Run:
    python train_lora.py --corpus corpus.jsonl --output ./apg-adapter-v1
    python train_lora.py --corpus corpus.jsonl --output ./apg-adapter-v1 --colab
"""

import argparse
import os
import sys
import json

# ── Dependency check ───────────────────────────────────────────────────────────

def check_dependencies():
    missing = []
    for pkg in ["transformers", "peft", "trl", "datasets", "accelerate", "bitsandbytes", "torch"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"Missing packages: {', '.join(missing)}")
        print(f"Install with: pip install {' '.join(missing)}")
        sys.exit(1)

check_dependencies()

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset


# ── Model options ──────────────────────────────────────────────────────────────

MODELS = {
    "deepseek-6.7b": "deepseek-ai/deepseek-coder-6.7b-instruct",
    "deepseek-1.3b": "deepseek-ai/deepseek-coder-1.3b-instruct",  # For Colab free tier
    "qwen-7b":       "Qwen/Qwen2.5-Coder-7B-Instruct",
}


# ── Corpus validator ───────────────────────────────────────────────────────────

def validate_corpus(corpus_path: str) -> int:
    """
    Check the corpus JSONL is valid and has enough records.
    Returns the record count.
    """
    count = 0
    with open(corpus_path, "r") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"  ✗ Line {i} is not valid JSON: {e}")
                sys.exit(1)

            if "messages" not in record:
                print(f"  ✗ Line {i} missing 'messages' key")
                sys.exit(1)

            roles = [m["role"] for m in record["messages"]]
            if roles != ["system", "user", "assistant"]:
                print(f"  ✗ Line {i} has wrong message roles: {roles}")
                sys.exit(1)

            count += 1

    print(f"  ✓ Corpus has {count} valid records.")
    if count < 500:
        print(f"  ⚠ WARNING: {count} records is below the 500 minimum.")
        print(f"    Fine-tuning will still run but results may be poor.")
    return count


# ── 4-bit quantization config ──────────────────────────────────────────────────

def get_bnb_config() -> BitsAndBytesConfig:
    """
    BitsAndBytes 4-bit quantization config.
    Reduces VRAM from ~14GB → ~5GB for a 6.7B model.
    """
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",        # NF4 is better than FP4 for LLMs
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,   # Nested quantization saves ~0.4GB extra
    )


# ── LoRA config ────────────────────────────────────────────────────────────────

def get_lora_config() -> LoraConfig:
    """
    LoRA adapter configuration.

    r=16, alpha=32 is a standard starting point.
    Target modules are the attention projection matrices.

    Trainable parameters: ~0.1% of total (roughly 8M out of 6.7B).
    """
    return LoraConfig(
        r=16,                        # Rank — higher = more capacity, more VRAM
        lora_alpha=32,               # Scaling factor (usually 2x rank)
        target_modules=[
            "q_proj",                # Query projection
            "v_proj",                # Value projection
            "k_proj",                # Key projection
            "o_proj",                # Output projection
        ],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )


# ── Training arguments ─────────────────────────────────────────────────────────

def get_training_args(output_dir: str, colab_mode: bool = False) -> SFTConfig:
    """
    Training arguments for SFTTrainer.

    colab_mode=True uses smaller batch sizes suitable for free T4 (15GB VRAM).
    """
    if colab_mode:
        # Colab T4: smaller batches, more gradient accumulation to compensate
        batch_size            = 1
        gradient_accumulation = 16
    else:
        # Local GPU (16GB+ VRAM)
        batch_size            = 4
        gradient_accumulation = 4

    return SFTConfig(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        fp16=True,                   # Mixed precision training
        bf16=False,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,          # Keep only last 2 checkpoints
        load_best_model_at_end=False,
        report_to="none",            # Set "wandb" if you want experiment tracking
        dataloader_num_workers=0,
        remove_unused_columns=False,
        max_seq_length=2048,
    )


# ── Main training function ─────────────────────────────────────────────────────

def train(
    corpus_path: str,
    output_dir:  str,
    model_key:   str  = "deepseek-6.7b",
    colab_mode:  bool = False,
    dry_run:     bool = False,
):
    """
    Full LoRA fine-tuning pipeline.

    Args:
        corpus_path: Path to corpus.jsonl (output of export_corpus.py).
        output_dir:  Directory to save the LoRA adapter weights.
        model_key:   Which model to fine-tune (see MODELS dict).
        colab_mode:  Use smaller batch sizes for Google Colab T4.
        dry_run:     Validate setup and print config without actually training.
    """
    model_name = MODELS.get(model_key)
    if not model_name:
        print(f"Unknown model key '{model_key}'. Choose from: {list(MODELS.keys())}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"APG LoRA Fine-Tuning")
    print(f"{'='*60}")
    print(f"Model:       {model_name}")
    print(f"Corpus:      {corpus_path}")
    print(f"Output:      {output_dir}")
    print(f"Colab mode:  {colab_mode}")
    print(f"{'='*60}\n")

    # ── Step 1: Validate corpus ────────────────────────────────────────────────
    print("Step 1: Validating corpus...")
    record_count = validate_corpus(corpus_path)

    if dry_run:
        print("\nDry run complete. Configuration looks good.")
        print(f"Would train on {record_count} examples for 3 epochs.")
        return

    # ── Step 2: Load model in 4-bit ────────────────────────────────────────────
    print(f"\nStep 2: Loading {model_name} in 4-bit quantization...")
    bnb_config = get_bnb_config()

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,      # Required for DeepSeek models
    )
    model.config.use_cache = False   # Must be False during training with gradient checkpointing

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token     = tokenizer.eos_token
    tokenizer.padding_side  = "right"   # Pad on the right for causal LM training

    print(f"  ✓ Model loaded. Device map: {model.hf_device_map}")

    # ── Step 3: Attach LoRA adapter ────────────────────────────────────────────
    print("\nStep 3: Attaching LoRA adapter...")
    lora_config = get_lora_config()
    model       = get_peft_model(model, lora_config)

    trainable, total = model.get_nb_trainable_parameters()
    print(f"  ✓ Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.3f}%)")

    # ── Step 4: Load dataset ───────────────────────────────────────────────────
    print(f"\nStep 4: Loading dataset from {corpus_path}...")
    dataset = load_dataset(
        "json",
        data_files={"train": corpus_path},
        split="train",
    )
    print(f"  ✓ Dataset loaded: {len(dataset)} examples")

    # ── Step 5: Configure trainer ──────────────────────────────────────────────
    print("\nStep 5: Configuring SFTTrainer...")
    training_args = get_training_args(output_dir, colab_mode)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=training_args,
    )

    # ── Step 6: Train ──────────────────────────────────────────────────────────
    print(f"\nStep 6: Starting training ({training_args.num_train_epochs} epochs)...")
    print("  This will take a while. Progress below:\n")

    trainer.train()

    # ── Step 7: Save adapter ───────────────────────────────────────────────────
    print(f"\nStep 7: Saving LoRA adapter to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"\n{'='*60}")
    print(f"✓ Fine-tuning complete!")
    print(f"  Adapter saved to: {output_dir}")
    print(f"  Load it with:     from peft import PeftModel")
    print(f"  Next step:        python ab_test.py --adapter {output_dir}")
    print(f"{'='*60}\n")


# ── Load fine-tuned model helper ───────────────────────────────────────────────

def load_finetuned(adapter_dir: str, model_key: str = "deepseek-6.7b", merge: bool = False):
    """
    Load the base model + LoRA adapter for inference.

    Args:
        adapter_dir: Path to the saved adapter (output of train()).
        model_key:   Which base model was used for training.
        merge:       If True, merge adapter into base model weights for faster inference.
                     Merged model cannot be un-merged — keep the adapter dir.

    Returns:
        (model, tokenizer) tuple ready for inference.
    """
    from peft import PeftModel

    model_name = MODELS[model_key]
    bnb_config = get_bnb_config()

    print(f"Loading base model {model_name}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir, trust_remote_code=True)

    print(f"Loading LoRA adapter from {adapter_dir}...")
    model = PeftModel.from_pretrained(base_model, adapter_dir)

    if merge:
        print("Merging adapter into base model (faster inference)...")
        model = model.merge_and_unload()

    model.eval()
    print("✓ Model ready for inference.")
    return model, tokenizer


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LoRA fine-tuning for APG parallel code generation."
    )
    parser.add_argument("--corpus",  default="corpus.jsonl",    help="Path to corpus.jsonl")
    parser.add_argument("--output",  default="./apg-adapter-v1",help="Output directory for adapter")
    parser.add_argument("--model",   default="deepseek-6.7b",   choices=list(MODELS.keys()))
    parser.add_argument("--colab",   action="store_true",        help="Use Colab-friendly batch sizes")
    parser.add_argument("--dry-run", action="store_true",        help="Validate only, don't train")

    args = parser.parse_args()

    train(
        corpus_path=args.corpus,
        output_dir=args.output,
        model_key=args.model,
        colab_mode=args.colab,
        dry_run=args.dry_run,
    )
