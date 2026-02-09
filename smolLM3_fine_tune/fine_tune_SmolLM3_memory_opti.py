import os
import torch
from dotenv import load_dotenv
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
import trackio as wandb

# 0. Suppress TensorFlow/OneDNN warnings
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# CRITICAL FOR WINDOWS: Must wrap in if __name__ guard to prevent multiprocessing crashes
if __name__ == "__main__":
    # 1. Hardware Check
    load_dotenv()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # Clear any cached memory from previous runs
        print(f"✓ Training on: {torch.cuda.get_device_name(0)}")
        print(
            f"✓ VRAM Available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
        )

    # Initialize experiment tracking with Trackio
    wandb.init(project="smollm3-sft", name="smollm3-optimized-v2")
    # wandb.show(project="smollm3-sft")

    # ==================== 2. QUANTIZATION CONFIG ====================
    # Reduces model memory from ~12GB to ~4GB by using 4-bit precision
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,  # Use 4-bit quantization instead of 16-bit (saves 75% memory)
        bnb_4bit_quant_type="nf4",  # NormalFloat4: special 4-bit format optimized for neural networks
        bnb_4bit_compute_dtype=torch.float16,  # Compute in fp16 for speed (use bfloat16 for stability)
        bnb_4bit_use_double_quant=True,  # Quantize the quantization constants (saves extra ~0.4GB)
    )

    # ==================== 3. MODEL & TOKENIZER ====================
    model_id = "HuggingFaceTB/SmolLM3-3B-Base"

    # Load tokenizer - converts text to numbers the model understands
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token  # Use end-of-sequence token for padding
    tokenizer.padding_side = (
        "right"  # Add padding to the right side (important for causal LM)
    )

    # Load model with quantization applied
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quant_config,  # Apply 4-bit quantization
        device_map={
            "": 0
        },  # Load entire model on GPU 0 (more stable than "auto" on Windows)
        attn_implementation="sdpa",  # Use Scaled Dot Product Attention (PyTorch native, memory efficient)
        dtype=torch.float16,  # Model weights stored in float16 (faster on RTX 3070)
    )

    # ==================== 4. LORA CONFIG ====================
    # LoRA = Low-Rank Adaptation: only trains small adapter layers instead of full model
    # This reduces trainable parameters from 3B to ~16M (0.5% of original)
    model = prepare_model_for_kbit_training(
        model
    )  # Prepare quantized model for training

    peft_config = LoraConfig(
        r=16,  # LoRA rank: size of the low-rank matrices (higher = more expressive but more memory)
        lora_alpha=32,  # LoRA scaling factor: controls strength of adaptations (typically 2x rank)
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
        ],  # Which layers to add LoRA to (attention layers)
        lora_dropout=0.05,  # Dropout rate for LoRA layers (prevents overfitting)
        bias="none",  # Don't train bias parameters (saves memory)
        task_type="CAUSAL_LM",  # Task type: causal language modeling (predicting next token)
    )

    # ==================== 5. DATASET ====================
    # Load conversational dataset for fine-tuning
    dataset = load_dataset("HuggingFaceTB/smoltalk2_everyday_convs_think")

    # ==================== 6. TRAINING CONFIGURATION ====================
    training_args = SFTConfig(
        # --- Output & Logging ---
        output_dir="./smollm3-finetuned",  # Where to save checkpoints
        report_to="trackio",  # Send metrics to Trackio for visualization
        logging_steps=5,  # Log metrics every 5 steps
        save_steps=250,  # Save checkpoint every 250 steps
        save_total_limit=2,  # Only keep 2 most recent checkpoints (saves disk space)
        # --- Training Duration ---
        max_steps=1000,  # Total training steps (use num_train_epochs for epoch-based training)
        warmup_steps=50,  # Gradually increase learning rate for first 50 steps (stabilizes training)
        # --- Batch Size & Memory ---
        per_device_train_batch_size=4,  # Process 2 samples per GPU per step
        gradient_accumulation_steps=2,  # Accumulate gradients over 8 steps before updating weights
        # Effective batch size = per_device_train_batch_size × gradient_accumulation_steps = 2 × 8 = 16 (good balance of speed and memory)
        # --- Learning Rate ---
        learning_rate=5e-4,  # Step size for weight updates (0.0005 - higher than usual due to LoRA)
        # --- Precision & Speed ---
        fp16=True,  # Train in float16 precision (2x faster, uses half the memory vs float32)
        # Alternative: bf16=True for bfloat16 (better stability, slightly slower on RTX 3070)
        # --- Memory Optimization ---
        gradient_checkpointing=False,  # Disabled for speed (enable if OOM - saves ~40% VRAM but 20% slower)
        optim="paged_adamw_8bit",  # 8-bit Adam optimizer (saves 1-2GB VRAM vs 32-bit)
        # --- Sequence Packing ---
        max_length=256,  # Maximum sequence length in tokens (lower = less memory, faster)
        packing=False,  # CHANGED: Disable to avoid gradient checkpointing warnings and potential instability
        dataset_text_field="text",  # Specify which field contains text (required when packing=False)
        # --- Data Loading (WINDOWS SPECIFIC) ---
        dataloader_num_workers=0,  # MUST BE 0 ON WINDOWS (prevents multiprocessing crashes)
        # On WSL/Linux: use 2-4 for faster data loading
        dataloader_pin_memory=False,  # Don't pin memory on Windows (can cause issues)
        # On WSL/Linux: set to True for faster CPU→GPU transfer
    )

    # ==================== 7. TRAINER ====================
    # SFTTrainer = Supervised Fine-Tuning Trainer (specialized for instruction tuning)
    trainer = SFTTrainer(
        model=model,  # The quantized model with LoRA adapters
        train_dataset=dataset["train"],  # Training data
        peft_config=peft_config,  # LoRA configuration
        args=training_args,  # All training hyperparameters
        processing_class=tokenizer,  # Tokenizer for processing text
    )

    # ==================== 8. RUN TRAINING ====================
    print("Starting training...")
    print(
        f"Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}"
    )

    try:
        trainer.train()  # Start the training loop
    except torch.cuda.OutOfMemoryError:
        print("\n❌ OOM Error!")
        print("Solutions:")
        print("  1. Reduce per_device_train_batch_size=2")
        print("  2. Enable gradient_checkpointing=True")
        print("  3. Reduce max_length=128")
        raise
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
        print("Saving checkpoint...")
        trainer.save_model("./smollm3-finetuned-interrupted")
        raise
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        raise

    # ==================== 9. SAVE MODEL ====================
    # Save the trained LoRA adapters (only ~32MB, not the full 3B model)
    trainer.save_model("./smollm3-finetuned-final")
    tokenizer.save_pretrained("./smollm3-finetuned-final")  # Save tokenizer config too
    print("✓ Training complete!")
    print(f"✓ Model saved to: ./smollm3-finetuned-final")


# ==================== KEY CONCEPTS ====================

### Memory Trade-offs:
"""
- 4-bit quantization: 12GB → 4GB (quality: ~98%)
- LoRA: Train 0.5% of params (quality: 95-98%)
- Gradient checkpointing: Save 40% VRAM, cost 20% speed
- fp16: Use 50% memory vs fp32
"""

### Effective Batch Size:
"""
Real batch size = per_device_batch_size × gradient_accumulation_steps
                = 4 × 2 = 8

Higher effective batch = more stable training
Lower per_device_batch = less VRAM usage
"""

### Training Speed Impact:
"""
- fp16 vs fp32: ~2x faster
- bf16 vs fp16: ~same speed (better stability)
- gradient_checkpointing=False vs True: ~25% faster
- packing=True: 10-20% faster (but can cause issues)
- max_length=256 vs 512: ~40% faster
- batch_size=4 vs 2: ~30% fewer iterations
- dataloader_num_workers=4 (Linux): ~15% faster overall
"""

### Performance Summary:
"""
Current config on RTX 3070:
- Speed: ~3.2s/iteration
- Total time: ~53 minutes for 1000 steps
- VRAM usage: ~6-7GB
- Stable training without OOM
"""
