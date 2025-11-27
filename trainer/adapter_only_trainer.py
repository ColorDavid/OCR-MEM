"""
Adapter-Only SFT Trainer for MEMModel
Minimalist implementation - only trains projection adapter with DeepSpeed ZeRO-2
"""

import os
import json
import torch
from typing import Optional
from dataclasses import dataclass, field
from functools import partial

from transformers import (
    Trainer,
    TrainingArguments,
)

from mem_adapter_only import MEMModel
from utils.data import MultiTurnConversationDataset, collate_fn


@dataclass
class AdapterTrainingArguments(TrainingArguments):
    """Extended training arguments for adapter-only training"""
    
    # Model config
    base_model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    ocr_model_name: str = "deepseek-ai/DeepSeek-OCR"
    vision_embedding_size: int = 512
    context_threshold: int = 2048
    
    # Data config
    train_data_path: str = "data/train.jsonl"
    eval_data_path: Optional[str] = None
    max_seq_length: int = 2048
    
    # Adapter config
    save_adapter_only: bool = True


class AdapterOnlyTrainer(Trainer):
    """Minimalist trainer - only trains projection adapter"""
    
    def __init__(self, model: MEMModel, args: AdapterTrainingArguments, **kwargs):
        # Freeze everything except projection adapter
        self._setup_trainable_params(model, args)
        super().__init__(model=model, args=args, **kwargs)
        
    def _setup_trainable_params(self, model: MEMModel, args: AdapterTrainingArguments):
        """Freeze base model and OCR encoder, only train adapter"""
        
        # Freeze all
        for param in model.parameters():
            param.requires_grad = False
        
        # Unfreeze projection adapter only
        for param in model.proj.parameters():
            param.requires_grad = True
        
        # Verify and print
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        
        print(f"\n{'='*60}")
        print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
        print(f"{'='*60}\n")
        
        # Safety check
        for name, param in model.named_parameters():
            if param.requires_grad and 'proj' not in name:
                raise ValueError(f"Error: {name} is trainable but not in adapter!")
    
    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        """Save adapter weights only"""
        output_dir = output_dir or self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        if self.args.save_adapter_only:
            # Save adapter weights
            adapter_state = {
                name: param.cpu() for name, param in self.model.named_parameters()
                if 'proj' in name and param.requires_grad
            }
            torch.save(adapter_state, os.path.join(output_dir, "adapter_model.bin"))
            
            # Save config
            config = {
                "vision_embedding_size": self.args.vision_embedding_size,
                "base_hidden_size": self.model.base_model_config.hidden_size,
                "context_threshold": self.args.context_threshold,
            }
            with open(os.path.join(output_dir, "adapter_config.json"), "w") as f:
                json.dump(config, f, indent=2)
            
            print(f"✓ Saved adapter to {output_dir}")
        else:
            super()._save(output_dir, state_dict)
        
        # Save tokenizer
        if self.tokenizer:
            self.tokenizer.save_pretrained(output_dir)


def setup_deepspeed_config(args: AdapterTrainingArguments) -> str:
    """
    Create DeepSpeed ZeRO-2 config with CPU offloading
    This is the ACTUAL DeepSpeed config that will be used
    """
    
    # Calculate batch sizes
    train_batch_size = (
        args.per_device_train_batch_size * 
        args.gradient_accumulation_steps * 
        args.world_size
    )
    micro_batch_size = args.per_device_train_batch_size
    
    config = {
        # Batch configuration - CRITICAL for DeepSpeed
        "train_batch_size": train_batch_size,
        "train_micro_batch_size_per_gpu": micro_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        
        # Precision - use bf16
        "bf16": {
            "enabled": True
        },
        "fp16": {
            "enabled": False
        },
        
        # ZeRO Stage 2 with CPU offloading - MAIN MEMORY SAVER
        "zero_optimization": {
            "stage": 2,
            
            # Offload optimizer states to CPU
            "offload_optimizer": {
                "device": "cpu",
                "pin_memory": True
            },
            
            # Communication optimization
            "allgather_partitions": True,
            "allgather_bucket_size": 5e8,
            "reduce_scatter": True,
            "reduce_bucket_size": 5e8,
            "overlap_comm": True,
            "contiguous_gradients": True,
        },
        
        # Gradient clipping
        "gradient_clipping": args.max_grad_norm,
        
        # Logging
        "steps_per_print": args.logging_steps,
        "wall_clock_breakdown": False,
    }
    
    # Save config
    config_path = os.path.join(args.output_dir, "ds_config.json")
    os.makedirs(args.output_dir, exist_ok=True)
    
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"✓ DeepSpeed config saved to {config_path}")
    print(f"  - ZeRO Stage 2 with optimizer offloading to CPU")
    print(f"  - Train batch size: {train_batch_size}")
    print(f"  - Micro batch size per GPU: {micro_batch_size}\n")
    
    return config_path


def train():
    """Main training function"""
    
    # ==================== Training Arguments ====================
    args = AdapterTrainingArguments(
        # Output
        output_dir="./output/adapter_checkpoints",
        run_name="mem_adapter",
        
        # Model
        base_model_name="Qwen/Qwen2.5-1.5B-Instruct",
        ocr_model_name="deepseek-ai/DeepSeek-OCR",
        vision_embedding_size=512,
        context_threshold=2048,
        
        # Data
        train_data_path="data/train.jsonl",
        eval_data_path="data/eval.jsonl",
        max_seq_length=2048,
        
        # Training
        num_train_epochs=3,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=8,
        
        # Optimization
        learning_rate=2e-4,
        warmup_ratio=0.1,
        weight_decay=0.01,
        max_grad_norm=1.0,
        
        # Precision
        bf16=True,
        tf32=True,
        
        # Evaluation & Saving
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=3,
        logging_steps=10,
        
        # DeepSpeed - will be set up below
        deepspeed=None,  # Will be populated
        
        # Misc
        dataloader_num_workers=4,
        remove_unused_columns=False,
        report_to=["tensorboard"],
        seed=42,
        
        # Adapter specific
        save_adapter_only=True,
    )
    
    # ==================== Setup DeepSpeed ====================
    # CRITICAL: Create DeepSpeed config BEFORE initializing trainer
    args.deepspeed = setup_deepspeed_config(args)
    
    # ==================== Initialize Model ====================
    print(f"{'='*60}")
    print("LOADING MODEL")
    print(f"{'='*60}\n")
    
    model = MEMModel(
        base_model_name=args.base_model_name,
        ocr_model_name=args.ocr_model_name,
        vision_embedding_size=args.vision_embedding_size,
        context_threshold=args.context_threshold,
    )
    tokenizer = model.tokenizer
    
    # ==================== Load Datasets ====================
    print(f"{'='*60}")
    print("LOADING DATASETS")
    print(f"{'='*60}\n")
    
    train_dataset = MultiTurnConversationDataset(
        jsonl_path=args.train_data_path,
        tokenizer=tokenizer,
        max_length=args.max_seq_length,
        return_dict=False
    )
    print(f"✓ Train: {len(train_dataset)} samples")
    
    eval_dataset = None
    if args.eval_data_path and os.path.exists(args.eval_data_path):
        eval_dataset = MultiTurnConversationDataset(
            jsonl_path=args.eval_data_path,
            tokenizer=tokenizer,
            max_length=args.max_seq_length,
            return_dict=False
        )
        print(f"✓ Eval: {len(eval_dataset)} samples\n")
    
    # ==================== Initialize Trainer ====================
    print(f"{'='*60}")
    print("INITIALIZING TRAINER")
    print(f"{'='*60}\n")
    
    trainer = AdapterOnlyTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=partial(collate_fn, tokenizer=tokenizer),
    )
    
    # ==================== Train ====================
    print(f"{'='*60}")
    print("TRAINING")
    print(f"{'='*60}\n")
    
    # Resume from checkpoint if exists
    checkpoint = None
    if os.path.exists(args.output_dir):
        checkpoints = [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint-")]
        if checkpoints:
            latest = max(checkpoints, key=lambda x: int(x.split("-")[-1]))
            checkpoint = os.path.join(args.output_dir, latest)
            print(f"✓ Resuming from {checkpoint}\n")
    
    trainer.train(resume_from_checkpoint=checkpoint)
    
    # ==================== Save Final Adapter ====================
    final_dir = os.path.join(args.output_dir, "final_adapter")
    trainer.save_model(final_dir)
    
    print(f"\n{'='*60}")
    print(f"✓ Training completed!")
    print(f"✓ Final adapter saved to: {final_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    train()