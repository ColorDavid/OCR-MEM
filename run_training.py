#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
MEMModel Adapter 训练脚本
================================================================================

本脚本用于启动 MEMModel 投影适配器的训练。

【运行位置】
    必须在 OCR-MEM 目录下运行：
    cd /path/to/OCR-MEM
    
【运行方式】
    1. 单机单卡训练：
        python run_training.py
    
    2. 单机多卡训练（推荐使用 DeepSpeed）：
        deepspeed --num_gpus=4 run_training.py
    
    3. 使用 torchrun 多卡训练：
        torchrun --nproc_per_node=4 run_training.py
    
    4. 使用提供的 shell 脚本：
        bash run_training.sh

【目录结构要求】
    OCR-MEM/
    ├── run_training.py          # 本脚本
    ├── run_training.sh          # Shell 启动脚本
    ├── ds_config.json           # DeepSpeed 配置（自动生成或手动指定）
    ├── mem_adapter_only.py      # MEMModel 定义
    ├── trainer/
    │   └── adapter_only_trainer.py
    ├── utils/
    │   └── data.py
    ├── data/                    # 训练数据目录
    │   ├── train.jsonl
    │   └── eval.jsonl
    ├── models/                  # 本地模型目录
    │   ├── qwen2.5-1.5b/        # 基础语言模型
    │   └── deepseek-ocr/        # OCR 编码器
    └── output/                  # 输出目录（自动创建）
        └── adapter_checkpoints/

【配置说明】
    请根据您的实际环境修改以下配置：
    1. MODEL_CONFIG: 模型相关路径
    2. DATA_CONFIG: 数据集路径
    3. TRAINING_CONFIG: 训练超参数
    4. DEEPSPEED_CONFIG: DeepSpeed 配置（可选）

作者：OCR-MEM Team
================================================================================
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import json
import argparse
from pathlib import Path
from functools import partial
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.distributed as dist

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("警告: wandb 未安装，将禁用 wandb 日志")

from transformers import TrainingArguments

# 添加项目根目录到 Python 路径
PROJECT_ROOT = Path(__file__).parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

from mem_adapter_only import MEMModel, MEMConfig
from trainer.adapter_only_trainer import AdapterOnlyTrainer, AdapterTrainingArguments
from utils.data import MultiTurnConversationDataset, collate_fn


# ============================================================================
# 配置区域 - 请根据您的环境修改
# ============================================================================

# -------------------- 模型配置 --------------------
# 注意：这里使用本地路径，而非 HuggingFace ID
MODEL_CONFIG = {
    # 基础语言模型的本地路径
    # 例如："/data/models/Qwen2.5-1.5B-Instruct" 或 "./models/qwen2.5-1.5b"
    "base_model_path": "./models/qwen2.5-1.5b-instruct",
    
    # OCR 视觉编码器的本地路径
    # 例如："/data/models/DeepSeek-OCR" 或 "./models/deepseek-ocr"
    "ocr_model_path": "./models/deepseek-ocr",
    
    # 视觉编码器输出的嵌入维度（需与 OCR 模型匹配）
    "vision_embedding_size": 512,
    
    # 触发视觉压缩的上下文长度阈值
    "context_threshold": 2048,
}

# -------------------- 数据配置 --------------------
DATA_CONFIG = {
    # 训练数据 JSONL 文件的本地路径
    "train_data_path": "./data/train.jsonl",
    
    # 验证数据 JSONL 文件的本地路径（可选，设为 None 跳过验证）
    "eval_data_path": "./data/eval.jsonl",
    
    # 最大序列长度
    "max_seq_length": 2048,
}

# -------------------- 训练配置 --------------------
TRAINING_CONFIG = {
    # 输出目录
    "output_dir": "./output/adapter_checkpoints",
    
    # 实验名称（用于日志和 WandB）
    "run_name": "mem_adapter_training",
    
    # WandB 项目名称
    "wandb_project": "OCR-MEM",
    
    # 训练轮数
    "num_train_epochs": 3,
    
    # 每个 GPU 的训练 batch size
    "per_device_train_batch_size": 2,
    
    # 每个 GPU 的验证 batch size
    "per_device_eval_batch_size": 4,
    
    # 梯度累积步数（有效 batch size = per_device * accumulation * num_gpus）
    "gradient_accumulation_steps": 8,
    
    # 学习率
    "learning_rate": 2e-4,
    
    # Warmup 比例
    "warmup_ratio": 0.1,
    
    # 权重衰减
    "weight_decay": 0.01,
    
    # 梯度裁剪
    "max_grad_norm": 1.0,
    
    # 评估策略
    "eval_strategy": "steps",
    "eval_steps": 200,
    
    # 保存策略
    "save_strategy": "steps",
    "save_steps": 200,
    "save_total_limit": 3,
    
    # 日志策略
    "logging_steps": 10,
    
    # 数据加载器配置
    "dataloader_num_workers": 4,
    
    # 随机种子
    "seed": 42,
}

# -------------------- DeepSpeed 配置 --------------------
# 如果设为 None，将自动生成配置
# 如果设为路径字符串，将使用指定的配置文件
DEEPSPEED_CONFIG_PATH = None  # 例如: "./ds_config.json"


# ============================================================================
# DeepSpeed 配置生成
# ============================================================================

def create_deepspeed_config(
    output_dir: str,
    per_device_batch_size: int = 2,
    gradient_accumulation_steps: int = 8,
    max_grad_norm: float = 1.0,
    logging_steps: int = 10,
) -> str:
    """
    创建 DeepSpeed ZeRO-2 配置文件
    
    【DeepSpeed ZeRO 优化级别说明】
    - ZeRO Stage 0: 无优化，基准
    - ZeRO Stage 1: 优化器状态分片
    - ZeRO Stage 2: 优化器状态 + 梯度分片（推荐用于微调）
    - ZeRO Stage 3: 优化器状态 + 梯度 + 模型参数分片（最省显存，但较慢）
    
    本配置使用 ZeRO Stage 2 + CPU Offloading，适合：
    - 单机多卡训练
    - 显存受限环境（如 24GB 显卡训练 7B 模型）
    - 适配器微调（只训练少量参数）
    
    Args:
        output_dir: 输出目录
        per_device_batch_size: 每个 GPU 的 batch size
        gradient_accumulation_steps: 梯度累积步数
        max_grad_norm: 梯度裁剪阈值
        logging_steps: 日志输出间隔
    
    Returns:
        配置文件路径
    """
    
    # 获取 world size（GPU 数量）
    if dist.is_available() and dist.is_initialized():
        world_size = dist.get_world_size()
    else:
        # 尝试从环境变量获取
        world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    # 计算全局 batch size
    train_batch_size = per_device_batch_size * gradient_accumulation_steps * world_size

    # 原始ds_config设置    
    # # DeepSpeed 配置
    # ds_config = {
    #     # ============================================================
    #     # 批次配置
    #     # ============================================================
    #     # 全局 batch size = per_device * accumulation * world_size
    #     "train_batch_size": train_batch_size,
        
    #     # 每个 GPU 每次前向传播的 batch size
    #     "train_micro_batch_size_per_gpu": per_device_batch_size,
        
    #     # 梯度累积步数
    #     "gradient_accumulation_steps": gradient_accumulation_steps,
        
    #     # ============================================================
    #     # 精度配置
    #     # ============================================================
    #     # 使用 BFloat16 混合精度（比 FP16 更稳定，A100/H100 推荐）
    #     "bf16": {
    #         "enabled": True
    #     },
        
    #     # 禁用 FP16（与 BF16 互斥）
    #     "fp16": {
    #         "enabled": False
    #     },
        
    #     # ============================================================
    #     # ZeRO 优化配置
    #     # ============================================================
    #     "zero_optimization": {
    #         # ZeRO Stage 2: 优化器状态 + 梯度分片
    #         # 显存节省约 50-60%，通信开销适中
    #         "stage": 2,
            
    #         # 将优化器状态 offload 到 CPU
    #         # Adam 优化器的 momentum 和 variance 会占用大量显存
    #         # Offload 后可节省约 2x 模型大小的显存
    #         "offload_optimizer": {
    #             "device": "cpu",           # 目标设备
    #             "pin_memory": True,        # 使用锁页内存加速传输
    #             "buffer_count": 4,         # 缓冲区数量
    #             "fast_init": False         # 快速初始化（可能不稳定）
    #         },
            
    #         # 是否 offload 参数到 CPU（Stage 3 时生效）
    #         # Stage 2 时此选项无效
    #         # "offload_param": {
    #         #     "device": "cpu",
    #         #     "pin_memory": True
    #         # },
            
    #         # AllGather 优化
    #         "allgather_partitions": True,
    #         "allgather_bucket_size": 5e8,
            
    #         # Reduce-Scatter 优化
    #         "reduce_scatter": True,
    #         "reduce_bucket_size": 5e8,
            
    #         # 通信与计算重叠
    #         "overlap_comm": True,
            
    #         # 使用连续梯度存储（减少内存碎片）
    #         "contiguous_gradients": True,
            
    #         # 子组大小（影响通信效率）
    #         # "sub_group_size": 1e9,
            
    #         # 是否在第一步后减少 bucket 大小
    #         "reduce_bucket_size": 1280*1280*2,#"auto",
            
    #         # 阶段 3 相关选项（此处不使用）
    #         # "stage3_prefetch_bucket_size": "auto",
    #         # "stage3_param_persistence_threshold": "auto",
    #         # "stage3_max_live_parameters": 1e9,
    #         # "stage3_max_reuse_distance": 1e9,
    #         # "stage3_gather_16bit_weights_on_model_save": True,
    #     },
        
    #     # ============================================================
    #     # 梯度配置
    #     # ============================================================
    #     # 梯度裁剪
    #     "gradient_clipping": max_grad_norm,
        
    #     # ============================================================
    #     # 激活检查点（节省显存，但增加计算时间）
    #     # ============================================================
    #     # 注意：这里不启用，因为我们只训练 adapter
    #     # 如果需要，可以在模型中手动启用 gradient_checkpointing
    #     # "activation_checkpointing": {
    #     #     "partition_activations": True,
    #     #     "cpu_checkpointing": True,
    #     #     "contiguous_memory_optimization": True,
    #     #     "number_checkpoints": None,
    #     #     "synchronize_checkpoint_boundary": False,
    #     #     "profile": False
    #     # },
        
    #     # ============================================================
    #     # 日志与调试
    #     # ============================================================
    #     "steps_per_print": logging_steps,
    #     "wall_clock_breakdown": False,
        
    #     # 禁用不需要的功能
    #     "dump_state": False,
    # }
    ds_config = {
        "train_batch_size": 128,
        "train_micro_batch_size_per_gpu": 2,
        "gradient_accumulation_steps": 8,

        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": 2e-4,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "weight_decay": 0.01
            }
        },

        "scheduler": {
            "type": "WarmupLR",
            "params": {
                "warmup_min_lr": 0,
                "warmup_max_lr": 2e-4,
                "warmup_num_steps": "auto"  # 关键修改
            }
        },

        "zero_optimization": {
            "stage": 2,
            "contiguous_gradients": True,
            "overlap_comm": True,
            "reduce_scatter": True,
            "allgather_partitions": True,
            "allgather_bucket_size": 5e8,
            "reduce_bucket_size": 5e8,
        },

        "fp16": {"enabled": False},
        "bf16": {"enabled": True},

        "gradient_clipping": 1.0,
        "steps_per_print": 10,
        "wall_clock_breakdown": False,
    }

    # 保存配置文件
    config_path = os.path.join(output_dir, "ds_config_auto.json")
    os.makedirs(output_dir, exist_ok=True)
    
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(ds_config, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print("DeepSpeed 配置已生成")
    print(f"{'='*60}")
    print(f"配置文件: {config_path}")
    print(f"ZeRO Stage: 2")
    print(f"CPU Offloading: 优化器状态")
    print(f"精度: BFloat16")
    print(f"全局 Batch Size: {train_batch_size}")
    print(f"每 GPU Batch Size: {per_device_batch_size}")
    print(f"梯度累积: {gradient_accumulation_steps}")
    print(f"World Size: {world_size}")
    print(f"{'='*60}\n")
    
    return config_path


# ============================================================================
# 主训练函数
# ============================================================================

def main():
    """
    主训练入口
    
    执行流程：
    1. 解析命令行参数（可覆盖配置）
    2. 验证路径和环境
    3. 创建/加载 DeepSpeed 配置
    4. 初始化模型
    5. 加载数据集
    6. 初始化训练器
    7. 执行训练
    8. 保存最终模型
    """
    
    # ==================== 解析命令行参数 ====================
    parser = argparse.ArgumentParser(description="MEMModel Adapter Training Script")
    
    # 模型路径参数（可覆盖配置文件中的设置）
    parser.add_argument(
        "--base_model_path", 
        type=str, 
        default=None,
        help="基础语言模型的本地路径"
    )
    parser.add_argument(
        "--ocr_model_path", 
        type=str, 
        default=None,
        help="OCR 编码器的本地路径"
    )
    
    # 数据路径参数
    parser.add_argument(
        "--train_data", 
        type=str, 
        default=None,
        help="训练数据 JSONL 文件路径"
    )
    parser.add_argument(
        "--eval_data", 
        type=str, 
        default=None,
        help="验证数据 JSONL 文件路径"
    )
    
    # 训练参数
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default=None,
        help="输出目录"
    )
    parser.add_argument(
        "--num_epochs", 
        type=int, 
        default=None,
        help="训练轮数"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=None,
        help="每 GPU 训练 batch size"
    )
    parser.add_argument(
        "--learning_rate", 
        type=float, 
        default=None,
        help="学习率"
    )
    parser.add_argument(
        "--gradient_accumulation", 
        type=int, 
        default=None,
        help="梯度累积步数"
    )
    
    # DeepSpeed 配置
    parser.add_argument(
        "--deepspeed", 
        type=str, 
        default=None,
        help="DeepSpeed 配置文件路径（如不指定则自动生成）"
    )
    parser.add_argument(
        "--local_rank", 
        type=int, 
        default=-1,
        help="分布式训练的 local rank（由 DeepSpeed 自动设置）"
    )
    
    # WandB 配置
    parser.add_argument(
        "--wandb_project", 
        type=str, 
        default=None,
        help="WandB 项目名称"
    )
    parser.add_argument(
        "--wandb_run_name", 
        type=str, 
        default=None,
        help="WandB 实验名称"
    )
    parser.add_argument(
        "--disable_wandb", 
        action="store_true",
        help="禁用 WandB 日志"
    )
    
    args = parser.parse_args()
    
    # ==================== 合并配置 ====================
    # 命令行参数优先级高于配置文件
    
    base_model_path = args.base_model_path or MODEL_CONFIG["base_model_path"]
    ocr_model_path = args.ocr_model_path or MODEL_CONFIG["ocr_model_path"]
    train_data_path = args.train_data or DATA_CONFIG["train_data_path"]
    eval_data_path = args.eval_data or DATA_CONFIG["eval_data_path"]
    output_dir = args.output_dir or TRAINING_CONFIG["output_dir"]
    num_epochs = args.num_epochs or TRAINING_CONFIG["num_train_epochs"]
    batch_size = args.batch_size or TRAINING_CONFIG["per_device_train_batch_size"]
    learning_rate = args.learning_rate or TRAINING_CONFIG["learning_rate"]
    gradient_accumulation = args.gradient_accumulation or TRAINING_CONFIG["gradient_accumulation_steps"]
    
    # WandB 配置
    wandb_project = args.wandb_project or TRAINING_CONFIG.get("wandb_project", "OCR-MEM")
    wandb_run_name = args.wandb_run_name or TRAINING_CONFIG["run_name"]
    use_wandb = WANDB_AVAILABLE and not args.disable_wandb
    
    # ==================== 验证路径 ====================
    print(f"\n{'='*60}")
    print("验证配置和路径")
    print(f"{'='*60}\n")
    
    # 检查模型路径
    if not os.path.exists(base_model_path):
        print(f"⚠️  警告: 基础模型路径不存在: {base_model_path}")
        print(f"   请确保路径正确，或者这是一个 HuggingFace 模型 ID")
    else:
        print(f"✓ 基础模型路径: {base_model_path}")
    
    if not os.path.exists(ocr_model_path):
        print(f"⚠️  警告: OCR 模型路径不存在: {ocr_model_path}")
        print(f"   请确保路径正确，或者这是一个 HuggingFace 模型 ID")
    else:
        print(f"✓ OCR 模型路径: {ocr_model_path}")
    
    # 检查数据路径
    if not os.path.exists(train_data_path):
        raise FileNotFoundError(f"训练数据文件不存在: {train_data_path}")
    print(f"✓ 训练数据: {train_data_path}")
    
    if eval_data_path and os.path.exists(eval_data_path):
        print(f"✓ 验证数据: {eval_data_path}")
    else:
        print(f"⚠️  未找到验证数据，将跳过验证: {eval_data_path}")
        eval_data_path = None
    
    print(f"✓ 输出目录: {output_dir}")
    
    # ==================== 创建 DeepSpeed 配置 ====================
    print(f"\n{'='*60}")
    print("配置 DeepSpeed")
    print(f"{'='*60}")
    
    deepspeed_config = args.deepspeed or DEEPSPEED_CONFIG_PATH
    
    if deepspeed_config is None:
        # 自动生成 DeepSpeed 配置
        deepspeed_config = create_deepspeed_config(
            output_dir=output_dir,
            per_device_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation,
            max_grad_norm=TRAINING_CONFIG["max_grad_norm"],
            logging_steps=TRAINING_CONFIG["logging_steps"],
        )
    elif os.path.exists(deepspeed_config):
        print(f"✓ 使用指定的 DeepSpeed 配置: {deepspeed_config}")
    else:
        raise FileNotFoundError(f"DeepSpeed 配置文件不存在: {deepspeed_config}")
    
    # ==================== 创建训练参数 ====================
    print(f"\n{'='*60}")
    print("创建训练参数")
    print(f"{'='*60}\n")
    
    training_args = AdapterTrainingArguments(
        # 输出配置
        output_dir=output_dir,
        run_name=TRAINING_CONFIG["run_name"],
        
        # 模型配置
        base_model_name=base_model_path,
        ocr_model_name=ocr_model_path,
        vision_embedding_size=MODEL_CONFIG["vision_embedding_size"],
        context_threshold=MODEL_CONFIG["context_threshold"],
        
        # 数据配置
        train_data_path=train_data_path,
        eval_data_path=eval_data_path,
        max_seq_length=DATA_CONFIG["max_seq_length"],
        
        # 训练超参数
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=TRAINING_CONFIG["per_device_eval_batch_size"],
        gradient_accumulation_steps=gradient_accumulation,
        
        # 优化器配置
        learning_rate=learning_rate,
        warmup_ratio=TRAINING_CONFIG["warmup_ratio"],
        weight_decay=TRAINING_CONFIG["weight_decay"],
        max_grad_norm=TRAINING_CONFIG["max_grad_norm"],
        
        # 精度配置
        bf16=True,
        tf32=True,
        
        # 评估与保存
        eval_strategy=TRAINING_CONFIG["eval_strategy"] if eval_data_path else "no",
        eval_steps=TRAINING_CONFIG["eval_steps"],
        save_strategy=TRAINING_CONFIG["save_strategy"],
        save_steps=TRAINING_CONFIG["save_steps"],
        save_total_limit=TRAINING_CONFIG["save_total_limit"],
        logging_steps=TRAINING_CONFIG["logging_steps"],
        
        # DeepSpeed
        deepspeed=deepspeed_config,
        
        # 其他配置
        dataloader_num_workers=TRAINING_CONFIG["dataloader_num_workers"],
        remove_unused_columns=False,
        report_to=["wandb", "tensorboard"] if use_wandb else ["tensorboard"],
        seed=TRAINING_CONFIG["seed"],
        
        # 适配器配置
        save_adapter_only=True,
    )
    
    # ==================== 初始化 WandB ====================
    if use_wandb:
        # 只在主进程初始化 wandb
        local_rank = int(os.environ.get("LOCAL_RANK", -1))
        if local_rank <= 0:
            print(f"\n{'='*60}")
            print("初始化 WandB")
            print(f"{'='*60}\n")
            
            wandb.init(
                project=wandb_project,
                name=wandb_run_name,
                config={
                    "base_model": base_model_path,
                    "ocr_model": ocr_model_path,
                    "num_epochs": num_epochs,
                    "batch_size": batch_size,
                    "gradient_accumulation": gradient_accumulation,
                    "learning_rate": learning_rate,
                    "max_seq_length": DATA_CONFIG["max_seq_length"],
                    "vision_embedding_size": MODEL_CONFIG["vision_embedding_size"],
                    "context_threshold": MODEL_CONFIG["context_threshold"],
                },
                reinit=True,
            )
            print(f"✓ WandB 项目: {wandb_project}")
            print(f"✓ WandB 实验: {wandb_run_name}")
    
    print(f"\n✓ 训练轮数: {num_epochs}")
    print(f"✓ 每 GPU Batch Size: {batch_size}")
    print(f"✓ 梯度累积: {gradient_accumulation}")
    print(f"✓ 学习率: {learning_rate}")
    
    # ==================== 初始化模型 ====================
    print(f"\n{'='*60}")
    print("初始化模型")
    print(f"{'='*60}\n")
    
    # 创建模型配置
    model_config = MEMConfig(
        base_model_name=base_model_path,
        ocr_model_name=ocr_model_path,
        vision_embedding_size=MODEL_CONFIG["vision_embedding_size"],
        context_threshold=MODEL_CONFIG["context_threshold"],
    )
    
    # 初始化模型
    print("正在加载模型，这可能需要几分钟...")
    model = MEMModel(config=model_config)
    tokenizer = model.tokenizer
    
    # 确保 tokenizer 有 pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    print(f"✓ 模型加载完成")
    model.print_trainable_parameters()
    
    # ==================== 加载数据集 ====================
    print(f"\n{'='*60}")
    print("加载数据集")
    print(f"{'='*60}\n")
    
    train_dataset = MultiTurnConversationDataset(
        jsonl_path=train_data_path,
        tokenizer=tokenizer,
        max_length=DATA_CONFIG["max_seq_length"],
        return_dict=False,
    )
    print(f"✓ 训练集: {len(train_dataset)} 样本")
    
    eval_dataset = None
    if eval_data_path:
        eval_dataset = MultiTurnConversationDataset(
            jsonl_path=eval_data_path,
            tokenizer=tokenizer,
            max_length=DATA_CONFIG["max_seq_length"],
            return_dict=False,
        )
        print(f"✓ 验证集: {len(eval_dataset)} 样本")
    
    # ==================== 初始化训练器 ====================
    print(f"\n{'='*60}")
    print("初始化训练器")
    print(f"{'='*60}\n")
    
    trainer = AdapterOnlyTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=partial(collate_fn, tokenizer=tokenizer),
    )
    
    print("✓ 训练器初始化完成")
    
    # ==================== 开始训练 ====================
    print(f"\n{'='*60}")
    print("开始训练")
    print(f"{'='*60}\n")
    
    # 检查是否有可恢复的检查点
    checkpoint = None
    if os.path.exists(output_dir):
        checkpoints = [
            d for d in os.listdir(output_dir) 
            if d.startswith("checkpoint-")
        ]
        if checkpoints:
            latest = max(checkpoints, key=lambda x: int(x.split("-")[-1]))
            checkpoint = os.path.join(output_dir, latest)
            print(f"✓ 发现检查点，将从 {checkpoint} 恢复训练\n")
    
    # 开始训练
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    
    # ==================== 保存最终模型 ====================
    final_dir = os.path.join(output_dir, "final_adapter")
    trainer.save_model(final_dir)
    
    # 保存训练指标
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    
    # 完成提示
    print(f"\n{'='*60}")
    print("🎉 训练完成！")
    print(f"{'='*60}")
    print(f"✓ 最终适配器: {final_dir}")
    print(f"✓ 训练损失: {metrics.get('train_loss', 'N/A'):.4f}")
    print(f"✓ 总步数: {metrics.get('train_steps', 'N/A')}")
    print(f"{'='*60}\n")
    
    # 关闭 WandB
    if use_wandb:
        local_rank = int(os.environ.get("LOCAL_RANK", -1))
        if local_rank <= 0:
            wandb.finish()
            print("✓ WandB 日志已同步完成")


# ============================================================================
# 程序入口
# ============================================================================

if __name__ == "__main__":
    main()
