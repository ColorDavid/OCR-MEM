"""
MEM Model SFT Trainer with DeepSpeed Support
基于 Transformers Trainer 的视觉记忆压缩模型训练器
"""

import os
import json
import torch
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from pathlib import Path

from transformers import (
    Trainer,
    TrainingArguments,
    AutoTokenizer,
    EvalPrediction,
    TrainerCallback,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.integrations import is_deepspeed_available

from utils.data import create_dataloader, MultiTurnConversationDataset
from mem_adapter_only import MEMModel,MEMConfig  # 使用 adapter-only 版本，因为它继承了 PreTrainedModel

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """模型相关参数"""
    base_model_name: str = field(
        metadata={"help": "基础语言模型的路径或名称"}
    )
    ocr_model_name: str = field(
        metadata={"help": "OCR 编码器的路径或名称"}
    )
    vision_embedding_size: int = field(
        default=512,
        metadata={"help": "视觉嵌入维度"}
    )
    context_threshold: int = field(
        default=2048,
        metadata={"help": "触发视觉压缩的上下文长度阈值"}
    )
    render_config_path: Optional[str] = field(
        default=None,
        metadata={"help": "文本渲染配置文件路径"}
    )


@dataclass
class DataArguments:
    """数据相关参数"""
    train_data_path: str = field(
        metadata={"help": "训练数据 JSONL 文件路径"}
    )
    eval_data_path: Optional[str] = field(
        default=None,
        metadata={"help": "验证数据 JSONL 文件路径"}
    )
    max_length: int = field(
        default=2048,
        metadata={"help": "最大序列长度"}
    )
    preprocessing_num_workers: int = field(
        default=4,
        metadata={"help": "数据预处理的进程数"}
    )


@dataclass
class MEMTrainingArguments(TrainingArguments):
    """扩展的训练参数"""
    # DeepSpeed 相关
    deepspeed_config: Optional[str] = field(
        default=None,
        metadata={"help": "DeepSpeed 配置文件路径"}
    )
    
    # 梯度检查点
    gradient_checkpointing: bool = field(
        default=True,
        metadata={"help": "是否使用梯度检查点以节省显存"}
    )
    
    # 混合精度训练
    bf16: bool = field(
        default=True,
        metadata={"help": "是否使用 bfloat16 混合精度训练"}
    )
    
    # 日志和保存
    logging_steps: int = field(
        default=10,
        metadata={"help": "每 N 步记录一次日志"}
    )
    save_steps: int = field(
        default=500,
        metadata={"help": "每 N 步保存一次模型"}
    )
    save_total_limit: int = field(
        default=3,
        metadata={"help": "最多保留的检查点数量"}
    )
    
    # 训练策略
    warmup_steps: int = field(
        default=100,
        metadata={"help": "学习率预热步数"}
    )
    max_grad_norm: float = field(
        default=1.0,
        metadata={"help": "梯度裁剪的最大范数"}
    )
    
    # 评估
    eval_steps: Optional[int] = field(
        default=500,
        metadata={"help": "每 N 步进行一次评估"}
    )
    evaluation_strategy: str = field(
        default="steps",
        metadata={"help": "评估策略"}
    )


class MEMTrainer(Trainer):
    """
    自定义 Trainer，专门处理 MEMModel 的训练
    """
    
    def __init__(
        self,
        model: MEMModel = None,
        args: MEMTrainingArguments = None,
        train_dataset=None,
        eval_dataset=None,
        tokenizer=None,
        **kwargs
    ):
        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            **kwargs
        )
        
        # 启用梯度检查点
        if args.gradient_checkpointing and hasattr(model.base_model, 'gradient_checkpointing_enable'):
            model.base_model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled")
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        重写损失计算函数，确保正确处理 MEMModel 的输出
        """
        # MEMModel 的 forward 已经返回包含 loss 的 CausalLMOutputWithPast
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=inputs.get("labels"),
        )
        
        loss = outputs.loss
        
        return (loss, outputs) if return_outputs else loss
    
    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        """
        保存模型，确保保存所有必要的组件
        """
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"Saving model checkpoint to {output_dir}")
        
        # 保存模型状态
        if state_dict is None:
            state_dict = self.model.state_dict()
        
        # 保存完整模型
        torch.save(state_dict, os.path.join(output_dir, "pytorch_model.bin"))
        
        # 保存配置
        model_config = {
            "base_model_name": self.model.base_model.config._name_or_path,
            "ocr_model_name": "deepseek-ai/DeepSeek-OCR",  # 根据实际情况修改
            "vision_embedding_size": self.model.proj[0].in_features,
            "context_threshold": self.model.context_threshold,
        }
        
        with open(os.path.join(output_dir, "model_config.json"), "w") as f:
            json.dump(model_config, f, indent=2)
        
        # 保存 tokenizer
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)
        
        logger.info(f"Model saved to {output_dir}")


class LoggingCallback(TrainerCallback):
    """自定义日志回调"""
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """记录训练日志"""
        if state.is_local_process_zero:
            # 计算并记录额外的指标
            if logs is not None:
                step = state.global_step
                
                # 记录学习率
                if "learning_rate" in logs:
                    logger.info(f"Step {step}: LR = {logs['learning_rate']:.2e}")
                
                # 记录损失
                if "loss" in logs:
                    logger.info(f"Step {step}: Loss = {logs['loss']:.4f}")
                
                # 记录评估指标
                if "eval_loss" in logs:
                    logger.info(f"Step {step}: Eval Loss = {logs['eval_loss']:.4f}")


def create_deepspeed_config(
    output_dir: str,
    per_device_train_batch_size: int = 1,
    gradient_accumulation_steps: int = 16,
    learning_rate: float = 2e-5,
    bf16: bool = True,
) -> str:
    """
    创建 DeepSpeed ZeRO-2 配置文件
    
    Args:
        output_dir: 输出目录
        per_device_train_batch_size: 每设备批次大小
        gradient_accumulation_steps: 梯度累积步数
        learning_rate: 学习率
        bf16: 是否使用 bfloat16
    
    Returns:
        配置文件路径
    """
    config = {
        "train_batch_size": "auto",
        "train_micro_batch_size_per_gpu": "auto",
        "gradient_accumulation_steps": "auto",
        
        # ZeRO 优化配置
        "zero_optimization": {
            "stage": 2,  # ZeRO-2: 分片优化器状态和梯度
            "offload_optimizer": {
                "device": "cpu",  # 将优化器状态卸载到 CPU
                "pin_memory": True
            },
            "allgather_partitions": True,
            "allgather_bucket_size": 5e8,
            "reduce_scatter": True,
            "reduce_bucket_size": 5e8,
            "overlap_comm": True,
            "contiguous_gradients": True,
        },
        
        # 混合精度训练
        "bf16": {
            "enabled": bf16
        },
        
        # 梯度裁剪
        "gradient_clipping": 1.0,
        
        # 检查点
        "steps_per_print": 10,
        
        # 通信优化
        "communication_data_type": "bf16" if bf16 else "fp32",
        
        # 激活检查点（梯度检查点）
        "activation_checkpointing": {
            "partition_activations": True,
            "cpu_checkpointing": False,
            "contiguous_memory_optimization": True,
            "number_checkpoints": None,
        },
        
        # Wall clock breakdown
        "wall_clock_breakdown": False,
    }
    
    # 保存配置
    config_path = os.path.join(output_dir, "ds_config.json")
    os.makedirs(output_dir, exist_ok=True)
    
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"DeepSpeed config saved to {config_path}")
    return config_path


def main():
    """主训练函数"""
    from transformers import HfArgumentParser
    
    # 解析命令行参数
    parser = HfArgumentParser((ModelArguments, DataArguments, MEMTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    # 设置日志级别
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    
    logger.info(f"Training/evaluation parameters {training_args}")
    
    # 检测最后的检查点
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is not None:
            logger.info(f"Checkpoint detected, resuming training at {last_checkpoint}")
    
    # 创建 DeepSpeed 配置
    if training_args.deepspeed_config is None and is_deepspeed_available():
        training_args.deepspeed = create_deepspeed_config(
            output_dir=training_args.output_dir,
            per_device_train_batch_size=training_args.per_device_train_batch_size,
            gradient_accumulation_steps=training_args.gradient_accumulation_steps,
            learning_rate=training_args.learning_rate,
            bf16=training_args.bf16,
        )
        logger.info(f"Using auto-generated DeepSpeed config: {training_args.deepspeed}")
    
    # 加载 tokenizer
    logger.info(f"Loading tokenizer from {model_args.base_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_args.base_model_name)
    
    # 确保 tokenizer 有 pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # 创建模型
    logger.info("Initializing MEMModel")
    model = MEMModel(
        base_model_name=model_args.base_model_name,
        ocr_model_name=model_args.ocr_model_name,
        vision_embedding_size=model_args.vision_embedding_size,
        context_threshold=model_args.context_threshold,
    )
    
    # 冻结 base_model 的某些层（可选）
    # 例如：只训练最后几层和投影层
    # for name, param in model.base_model.named_parameters():
    #     if "layers.0." in name or "layers.1." in name:  # 冻结前2层
    #         param.requires_grad = False
    
    logger.info(f"Model initialized with {sum(p.numel() for p in model.parameters())} total parameters")
    logger.info(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    # 加载训练数据
    logger.info(f"Loading training data from {data_args.train_data_path}")
    train_dataset = MultiTurnConversationDataset(
        jsonl_path=data_args.train_data_path,
        tokenizer=tokenizer,
        max_length=data_args.max_length,
        return_dict=False,  # Trainer 不需要额外的字段
    )
    
    # 加载验证数据
    eval_dataset = None
    if data_args.eval_data_path is not None:
        logger.info(f"Loading evaluation data from {data_args.eval_data_path}")
        eval_dataset = MultiTurnConversationDataset(
            jsonl_path=data_args.eval_data_path,
            tokenizer=tokenizer,
            max_length=data_args.max_length,
            return_dict=False,
        )
    
    # 数据整理函数（已在 dataset 中的 collate_fn 中实现）
    from utils.data import collate_fn
    data_collator = lambda batch: collate_fn(batch, tokenizer)
    
    # 创建 Trainer
    trainer = MEMTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[LoggingCallback()],
    )
    
    # 开始训练
    if training_args.do_train:
        logger.info("*** Starting training ***")
        
        checkpoint = None
        if last_checkpoint is not None:
            checkpoint = last_checkpoint
        elif training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        
        # 保存模型
        trainer.save_model()
        
        # 保存训练指标
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        
        logger.info("*** Training completed ***")
    
    # 评估
    if training_args.do_eval:
        logger.info("*** Evaluating ***")
        metrics = trainer.evaluate()
        
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        
        logger.info("*** Evaluation completed ***")


if __name__ == "__main__":
    main()