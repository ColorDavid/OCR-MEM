"""
================================================================================
Adapter-Only SFT Trainer for MEMModel
================================================================================

本模块实现了一个专门用于训练 MEMModel 投影适配器 (Projection Adapter) 的训练器。

核心设计理念：
1. 只训练投影层 (proj)，冻结基础语言模型和 OCR 编码器
2. 使用 DeepSpeed ZeRO-2 优化，支持 CPU offloading 以节省 GPU 显存
3. 只保存适配器权重，而非整个模型，极大减少存储需求

适用场景：
- 在预训练的视觉-语言模型基础上微调投影层
- 资源受限环境下的高效训练
- 快速迭代实验

依赖：
- transformers: HuggingFace Trainer 和 TrainingArguments
- deepspeed: 分布式训练和内存优化 (可选)
- torch: PyTorch 深度学习框架

作者：OCR-MEM Team
更新日期：2024
================================================================================
"""

import os
import json
import torch
import torch.distributed as dist
from typing import Optional, Dict, Any, Union, List
from dataclasses import dataclass, field
from functools import partial

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from transformers import (
    Trainer,
    TrainingArguments,
)
from transformers.modeling_utils import unwrap_model

# 导入自定义模型和数据处理模块
from mem_adapter_only import MEMModel, MEMConfig
from utils.data import MultiTurnConversationDataset, collate_fn


# ============================================================================
# 训练参数配置类
# ============================================================================

@dataclass
class AdapterTrainingArguments(TrainingArguments):
    """
    扩展的训练参数类，继承自 HuggingFace TrainingArguments
    
    新增参数分为三类：
    1. 模型配置 (Model Config): 定义模型结构相关参数
    2. 数据配置 (Data Config): 定义数据集路径和处理参数
    3. 适配器配置 (Adapter Config): 控制适配器保存行为
    
    使用示例：
        args = AdapterTrainingArguments(
            output_dir="./checkpoints",
            base_model_name="Qwen/Qwen2.5-1.5B-Instruct",
            per_device_train_batch_size=2,
            learning_rate=2e-4,
        )
    """
    
    # -------------------- 模型配置 --------------------
    base_model_name: str = field(
        default="Qwen/Qwen2.5-1.5B-Instruct",
        metadata={"help": "基础语言模型的 HuggingFace 模型名称或本地路径"}
    )
    ocr_model_name: str = field(
        default="deepseek-ai/DeepSeek-OCR",
        metadata={"help": "OCR 视觉编码器的模型名称或路径"}
    )
    vision_embedding_size: int = field(
        default=512,
        metadata={"help": "视觉编码器输出的嵌入维度"}
    )
    context_threshold: int = field(
        default=2048,
        metadata={"help": "触发视觉压缩的上下文长度阈值（超过此长度将压缩历史为视觉嵌入）"}
    )
    
    # -------------------- 数据配置 --------------------
    train_data_path: str = field(
        default="data/train.jsonl",
        metadata={"help": "训练数据 JSONL 文件路径"}
    )
    eval_data_path: Optional[str] = field(
        default=None,
        metadata={"help": "验证数据 JSONL 文件路径（可选）"}
    )
    max_seq_length: int = field(
        default=2048,
        metadata={"help": "最大序列长度，超过此长度将被截断"}
    )
    
    # -------------------- 适配器配置 --------------------
    save_adapter_only: bool = field(
        default=True,
        metadata={"help": "是否只保存适配器权重（True）或完整模型（False）"}
    )


# ============================================================================
# 适配器训练器类
# ============================================================================

class AdapterOnlyTrainer(Trainer):
    """
    专门用于训练 MEMModel 投影适配器的训练器
    
    核心功能：
    1. 自动冻结非适配器参数，只训练 proj 层
    2. 自定义保存逻辑，只保存适配器权重
    3. 与 MEMModel 的 save_pretrained/from_pretrained 兼容
    4. 支持 DeepSpeed ZeRO 优化
    
    继承自 HuggingFace Trainer，保留所有原有功能：
    - 分布式训练支持
    - 混合精度训练 (bf16/fp16)
    - 梯度累积
    - 学习率调度
    - 日志记录和检查点保存
    
    使用示例：
        trainer = AdapterOnlyTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            tokenizer=tokenizer,
        )
        trainer.train()
    """
    
    def __init__(
        self, 
        model: MEMModel, 
        args: AdapterTrainingArguments, 
        **kwargs
    ):
        """
        初始化适配器训练器
        
        Args:
            model (MEMModel): MEMModel 实例，将自动冻结非适配器参数
            args (AdapterTrainingArguments): 训练参数配置
            **kwargs: 传递给父类 Trainer 的其他参数，如：
                - train_dataset: 训练数据集
                - eval_dataset: 验证数据集
                - tokenizer: 分词器
                - data_collator: 数据整理函数
                - callbacks: 回调函数列表
        
        注意：
            模型参数冻结在调用父类 __init__ 之前完成，
            确保优化器只包含可训练参数
        """
        # 在调用父类初始化之前，设置可训练参数
        # 这很重要，因为 Trainer 会根据 requires_grad 创建优化器
        self._setup_trainable_params(model, args)
        
        # 调用父类初始化
        super().__init__(model=model, args=args, **kwargs)
    
    def _setup_trainable_params(
        self, 
        model: MEMModel, 
        args: AdapterTrainingArguments
    ) -> None:
        """
        设置模型的可训练参数
        
        策略：
        1. 首先冻结所有参数
        2. 然后只解冻 proj (投影层) 的参数
        3. 验证只有预期的参数是可训练的
        
        Args:
            model: MEMModel 实例
            args: 训练参数（当前未使用，预留扩展）
        
        Raises:
            ValueError: 如果发现非 proj 层的参数是可训练的
        
        效果：
            - 基础语言模型参数: 冻结
            - OCR 编码器参数: 冻结
            - 投影层参数: 可训练
        """
        # Step 1: 冻结所有参数
        for param in model.parameters():
            param.requires_grad = False
        
        # Step 2: 解冻投影适配器
        for param in model.proj.parameters():
            param.requires_grad = True
        
        # Step 3: 统计并打印参数信息
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        
        print(f"\n{'='*60}")
        print(f"参数统计 (Parameter Statistics)")
        print(f"{'='*60}")
        print(f"可训练参数 (Trainable):     {trainable:,}")
        print(f"总参数 (Total):             {total:,}")
        print(f"可训练比例 (Trainable %):   {100*trainable/total:.4f}%")
        print(f"{'='*60}\n")
        
        # Step 4: 安全检查 - 确保只有 proj 层是可训练的
        for name, param in model.named_parameters():
            if param.requires_grad and 'proj' not in name:
                raise ValueError(
                    f"安全检查失败: 参数 '{name}' 是可训练的，但它不属于 proj 层！\n"
                    f"这可能导致意外的参数更新。请检查模型结构。"
                )
    
    def log(self, logs: Dict[str, float]) -> None:
        """
        重写日志方法，添加额外的训练监控指标
        
        Args:
            logs: 日志字典
        """
        # 添加额外的监控指标
        if self.state.global_step > 0:
            # 添加学习率信息
            if "learning_rate" not in logs and self.lr_scheduler is not None:
                logs["learning_rate"] = self.lr_scheduler.get_last_lr()[0]
            
            # 添加训练进度
            if hasattr(self.state, "max_steps") and self.state.max_steps > 0:
                logs["progress"] = self.state.global_step / self.state.max_steps
        
        # 调用父类的 log 方法（会自动发送到 wandb）
        super().log(logs)
    
    def compute_loss(
        self, 
        model: MEMModel, 
        inputs: Dict[str, torch.Tensor], 
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None,
    ) -> Union[torch.Tensor, tuple]:
        """
        计算训练损失
        
        重写此方法以确保与 MEMModel 的 forward 方法正确配合。
        MEMModel.forward 返回包含 loss 的 CausalLMOutputWithPast 对象。
        
        Args:
            model: 模型实例（可能被 DeepSpeed/FSDP 包装）
            inputs: 输入字典，包含：
                - input_ids: 输入 token IDs [batch_size, seq_len]
                - attention_mask: 注意力掩码 [batch_size, seq_len]
                - labels: 标签，-100 表示不计算损失 [batch_size, seq_len]
            return_outputs: 是否返回模型输出
            num_items_in_batch: 批次中的样本数（用于梯度累积归一化）
        
        Returns:
            如果 return_outputs=False: 返回 loss (torch.Tensor)
            如果 return_outputs=True: 返回 (loss, outputs) 元组
        """
        # 移除非模型参数（如 messages 等调试信息）
        # 这些可能由 data_collator 添加，但不应传入模型
        model_inputs = {
            k: v for k, v in inputs.items() 
            if k in ['input_ids', 'attention_mask', 'labels', 'position_ids']
        }
        
        # 前向传播
        outputs = model(**model_inputs)
        
        # 获取损失
        # MEMModel.forward 在有 labels 时会自动计算损失
        loss = outputs.loss
        
        # 可选：添加额外的损失监控日志
        if self.args.logging_steps > 0 and self.state.global_step % self.args.logging_steps == 0:
            if hasattr(self, '_log_loss_once'):
                pass  # 避免重复日志
            else:
                self._log_loss_once = True
        
        return (loss, outputs) if return_outputs else loss
    
    def _save(
        self, 
        output_dir: Optional[str] = None, 
        state_dict: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        保存模型检查点
        
        根据 save_adapter_only 配置：
        - True: 只保存适配器权重（推荐，节省存储空间）
        - False: 保存完整模型
        
        保存的文件格式与 MEMModel.from_pretrained 兼容：
        - config.json: 模型配置
        - projection_layer.pt: 投影层权重
        
        Args:
            output_dir: 输出目录，默认使用 args.output_dir
            state_dict: 可选的 state_dict，通常由 DeepSpeed 提供
        
        注意：
            在分布式训练中，只有主进程 (rank 0) 会执行保存操作
        """
        output_dir = output_dir or self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 获取实际的模型（可能被 DataParallel/DeepSpeed 包装）
        model = unwrap_model(self.model)
        
        if self.args.save_adapter_only:
            # ============ 只保存适配器权重 ============
            
            # 提取 proj 层的权重
            if state_dict is not None:
                # 从提供的 state_dict 中提取（DeepSpeed ZeRO-3 场景）
                proj_state_dict = {
                    k.replace("proj.", ""): v.cpu()
                    for k, v in state_dict.items()
                    if k.startswith("proj.")
                }
            else:
                # 直接从模型获取
                proj_state_dict = {
                    k: v.cpu() for k, v in model.proj.state_dict().items()
                }
            
            # 保存投影层权重（与 MEMModel.save_pretrained 格式一致）
            torch.save(
                proj_state_dict, 
                os.path.join(output_dir, "projection_layer.pt")
            )
            
            # 保存配置（与 MEMModel.save_pretrained 格式一致）
            config_dict = {
                "model_type": model.config.model_type,
                "base_model_name": model.config.base_model_name,
                "ocr_model_name": model.config.ocr_model_name,
                "vision_embedding_size": model.config.vision_embedding_size,
                "context_threshold": model.config.context_threshold,
            }
            with open(os.path.join(output_dir, "config.json"), "w") as f:
                json.dump(config_dict, f, indent=2)
            
            print(f"✓ 已保存适配器到 {output_dir}")
            print(f"  - projection_layer.pt: 投影层权重")
            print(f"  - config.json: 模型配置")
            
        else:
            # ============ 保存完整模型 ============
            # 使用 MEMModel 的 save_pretrained 方法
            model.save_pretrained(
                output_dir,
                is_main_process=self.args.should_save,
                state_dict=state_dict,
            )
            print(f"✓ 已保存完整模型到 {output_dir}")
        
        # 保存 tokenizer（用于推理时加载）
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)
            print(f"  - tokenizer: 分词器配置")
    
    def _load_from_checkpoint(
        self, 
        resume_from_checkpoint: str,
        model: Optional[MEMModel] = None
    ) -> None:
        """
        从检查点恢复训练
        
        支持两种检查点格式：
        1. 适配器格式: 只有 projection_layer.pt
        2. 完整模型格式: 使用 MEMModel.from_pretrained
        
        Args:
            resume_from_checkpoint: 检查点目录路径
            model: 可选的模型实例
        """
        if model is None:
            model = unwrap_model(self.model)
        
        # 检查是否为适配器格式
        proj_path = os.path.join(resume_from_checkpoint, "projection_layer.pt")
        
        if os.path.exists(proj_path):
            # 加载适配器权重
            proj_state_dict = torch.load(proj_path, map_location="cpu", weights_only=True)
            model.proj.load_state_dict(proj_state_dict)
            print(f"✓ 从 {proj_path} 加载了适配器权重")
        else:
            # 尝试加载完整检查点
            super()._load_from_checkpoint(resume_from_checkpoint, model)



# ============================================================================
# DeepSpeed 配置辅助函数
# ============================================================================

def get_world_size() -> int:
    """
    获取分布式训练的 world size（进程总数）
    
    Returns:
        int: 进程数，非分布式环境返回 1
    
    说明：
        - 分布式环境：返回所有参与训练的 GPU/进程数
        - 单机环境：返回 1
    """
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1


def setup_deepspeed_config(args: AdapterTrainingArguments) -> str:
    """
    创建 DeepSpeed ZeRO-2 配置文件
    
    DeepSpeed 优化策略：
    - ZeRO Stage 2: 优化器状态分片 + 梯度分片
    - CPU Offloading: 将优化器状态移至 CPU，节省 GPU 显存
    - BF16 混合精度: 使用 bfloat16 以提高训练效率
    
    配置参数说明：
    - train_batch_size: 全局 batch size = per_gpu * accumulation * world_size
    - train_micro_batch_size_per_gpu: 每个 GPU 每次前向的 batch size
    - offload_optimizer: 将 Adam 优化器状态移到 CPU
    
    Args:
        args: 训练参数配置
    
    Returns:
        str: DeepSpeed 配置文件的路径
    
    注意：
        该配置专为适配器训练优化，适合显存受限的环境
    """
    # 获取分布式训练的 world size
    world_size = get_world_size()
    
    # 计算批次大小
    # 有效 batch size = per_device_batch * gradient_accumulation * world_size
    train_batch_size = (
        args.per_device_train_batch_size * 
        args.gradient_accumulation_steps * 
        world_size
    )
    micro_batch_size = args.per_device_train_batch_size
    
    # ============ DeepSpeed 配置 ============
    config = {
        # -------- 批次配置（必须正确设置） --------
        "train_batch_size": train_batch_size,
        "train_micro_batch_size_per_gpu": micro_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        
        # -------- 精度配置 --------
        # 使用 bfloat16 混合精度训练
        # bf16 相比 fp16 有更大的动态范围，更稳定
        "bf16": {
            "enabled": True
        },
        "fp16": {
            "enabled": False
        },
        
        # -------- ZeRO Stage 2 优化 --------
        # ZeRO-2: 分片优化器状态和梯度
        # 显存节省约 50-60%，适合大模型微调
        "zero_optimization": {
            "stage": 2,  # Stage 2: 优化器状态 + 梯度分片
            
            # 将优化器状态移到 CPU（主要显存节省来源）
            # Adam 有两个 momentum buffer，占用大量显存
            "offload_optimizer": {
                "device": "cpu",
                "pin_memory": True  # 使用 pinned memory 加速 CPU-GPU 传输
            },
            
            # 通信优化参数
            "allgather_partitions": True,       # 使用 allgather 收集梯度
            "allgather_bucket_size": 5e8,       # allgather bucket 大小
            "reduce_scatter": True,              # 使用 reduce_scatter
            "reduce_bucket_size": 5e8,          # reduce bucket 大小
            "overlap_comm": True,               # 通信和计算重叠
            "contiguous_gradients": True,       # 使用连续内存存储梯度
        },
        
        # -------- 梯度裁剪 --------
        "gradient_clipping": args.max_grad_norm,
        
        # -------- 日志配置 --------
        "steps_per_print": args.logging_steps,
        "wall_clock_breakdown": False,  # 关闭性能分析以提高速度
    }
    
    # 保存配置文件
    config_path = os.path.join(args.output_dir, "ds_config.json")
    os.makedirs(args.output_dir, exist_ok=True)
    
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    
    # 打印配置信息
    print(f"\n{'='*60}")
    print("DeepSpeed 配置")
    print(f"{'='*60}")
    print(f"✓ 配置文件: {config_path}")
    print(f"✓ ZeRO Stage: 2 (优化器状态 + 梯度分片)")
    print(f"✓ CPU Offloading: 优化器状态")
    print(f"✓ 精度: BFloat16")
    print(f"✓ 全局 Batch Size: {train_batch_size}")
    print(f"✓ 每 GPU Micro Batch: {micro_batch_size}")
    print(f"✓ 梯度累积步数: {args.gradient_accumulation_steps}")
    print(f"✓ World Size: {world_size}")
    print(f"{'='*60}\n")
    
    return config_path


# ============================================================================
# 主训练函数
# ============================================================================

def train():
    """
    主训练入口函数
    
    执行流程：
    1. 配置训练参数
    2. 创建 DeepSpeed 配置
    3. 加载模型
    4. 加载数据集
    5. 初始化训练器
    6. 执行训练
    7. 保存最终适配器
    
    运行方式：
        单机单卡：
            python adapter_only_trainer.py
        
        单机多卡（DeepSpeed）：
            deepspeed --num_gpus=4 adapter_only_trainer.py
        
        多机多卡：
            deepspeed --hostfile=hostfile adapter_only_trainer.py
    """
    
    # ==================== Step 1: 配置训练参数 ====================
    print(f"\n{'='*60}")
    print("步骤 1/6: 配置训练参数")
    print(f"{'='*60}\n")
    
    args = AdapterTrainingArguments(
        # ---------- 输出配置 ----------
        output_dir="./output/adapter_checkpoints",  # 检查点保存目录
        run_name="mem_adapter",                      # 实验名称（用于日志）
        
        # ---------- 模型配置 ----------
        base_model_name="Qwen/Qwen2.5-1.5B-Instruct",  # 基础语言模型
        ocr_model_name="deepseek-ai/DeepSeek-OCR",     # OCR 视觉编码器
        vision_embedding_size=512,                      # 视觉嵌入维度
        context_threshold=2048,                         # 视觉压缩阈值
        
        # ---------- 数据配置 ----------
        train_data_path="data/train.jsonl",    # 训练数据路径
        eval_data_path="data/eval.jsonl",      # 验证数据路径
        max_seq_length=2048,                   # 最大序列长度
        
        # ---------- 训练超参数 ----------
        num_train_epochs=3,                    # 训练轮数
        per_device_train_batch_size=2,         # 每 GPU 训练 batch size
        per_device_eval_batch_size=4,          # 每 GPU 验证 batch size
        gradient_accumulation_steps=8,         # 梯度累积步数
        
        # ---------- 优化器配置 ----------
        learning_rate=2e-4,                    # 学习率
        warmup_ratio=0.1,                      # warmup 比例（前 10% 步数）
        weight_decay=0.01,                     # 权重衰减
        max_grad_norm=1.0,                     # 梯度裁剪阈值
        
        # ---------- 精度配置 ----------
        bf16=True,                             # 使用 bfloat16
        tf32=True,                             # 启用 TF32（A100+ GPU）
        
        # ---------- 评估与保存 ----------
        eval_strategy="steps",                 # 按步数评估
        eval_steps=200,                        # 每 200 步评估
        save_strategy="steps",                 # 按步数保存
        save_steps=200,                        # 每 200 步保存
        save_total_limit=3,                    # 最多保留 3 个检查点
        logging_steps=10,                      # 每 10 步记录日志
        
        # ---------- DeepSpeed ----------
        deepspeed=None,  # 将在下面设置
        
        # ---------- 其他配置 ----------
        dataloader_num_workers=4,              # 数据加载进程数
        remove_unused_columns=False,           # 保留所有列（自定义数据格式需要）
        report_to=["tensorboard"],             # 日志报告目标
        seed=42,                               # 随机种子
        
        # ---------- 适配器配置 ----------
        save_adapter_only=True,                # 只保存适配器权重
    )
    
    print("✓ 训练参数配置完成")
    
    # ==================== Step 2: 配置 DeepSpeed ====================
    print(f"\n{'='*60}")
    print("步骤 2/6: 配置 DeepSpeed")
    print(f"{'='*60}")
    
    # 创建 DeepSpeed 配置（在初始化 Trainer 之前）
    args.deepspeed = setup_deepspeed_config(args)
    
    # ==================== Step 3: 加载模型 ====================
    print(f"{'='*60}")
    print("步骤 3/6: 加载模型")
    print(f"{'='*60}\n")
    
    # 创建模型配置
    model_config = MEMConfig(
        base_model_name=args.base_model_name,
        ocr_model_name=args.ocr_model_name,
        vision_embedding_size=args.vision_embedding_size,
        context_threshold=args.context_threshold,
    )
    
    # 初始化模型
    model = MEMModel(config=model_config)
    tokenizer = model.tokenizer
    
    # 确保 tokenizer 有 pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    print(f"✓ 基础模型: {args.base_model_name}")
    print(f"✓ OCR 模型: {args.ocr_model_name}")
    print(f"✓ 视觉嵌入维度: {args.vision_embedding_size}")
    print(f"✓ 上下文阈值: {args.context_threshold}")
    
    # 打印可训练参数（模型内置方法）
    model.print_trainable_parameters()
    
    # ==================== Step 4: 加载数据集 ====================
    print(f"\n{'='*60}")
    print("步骤 4/6: 加载数据集")
    print(f"{'='*60}\n")
    
    # 加载训练数据
    train_dataset = MultiTurnConversationDataset(
        jsonl_path=args.train_data_path,
        tokenizer=tokenizer,
        max_length=args.max_seq_length,
        return_dict=False  # 不返回 messages 字典，只返回张量
    )
    print(f"✓ 训练集: {len(train_dataset)} 样本")
    
    # 加载验证数据（可选）
    eval_dataset = None
    if args.eval_data_path and os.path.exists(args.eval_data_path):
        eval_dataset = MultiTurnConversationDataset(
            jsonl_path=args.eval_data_path,
            tokenizer=tokenizer,
            max_length=args.max_seq_length,
            return_dict=False
        )
        print(f"✓ 验证集: {len(eval_dataset)} 样本")
    else:
        print(f"⚠ 未找到验证数据: {args.eval_data_path}")
    
    # ==================== Step 5: 初始化训练器 ====================
    print(f"\n{'='*60}")
    print("步骤 5/6: 初始化训练器")
    print(f"{'='*60}\n")
    
    trainer = AdapterOnlyTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=partial(collate_fn, tokenizer=tokenizer),
    )
    
    print("✓ AdapterOnlyTrainer 初始化完成")
    
    # ==================== Step 6: 开始训练 ====================
    print(f"\n{'='*60}")
    print("步骤 6/6: 开始训练")
    print(f"{'='*60}\n")
    
    # 检查是否有可恢复的检查点
    checkpoint = None
    if os.path.exists(args.output_dir):
        checkpoints = [
            d for d in os.listdir(args.output_dir) 
            if d.startswith("checkpoint-")
        ]
        if checkpoints:
            # 找到最新的检查点
            latest = max(checkpoints, key=lambda x: int(x.split("-")[-1]))
            checkpoint = os.path.join(args.output_dir, latest)
            print(f"✓ 发现检查点，将从 {checkpoint} 恢复训练\n")
    
    # 开始训练
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    
    # ==================== 保存最终模型 ====================
    final_dir = os.path.join(args.output_dir, "final_adapter")
    trainer.save_model(final_dir)
    
    # 保存训练指标
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    
    # 完成提示
    print(f"\n{'='*60}")
    print("🎉 训练完成！")
    print(f"{'='*60}")
    print(f"✓ 最终适配器保存至: {final_dir}")
    print(f"✓ 训练损失: {metrics.get('train_loss', 'N/A'):.4f}")
    print(f"✓ 总训练步数: {metrics.get('train_steps', 'N/A')}")
    print(f"{'='*60}\n")


# ============================================================================
# 程序入口
# ============================================================================

if __name__ == "__main__":
    train()