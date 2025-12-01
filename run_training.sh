#!/bin/bash
# ==============================================================================
# MEMModel Adapter 训练启动脚本
# ==============================================================================
#
# 本脚本用于多卡分布式训练，使用 DeepSpeed 或 torchrun 启动。
#
# 【使用方法】
#     1. 赋予执行权限：chmod +x run_training.sh
#     2. 运行脚本：./run_training.sh
#
# 【运行位置】
#     必须在 OCR-MEM 目录下运行
#
# 【环境要求】
#     - Python 3.8+
#     - PyTorch 2.0+
#     - transformers 4.35+
#     - deepspeed 0.12+
#     - wandb
#
# ==============================================================================

# 设置工作目录为脚本所在目录
cd "$(dirname "$0")"

echo "============================================================"
echo "MEMModel Adapter Training Script"
echo "============================================================"
echo "工作目录: $(pwd)"
echo "Python: $(which python)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA 可用: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU 数量: $(python -c 'import torch; print(torch.cuda.device_count())')"
echo "============================================================"


# ==============================================================================
# 配置区域 - 请根据您的环境修改
# ==============================================================================

# -------------------- WandB 配置 --------------------
# WandB API Key（用于日志记录和可视化）
export WANDB_API_KEY="c8769e5bbc8fd36df4155b757331cd139e0fc327"

# WandB 项目名称
export WANDB_PROJECT="OCR-MEM"

# WandB 实验名称（可选，留空则自动生成）
export WANDB_RUN_NAME="mem_adapter_training"

# WandB 模式：online（实时同步）、offline（离线保存）、disabled（禁用）
export WANDB_MODE="online"

# -------------------- 模型路径（本地路径）--------------------
# 基础语言模型路径
export BASE_MODEL_PATH="/mmu_nlp_ssd/tangjingyi03/OCR-MEM/model/Qwen/Qwen3-8B"

# OCR 编码器路径
export OCR_MODEL_PATH="/mmu_nlp_ssd/tangjingyi03/models/deepseek-ai/DeepSeek-OCR"

# -------------------- 数据路径 --------------------
# 训练数据
export TRAIN_DATA="/mmu_nlp_ssd/tangjingyi03/OCR-MEM/data/train_data_of_merged_total_66025_rmtestset_first_100.jsonl"

# 验证数据（可选）
export EVAL_DATA="/mmu_nlp_ssd/tangjingyi03/OCR-MEM/data/train_data_of_merged_total_66025_rmtestset_first_100.jsonl"

# -------------------- 输出目录 --------------------
export OUTPUT_DIR="./adapter_checkpoints"

# -------------------- 训练超参数 --------------------
export NUM_EPOCHS=2
export BATCH_SIZE=2
export GRADIENT_ACCUMULATION=8
export LEARNING_RATE=2e-4

# -------------------- GPU 配置 --------------------
# 指定使用的 GPU（例如 "0,1,2,3"）
export CUDA_VISIBLE_DEVICES="0,1,2,3"

# GPU 数量（与 CUDA_VISIBLE_DEVICES 中的数量一致）
export NUM_GPUS=4

# 分布式训练端口（如有冲突请修改）
export MASTER_PORT=29500


# ==============================================================================
# 选择训练模式
# ==============================================================================
# 可选值：
#   deepspeed - DeepSpeed 多卡训练（推荐）
#   torchrun  - PyTorch 原生分布式训练

TRAINING_MODE="deepspeed"


# ==============================================================================
# 训练启动
# ==============================================================================

case $TRAINING_MODE in

    # --------------------------------------------------------------------------
    # DeepSpeed 多卡训练（推荐）
    # --------------------------------------------------------------------------
    "deepspeed")
        echo ""
        echo "启动 DeepSpeed 多卡训练..."
        echo "GPU: $CUDA_VISIBLE_DEVICES"
        echo "GPU 数量: $NUM_GPUS"
        echo "WandB 项目: $WANDB_PROJECT"
        echo ""
        
        # DeepSpeed 启动命令
        # --num_gpus: GPU 数量
        # --master_port: 主节点端口（如有冲突请修改）
        deepspeed --num_gpus=$NUM_GPUS \
            --master_port=$MASTER_PORT \
            run_training.py \
            --base_model_path "$BASE_MODEL_PATH" \
            --ocr_model_path "$OCR_MODEL_PATH" \
            --train_data "$TRAIN_DATA" \
            --eval_data "$EVAL_DATA" \
            --output_dir "$OUTPUT_DIR" \
            --num_epochs $NUM_EPOCHS \
            --batch_size $BATCH_SIZE \
            --gradient_accumulation $GRADIENT_ACCUMULATION \
            --learning_rate $LEARNING_RATE \
            --wandb_project "$WANDB_PROJECT" \
            --wandb_run_name "$WANDB_RUN_NAME"
        ;;

    # --------------------------------------------------------------------------
    # PyTorch 原生分布式训练
    # --------------------------------------------------------------------------
    "torchrun")
        echo ""
        echo "启动 torchrun 分布式训练..."
        echo "GPU: $CUDA_VISIBLE_DEVICES"
        echo "GPU 数量: $NUM_GPUS"
        echo "WandB 项目: $WANDB_PROJECT"
        echo ""
        
        # torchrun 启动命令
        # --nproc_per_node: 每个节点的进程数（GPU 数量）
        # --master_port: 主节点端口
        torchrun --nproc_per_node=$NUM_GPUS \
            --master_port=$MASTER_PORT \
            run_training.py \
            --base_model_path "$BASE_MODEL_PATH" \
            --ocr_model_path "$OCR_MODEL_PATH" \
            --train_data "$TRAIN_DATA" \
            --eval_data "$EVAL_DATA" \
            --output_dir "$OUTPUT_DIR" \
            --num_epochs $NUM_EPOCHS \
            --batch_size $BATCH_SIZE \
            --gradient_accumulation $GRADIENT_ACCUMULATION \
            --learning_rate $LEARNING_RATE \
            --wandb_project "$WANDB_PROJECT" \
            --wandb_run_name "$WANDB_RUN_NAME"
        ;;

    *)
        echo "错误: 未知的训练模式 '$TRAINING_MODE'"
        echo "可选值: deepspeed, torchrun"
        exit 1
        ;;
esac


# ==============================================================================
# 训练完成
# ==============================================================================

if [ $? -eq 0 ]; then
    echo ""
    echo "============================================================"
    echo "🎉 训练完成！"
    echo "============================================================"
    echo "检查点保存在: $OUTPUT_DIR"
    echo "最终适配器: $OUTPUT_DIR/final_adapter"
    echo "============================================================"
else
    echo ""
    echo "============================================================"
    echo "❌ 训练失败，请检查错误日志"
    echo "============================================================"
    exit 1
fi
