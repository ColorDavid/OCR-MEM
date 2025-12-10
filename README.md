# OCR-MEM 训练项目

使用 DeepSeek-OCR 进行视觉记忆压缩的模型训练。

## 项目结构

```
OCR-MEM/
├── mem_adapter_only.py      # 核心模型实现（已修复梯度流问题）
├── run_training.py           # 训练脚本
├── run_training.sh           # 训练启动脚本
├── ds_config.json            # DeepSpeed 配置
├── trainer/                  # 自定义训练器
├── utils/                    # 工具函数
└── data/                     # 训练数据
```

## 快速开始

```bash
# 启动训练
bash run_training.sh
```

## 重要修复说明

**问题**：训练时出现 `RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn`

**根本原因**：`deepseek-ocr-encoder` 库的**所有前向传播方法**都被 `@torch.inference_mode()` 装饰，包括:
- `encode()` 方法 (第 416 行)
- `_forward_core()` 方法 (第 394 行) ← **这是关键发现!**

这导致即使我们绕过 `encode()` 直接调用 `_forward_core()`,梯度仍然会断裂。

**最终解决方案**：在 `mem_adapter_only.py` 中完全绕过所有装饰的方法，直接手动调用底层组件：
- `self.ocr_embed.sam(x)` - SAM encoder
- `self.ocr_embed.clip_pre(tokens)` - CLIP pre-layernorm  
- `self.ocr_embed.clip_tr(x_tok)` - CLIP transformer

这样可以保持梯度流动,因为这些底层模块没有被 `@torch.inference_mode()` 装饰。

**详细技术分析**：请查看 `CRITICAL_FINDING.md`

**解决方案**：在 `mem_adapter_only.py` 中绕过 `encode()` 方法，直接调用内部的 `_forward_core()` 方法，确保梯度正常流动。

详细技术分析见：[REAL_GRADIENT_ISSUE_ANALYSIS.md](./REAL_GRADIENT_ISSUE_ANALYSIS.md)

## 训练配置

- 基础模型：Qwen3-8B
- OCR 编码器：DeepSeek-OCR
- 训练方式：Adapter-only（只训练投影层）
- 可训练参数：18,882,560 (0.22%)
- GPU：8卡训练
- 优化器：DeepSpeed ZeRO-2

## 依赖要求

```bash
pip install torch transformers deepspeed wandb
pip install deepseek-ocr-encoder
pip install pdf2image pymupdf pillow reportlab
```

**注意**：`transformers` 版本需要 `>=4.30.0,<4.48.0`
