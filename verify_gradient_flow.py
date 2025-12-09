#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速验证梯度流动脚本
用于确认修复后的模型能够正确传播梯度
"""

import sys
import torch
from pathlib import Path

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from mem_adapter_only import MEMModel, MEMConfig

def verify_gradient_flow():
    """验证梯度流动是否正常"""
    
    print("=" * 70)
    print("梯度流动验证脚本")
    print("=" * 70)
    
    # 配置参数 - 请根据实际情况修改路径
    base_model_path = input("请输入基础模型路径 (或按Enter使用默认): ").strip()
    ocr_model_path = input("请输入OCR模型路径 (或按Enter使用默认): ").strip()
    
    if not base_model_path:
        base_model_path = "/mmu_nlp_ssd/tangjingyi03/OCR-MEM/model/Qwen/Qwen3-8B"
    if not ocr_model_path:
        ocr_model_path = "/mmu_nlp_ssd/tangjingyi03/models/deepseek-ai/DeepSeek-OCR"
    
    print(f"\n使用模型路径:")
    print(f"  Base Model: {base_model_path}")
    print(f"  OCR Model:  {ocr_model_path}")
    
    try:
        # 1. 创建配置
        print("\n[1/6] 创建模型配置...")
        config = MEMConfig(
            base_model_name=base_model_path,
            ocr_model_name=ocr_model_path,
            vision_embedding_size=1024,
            context_threshold=2048
        )
        print("✅ 配置创建成功")
        
        # 2. 加载模型
        print("\n[2/6] 加载模型（这可能需要几分钟）...")
        model = MEMModel(config)
        model.train()  # 确保在训练模式
        print("✅ 模型加载成功")
        
        # 3. 打印参数统计
        print("\n[3/6] 验证参数统计...")
        model.print_trainable_parameters()
        
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if trainable_params == 0:
            print("❌ 错误：没有可训练参数！")
            return False
        print("✅ 参数统计正确")
        
        # 4. 创建虚拟输入
        print("\n[4/6] 创建虚拟输入...")
        batch_size = 1
        seq_len = 100
        vocab_size = model.tokenizer.vocab_size
        
        dummy_input = torch.randint(0, vocab_size, (batch_size, seq_len))
        dummy_attention_mask = torch.ones_like(dummy_input)
        dummy_labels = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        # 将一些位置设为-100（不计算loss）
        dummy_labels[:, :50] = -100
        print("✅ 虚拟输入创建成功")
        
        # 5. 前向传播
        print("\n[5/6] 执行前向传播...")
        outputs = model(
            input_ids=dummy_input,
            attention_mask=dummy_attention_mask,
            labels=dummy_labels
        )
        
        # 检查loss
        if outputs.loss is None:
            print("❌ 错误：loss为None！")
            return False
        
        if not outputs.loss.requires_grad:
            print("❌ 错误：loss没有requires_grad！")
            return False
        
        if outputs.loss.grad_fn is None:
            print("❌ 错误：loss没有grad_fn！")
            print("   这表明梯度计算图被断开了。")
            return False
        
        print(f"✅ 前向传播成功")
        print(f"   Loss值: {outputs.loss.item():.4f}")
        print(f"   Loss requires_grad: {outputs.loss.requires_grad}")
        print(f"   Loss grad_fn: {outputs.loss.grad_fn}")
        
        # 6. 反向传播
        print("\n[6/6] 执行反向传播...")
        outputs.loss.backward()
        print("✅ 反向传播成功")
        
        # 7. 验证梯度
        print("\n验证梯度分布...")
        proj_has_grad = False
        base_has_grad = False
        ocr_has_grad = False
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                if param.grad is None:
                    print(f"❌ 警告：{name} 需要梯度但grad为None")
                else:
                    if 'proj' in name:
                        proj_has_grad = True
                        print(f"✅ proj参数有梯度: {name}")
                        print(f"   梯度统计: mean={param.grad.mean().item():.6f}, "
                              f"std={param.grad.std().item():.6f}, "
                              f"max={param.grad.abs().max().item():.6f}")
            else:
                # 冻结参数不应该有梯度
                if param.grad is not None:
                    if 'base_llm_model' in name:
                        base_has_grad = True
                        print(f"❌ 错误：base_llm参数不应该有梯度: {name}")
                    elif 'ocr_embed' in name:
                        ocr_has_grad = True
                        print(f"❌ 错误：ocr_embed参数不应该有梯度: {name}")
        
        # 最终验证
        print("\n" + "=" * 70)
        print("验证结果汇总")
        print("=" * 70)
        
        all_passed = True
        
        if not proj_has_grad:
            print("❌ 失败：proj层没有接收到梯度")
            all_passed = False
        else:
            print("✅ 通过：proj层正确接收梯度")
        
        if base_has_grad:
            print("❌ 失败：base_llm模型有梯度（应该被冻结）")
            all_passed = False
        else:
            print("✅ 通过：base_llm模型正确冻结")
        
        if ocr_has_grad:
            print("❌ 失败：ocr_embed有梯度（应该被冻结）")
            all_passed = False
        else:
            print("✅ 通过：ocr_embed正确冻结")
        
        print("=" * 70)
        
        if all_passed:
            print("🎉 所有验证通过！梯度流动正常，可以开始训练。")
            return True
        else:
            print("⚠️  部分验证失败，请检查上述错误。")
            return False
        
    except Exception as e:
        print(f"\n❌ 验证过程中出现错误：")
        print(f"   {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = verify_gradient_flow()
    sys.exit(0 if success else 1)
