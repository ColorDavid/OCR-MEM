#!/usr/bin/env python3
"""
测试梯度流的简单脚本
验证从 OCR encoder -> proj -> LLM 的梯度传播
"""

import torch
import torch.nn as nn

def test_gradient_flow():
    print("="*60)
    print("测试梯度流")
    print("="*60)
    
    # 模拟场景
    batch_size = 2
    seq_len = 10
    embed_dim = 1024
    hidden_size = 4096
    
    # 1. 模拟冻结的 OCR encoder 输出（有梯度流过，但参数冻结）
    print("\n1. 创建模拟 OCR encoder")
    ocr_output = torch.randn(batch_size, seq_len, embed_dim, requires_grad=True)
    print(f"   ocr_output.requires_grad: {ocr_output.requires_grad}")
    
    # 2. 可训练的 proj 层
    print("\n2. 创建可训练的 proj 层")
    proj = nn.Linear(embed_dim, hidden_size)
    for param in proj.parameters():
        param.requires_grad = True
    print(f"   proj.weight.requires_grad: {proj.weight.requires_grad}")
    
    # 3. 前向传播
    print("\n3. 前向传播: ocr_output -> proj")
    proj_output = proj(ocr_output)
    print(f"   proj_output.requires_grad: {proj_output.requires_grad}")
    print(f"   proj_output.grad_fn: {proj_output.grad_fn}")
    
    # 4. 模拟与冻结的 embedding 拼接
    print("\n4. 测试拼接: proj_output + frozen_embeds")
    frozen_embeds = torch.randn(batch_size, 5, hidden_size)  # 无梯度
    print(f"   frozen_embeds.requires_grad: {frozen_embeds.requires_grad}")
    
    # 拼接
    combined = torch.cat([proj_output, frozen_embeds], dim=1)
    print(f"   combined.requires_grad: {combined.requires_grad}")
    print(f"   combined.grad_fn: {combined.grad_fn}")
    
    # 5. 模拟 padding
    print("\n5. 测试 padding")
    pad_embeds = torch.zeros(batch_size, 3, hidden_size)
    print(f"   pad_embeds.requires_grad: {pad_embeds.requires_grad}")
    
    padded = torch.cat([pad_embeds, combined], dim=1)
    print(f"   padded.requires_grad: {padded.requires_grad}")
    print(f"   padded.grad_fn: {padded.grad_fn}")
    
    # 6. 模拟 loss 和 backward
    print("\n6. 测试 backward")
    loss = padded.sum()
    print(f"   loss.requires_grad: {loss.requires_grad}")
    print(f"   loss.grad_fn: {loss.grad_fn}")
    
    print("\n   执行 backward...")
    loss.backward()
    
    print(f"   proj.weight.grad is not None: {proj.weight.grad is not None}")
    if proj.weight.grad is not None:
        print(f"   proj.weight.grad.norm(): {proj.weight.grad.norm().item():.4f}")
    
    # 7. 测试 torch.stack
    print("\n7. 测试 torch.stack")
    tensor_list = [
        torch.randn(10, hidden_size, requires_grad=True),
        torch.randn(10, hidden_size, requires_grad=True)
    ]
    print(f"   tensor_list[0].requires_grad: {tensor_list[0].requires_grad}")
    
    stacked = torch.stack(tensor_list, dim=0)
    print(f"   stacked.requires_grad: {stacked.requires_grad}")
    print(f"   stacked.grad_fn: {stacked.grad_fn}")
    
    print("\n" + "="*60)
    print("✓ 梯度流测试通过!")
    print("="*60)

if __name__ == "__main__":
    test_gradient_flow()
