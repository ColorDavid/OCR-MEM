import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    PreTrainedTokenizerBase,
    GenerationConfig
)
from transformers.modeling_outputs import CausalLMOutputWithPast
from peft import PeftConfig, LoraConfig

import random
from typing import Tuple, Optional, List, Union
import logging
from utils.render import text_to_images
from deepseek_ocr_encoder import DeepSeekOCREncoder

# # One-line initialization - automatically handles device, dtype, and model loading
# encoder = DeepSeekOCREncoder.from_pretrained("deepseek-ai/DeepSeek-OCR")

# # Encode an image
# vision_tokens = encoder("your_image.png")

@registry.register_model("memmodel")
class MEMModel(nn.Module):

    def __init__(
        self, 
        base_model_name: str, 
        ocr_model_name: str,
        vision_embedding_size: str,
        inference_latents_len: int,
        adapter_peft_config: Optional[PeftConfig] = None,
    ):   
        super().__init__()
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"
        )
        self.tokenizer =  AutoTokenizer.from_pretrained(base_model_name)
        self.base_model_config = self.base_model.config

        self.render_func = text_to_images
        self.ocr_embed = DeepSeekOCREncoder.from_pretrained(ocr_model_name)#"deepseek-ai/DeepSeek-OCR"
        mlp_hidden_dim = max(vision_embedding_size, self.base_model_config.hidden_size)
        
        self.proj = nn.Sequential(
            nn.Linear(vision_embedding_size, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, self.base_model_config.hidden_size)
        )

    @classmethod
    def from_config(cls, config):
        pass

    def _forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        context_threshold: int = 2048,
        past_context: Optional[Union[str, List[str]]] = None,
        **kwargs
    ) -> CausalLMOutputWithPast:
        """
        Forward pass with visual memory compression for long contexts.
        
        Args:
            input_ids: Current input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            labels: Target labels for loss computation [batch_size, seq_len]
            context_threshold: Maximum context length before visual compression
            past_context: Historical context text(s) to be compressed
        
        Returns:
            CausalLMOutputWithPast containing loss, logits, and other outputs
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        dtype = self.base_model.dtype
        
        # Get current input embeddings
        current_embeds = self.base_model.get_input_embeddings()(input_ids)
        current_attention_mask = attention_mask
        
        # Check if we need to apply visual memory compression
        use_visual_memory = False
        if past_context is not None:
            # Handle both single string and batch of strings
            if isinstance(past_context, str):
                past_context = [past_context] * batch_size
            
            # Check if any context exceeds threshold
            for ctx in past_context:
                past_tokens = self.tokenizer(
                    ctx,
                    add_special_tokens=False,
                    return_tensors="pt"
                )["input_ids"]
                
                if past_tokens.size(1) > context_threshold:
                    use_visual_memory = True
                    break
        
        if use_visual_memory and past_context is not None:
            # Process each example in the batch
            vision_embeds_batch = []
            
            for ctx in past_context:
                # Render text context as image(s)
                images = self.render_func(ctx)
                
                # Handle single image or list of images
                if not isinstance(images, list):
                    images = [images]
                
                # Encode images using OCR encoder and project
                vision_tokens_list = []
                for img in images:
                    # OCR encoding: [num_tokens, vision_embedding_size]
                    vision_tokens = self.ocr_embed(img)
                    
                    # Project to base model hidden size
                    projected = self.proj(vision_tokens)  # [num_tokens, hidden_size]
                    vision_tokens_list.append(projected)
                
                # Concatenate all vision tokens for this example
                vision_embeds = torch.cat(vision_tokens_list, dim=0)  # [total_tokens, hidden_size]
                vision_embeds_batch.append(vision_embeds)
            
            # Pad vision embeddings to same length within batch
            max_vision_len = max(ve.size(0) for ve in vision_embeds_batch)
            padded_vision_embeds = []
            vision_attention_masks = []
            
            for ve in vision_embeds_batch:
                num_tokens = ve.size(0)
                if num_tokens < max_vision_len:
                    # Left padding
                    padding = torch.zeros(
                        max_vision_len - num_tokens,
                        ve.size(1),
                        dtype=dtype,
                        device=device
                    )
                    ve_padded = torch.cat([padding, ve], dim=0)
                    attn_mask = torch.cat([
                        torch.zeros(max_vision_len - num_tokens, device=device),
                        torch.ones(num_tokens, device=device)
                    ], dim=0)
                else:
                    ve_padded = ve
                    attn_mask = torch.ones(max_vision_len, device=device)
                
                padded_vision_embeds.append(ve_padded)
                vision_attention_masks.append(attn_mask)
            
            # Stack into batch tensors
            vision_embeds = torch.stack(padded_vision_embeds, dim=0)  # [batch, max_vision_len, hidden]
            vision_attn_mask = torch.stack(vision_attention_masks, dim=0)  # [batch, max_vision_len]
            
            # Concatenate vision embeddings with current input
            inputs_embeds = torch.cat([vision_embeds, current_embeds], dim=1)
            attention_mask = torch.cat([vision_attn_mask, current_attention_mask], dim=1)
            
            # Update labels if provided (vision tokens don't have labels)
            if labels is not None:
                vision_labels = torch.full(
                    (batch_size, vision_embeds.size(1)),
                    -100,
                    dtype=labels.dtype,
                    device=device
                )
                labels = torch.cat([vision_labels, labels], dim=1)
        else:
            inputs_embeds = current_embeds
        
        # Forward through base model
        outputs = self.base_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
            **kwargs
        )
        
        return outputs








@registry.register_model("memmodel")
class MEMModel(nn.Module):
    """
    带有视觉记忆压缩的模型
    
    在每个user消息位置检查历史长度，如果超过阈值则进行视觉压缩增强。
    """

    def __init__(
        self, 
        base_model_name: str, 
        ocr_model_name: str,
        vision_embedding_size: int,
        context_threshold: int = 2048,
        use_adapter: bool = False,
        **kwargs
    ):   
        super().__init__()
        
        # 基础语言模型
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name, 
            torch_dtype=torch.bfloat16, 
            attn_implementation="flash_attention_2"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        self.base_model_config = self.base_model.config
        
        # 全局配置参数
        self.context_threshold = context_threshold
        self.use_adapter = use_adapter

        # 文本到图像渲染函数
        self.render_func = text_to_images
        
        # OCR视觉编码器
        self.ocr_embed = DeepSeekOCREncoder.from_pretrained(ocr_model_name)
        
        # 投影层：将视觉特征投影到语言模型的隐藏空间
        mlp_hidden_dim = max(vision_embedding_size, self.base_model_config.hidden_size)
        self.proj = nn.Sequential(
            nn.Linear(vision_embedding_size, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, self.base_model_config.hidden_size)
        )

        # 可选的适配器层
        if self.use_adapter:
            self.adapter = nn.Linear(
                self.base_model_config.hidden_size,
                self.base_model_config.hidden_size
            )
        
        # 编码user role起始序列：<|im_start|>user\n
        self.user_role_start = self.tokenizer.encode(
            "<|im_start|>user\n", 
            add_special_tokens=False
        )
        self.user_role_start_tensor = None  # 延迟初始化
    
    @classmethod
    def from_config(cls, config: dict):
        """从配置字典创建模型实例"""
        return cls(
            base_model_name=config.get("base_model_name"),
            ocr_model_name=config.get("ocr_model_name"),
            vision_embedding_size=config.get("vision_embedding_size"),
            context_threshold=config.get("context_threshold", 2048),
            use_adapter=config.get("use_adapter", False)
        )
    
    def _find_all_user_positions(self, input_ids: torch.Tensor) -> List[List[int]]:
        """
        找到每个样本中所有user role token的起始位置
        
        Args:
            input_ids: [batch_size, seq_len]
            
        Returns:
            positions: List[List[int]] 每个样本的所有user位置列表
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # 初始化user_role_start_tensor
        if self.user_role_start_tensor is None or self.user_role_start_tensor.device != device:
            self.user_role_start_tensor = torch.tensor(
                self.user_role_start, 
                dtype=torch.long, 
                device=device
            )
        
        pattern = self.user_role_start_tensor
        pattern_len = len(pattern)
        
        batch_positions = []
        
        for b in range(batch_size):
            seq = input_ids[b]
            positions = []
            
            # 从前向后搜索，找到所有匹配位置
            for i in range(seq_len - pattern_len + 1):
                if torch.all(seq[i:i + pattern_len] == pattern):
                    positions.append(i)
            
            batch_positions.append(positions)
        
        return batch_positions

    def _compress_text_to_vision(
        self, 
        text: str
    ) -> torch.Tensor:
        """
        将文本压缩为视觉embedding
        
        Args:
            text: 要压缩的文本
            
        Returns:
            vision_embeds: [num_vision_tokens, hidden_size]
        """
        device = next(self.parameters()).device
        dtype = self.base_model.dtype
        
        # 如果文本为空，返回单个零向量
        if not text:
            return torch.zeros(1, self.base_model_config.hidden_size, dtype=dtype, device=device)
        
        # 将文本渲染为图像
        images = self.render_func(text)
        if not isinstance(images, list):
            images = [images]
        
        # 使用OCR编码器编码每张图像
        vision_tokens_list = []
        for img in images:
            # OCR编码: [num_tokens, vision_embedding_size]
            vision_tokens = self.ocr_embed(img)
            
            # 投影到基础模型的隐藏空间: [num_tokens, hidden_size]
            projected = self.proj(vision_tokens)
            
            # 如果使用适配器
            if self.use_adapter:
                projected = self.adapter(projected)
            
            vision_tokens_list.append(projected)
        
        # 拼接所有视觉token
        vision_embeds = torch.cat(vision_tokens_list, dim=0)  # [total_tokens, hidden_size]
        
        return vision_embeds

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> CausalLMOutputWithPast:
        """
        前向传播，参照MemGen的_forward实现迭代式context增强
        
        在每个user消息位置检查历史长度：
        - 如果超过阈值，将之前的累积context压缩为视觉embedding并插入
        - 逐段拼接，最终得到增强后的完整序列
        
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            labels: [batch_size, seq_len]，可选
            
        Returns:
            CausalLMOutputWithPast
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        dtype = self.base_model.dtype
        hidden_size = self.base_model_config.hidden_size
        
        # 获取原始输入的embedding
        inputs_embeds = self.base_model.get_input_embeddings()(input_ids)
        
        # 找到所有user位置
        batch_user_positions = self._find_all_user_positions(input_ids)
        
        # 初始化输出列表（每个样本单独处理）
        batch_outputs = []
        batch_final_attention_masks = []
        batch_final_labels = []
        
        # 逐样本处理
        for b in range(batch_size):
            user_positions = batch_user_positions[b]
            seq_embeds = inputs_embeds[b]  # [seq_len, hidden_size]
            seq_mask = attention_mask[b]   # [seq_len]
            seq_labels = labels[b] if labels is not None else None
            
            # 初始化累积结果（参照MemGen的模式）
            current_start_idx = 0
            current_embeds = torch.empty((0, hidden_size), device=device, dtype=dtype)
            current_mask = torch.empty((0,), device=device, dtype=seq_mask.dtype)
            current_labels_list = [] if seq_labels is not None else None
            
            # 用于记录是否插入了视觉token（用于labels处理）
            vision_token_positions = []
            
            # 迭代每个user位置
            for user_pos in user_positions:
                # 提取当前段（从上次的结束位置到这个user位置）
                segment_embeds = seq_embeds[current_start_idx:user_pos]
                segment_mask = seq_mask[current_start_idx:user_pos]
                
                # 拼接到累积的embedding
                current_embeds = torch.cat([current_embeds, segment_embeds], dim=0)
                current_mask = torch.cat([current_mask, segment_mask], dim=0)
                
                # 计算当前累积的有效token数量
                valid_token_count = current_mask.sum().item()
                
                # 判断是否需要进行视觉压缩
                if valid_token_count > self.context_threshold:
                    # 需要压缩：提取当前累积的context文本
                    # 获取有效的token ids
                    valid_indices = current_mask.bool()
                    
                    # 从原始input_ids中重建当前的context
                    # 这里需要追踪原始位置
                    context_token_ids = input_ids[b, :current_start_idx + (user_pos - current_start_idx)]
                    context_text = self.tokenizer.decode(context_token_ids, skip_special_tokens=False)
                    
                    # 压缩为视觉embedding
                    vision_embeds = self._compress_text_to_vision(context_text)  # [vision_len, hidden_size]
                    vision_len = vision_embeds.size(0)
                    
                    # 将视觉embedding插入到当前累积的embedding之后
                    current_embeds = torch.cat([current_embeds, vision_embeds], dim=0)
                    
                    # 更新attention mask（视觉token全部有效）
                    vision_mask = torch.ones(vision_len, dtype=current_mask.dtype, device=device)
                    current_mask = torch.cat([current_mask, vision_mask], dim=0)
                    
                    # 记录视觉token的位置（用于后续labels处理）
                    vision_start = current_embeds.size(0) - vision_len
                    vision_token_positions.append((vision_start, current_embeds.size(0)))
                
                # 更新起始索引
                current_start_idx = user_pos
            
            # 处理最后一段（最后一个user之后的部分）
            remaining_embeds = seq_embeds[current_start_idx:]
            remaining_mask = seq_mask[current_start_idx:]
            
            current_embeds = torch.cat([current_embeds, remaining_embeds], dim=0)
            current_mask = torch.cat([current_mask, remaining_mask], dim=0)
            
            # 处理labels
            if seq_labels is not None:
                # 初始化新的labels（与current_embeds长度一致）
                new_labels = torch.full(
                    (current_embeds.size(0),), 
                    -100, 
                    dtype=seq_labels.dtype, 
                    device=device
                )
                
                # 映射原始labels到新位置
                # 这需要追踪哪些位置是原始token，哪些是插入的视觉token
                
                # 简化处理：
                # 1. 视觉token的labels设为-100
                # 2. 原始token的labels保持原样
                
                # 重新构建labels（逐段处理）
                label_idx = 0
                embed_idx = 0
                current_start_idx = 0
                
                for user_pos in user_positions:
                    segment_len = user_pos - current_start_idx
                    
                    # 原始segment的labels
                    new_labels[embed_idx:embed_idx + segment_len] = seq_labels[label_idx:label_idx + segment_len]
                    embed_idx += segment_len
                    label_idx += segment_len
                    
                    # 检查这个位置是否插入了视觉token
                    for vis_start, vis_end in vision_token_positions:
                        if vis_start == embed_idx:
                            # 跳过视觉token（已经初始化为-100）
                            embed_idx = vis_end
                            break
                    
                    current_start_idx = user_pos
                
                # 处理剩余的labels
                remaining_len = seq_len - current_start_idx
                new_labels[embed_idx:embed_idx + remaining_len] = seq_labels[label_idx:label_idx + remaining_len]
                
                batch_final_labels.append(new_labels)
            
            batch_outputs.append(current_embeds)
            batch_final_attention_masks.append(current_mask)
        
        # Padding到相同长度（左padding）
        max_len = max(emb.size(0) for emb in batch_outputs)
        
        padded_embeds = []
        padded_masks = []
        padded_labels = [] if labels is not None else None
        
        for idx, (emb, mask) in enumerate(zip(batch_outputs, batch_final_attention_masks)):
            pad_len = max_len - emb.size(0)
            
            if pad_len > 0:
                # 左padding
                emb_padded = torch.cat([
                    torch.zeros(pad_len, hidden_size, dtype=dtype, device=device),
                    emb
                ], dim=0)
                mask_padded = torch.cat([
                    torch.zeros(pad_len, dtype=mask.dtype, device=device),
                    mask
                ], dim=0)
                
                if labels is not None:
                    lab = batch_final_labels[idx]
                    lab_padded = torch.cat([
                        torch.full((pad_len,), -100, dtype=lab.dtype, device=device),
                        lab
                    ], dim=0)
                    padded_labels.append(lab_padded)
            else:
                emb_padded = emb
                mask_padded = mask
                if labels is not None:
                    padded_labels.append(batch_final_labels[idx])
            
            padded_embeds.append(emb_padded)
            padded_masks.append(mask_padded)
        
        # 堆叠成batch tensor
        final_inputs_embeds = torch.stack(padded_embeds, dim=0)  # [batch_size, max_len, hidden_size]
        final_attention_mask = torch.stack(padded_masks, dim=0)  # [batch_size, max_len]
        final_labels = torch.stack(padded_labels, dim=0) if labels is not None else None
        
        # 通过基础模型进行前向传播
        outputs = self.base_model(
            inputs_embeds=final_inputs_embeds,
            attention_mask=final_attention_mask,
            labels=final_labels,
            output_hidden_states=True,
            return_dict=True,
            **kwargs
        )
        
        return outputs