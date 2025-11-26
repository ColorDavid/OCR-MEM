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

from transformers.modeling_utils import PreTrainedModel

@registry.register_model("memmodel")
class MEMModel(PreTrainedModel):
    """
    带有视觉记忆压缩的模型
    
    在每个user消息位置检查历史长度，如果超过阈值则进行视觉压缩增强。
    """

    def __init__(
        self, 
        base_model_name: str, 
        ocr_model_name: str,
        vision_embedding_size: int = 512,
        context_threshold: int = 2048,
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

    
    @classmethod
    def from_config(cls, config: dict):
        """从配置字典创建模型实例"""
        return cls(
            base_model_name=config.get("base_model_name"),
            ocr_model_name=config.get("ocr_model_name"),
            vision_embedding_size=config.get("vision_embedding_size"),
            context_threshold=config.get("context_threshold", 2048),
        )
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> CausalLMOutputWithPast:
        """
        前向传播函数
        
        处理流程：
        1. 找到最后一个assistant role的位置
        2. 如果之前的历史超过context_threshold，将其压缩为视觉嵌入
        3. 保留最后一个assistant及之后的内容
        4. 正确处理维度变化
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        hidden_size = self.base_model_config.hidden_size
        
        # 获取原始输入嵌入
        inputs_embeds = self.base_model.get_input_embeddings()(input_ids)
        
        # 识别assistant消息位置 (使用"<|im_start|>assistant\n"模式)
        assistant_pattern_ids = self.tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
        pattern_len = len(assistant_pattern_ids)
        
        # 为每个batch构建增强后的序列
        final_embeds_list = []
        final_masks_list = []
        final_labels_list = [] if labels is not None else None
        
        for b in range(batch_size):
            seq = input_ids[b].tolist()
            
            # 找到所有assistant消息的起始位置
            assistant_positions = []
            for i in range(seq_len - pattern_len + 1):
                if seq[i:i+pattern_len] == assistant_pattern_ids:
                    assistant_positions.append(i)
            
            # 如果没有assistant位置，或者历史未超过阈值，直接使用原始输入
            if not assistant_positions:
                final_embeds_list.append(inputs_embeds[b])
                final_masks_list.append(attention_mask[b])
                if labels is not None:
                    final_labels_list.append(labels[b])
                continue
            
            # 找到最后一个assistant的位置
            last_assistant_pos = assistant_positions[-1]
            
            # 检查历史长度（最后一个assistant之前的内容）
            history_length = last_assistant_pos
            
            if history_length <= self.context_threshold:
                # 历史未超过阈值，直接使用原始输入
                final_embeds_list.append(inputs_embeds[b])
                final_masks_list.append(attention_mask[b])
                if labels is not None:
                    final_labels_list.append(labels[b])
                continue
            
            # 需要进行视觉压缩
            # 1. 提取历史部分（0到last_assistant_pos）
            history_ids = input_ids[b, :last_assistant_pos]
            history_text = self.tokenizer.decode(history_ids, skip_special_tokens=False)
            
            # 2. 将文本渲染为图像
            images = self.render_func([history_text])  # 返回PIL.Image列表
            
            # 3. 通过OCR编码器提取视觉特征
            with torch.no_grad():
                vision_features = self.ocr_embed(images)  # [1, vision_embedding_size]
            
            # 4. 投影到语言模型的隐藏空间
            vision_embeds = self.proj(vision_features)  # [1, hidden_size]
            
            # 5. 提取保留部分（last_assistant_pos到结尾）
            remaining_embeds = inputs_embeds[b, last_assistant_pos:]  # [remaining_len, hidden_size]
            remaining_mask = attention_mask[b, last_assistant_pos:]   # [remaining_len]
            
            # 6. 拼接：vision_embeds + remaining_embeds
            current_embeds = torch.cat([vision_embeds, remaining_embeds], dim=0)  # [1+remaining_len, hidden_size]
            current_mask = torch.cat([
                torch.ones(vision_embeds.shape[0], device=device, dtype=attention_mask.dtype),
                remaining_mask
            ], dim=0)  # [1+remaining_len]
            
            # 7. 处理labels
            if labels is not None:
                remaining_labels = labels[b, last_assistant_pos:]  # [remaining_len]
                # vision token对应的label设为-100（不计算loss）
                vision_labels = torch.full(
                    (vision_embeds.shape[0],), -100, device=device, dtype=labels.dtype
                )
                current_labels = torch.cat([vision_labels, remaining_labels], dim=0)  # [1+remaining_len]
            else:
                current_labels = None
            
            final_embeds_list.append(current_embeds)
            final_masks_list.append(current_mask)
            if labels is not None:
                final_labels_list.append(current_labels)
        
        # Padding所有序列到相同长度（使用左padding保持因果关系）
        max_len = max(emb.shape[0] for emb in final_embeds_list)
        
        padded_embeds = []
        padded_masks = []
        padded_labels = [] if labels is not None else None
        
        for i in range(batch_size):
            pad_len = max_len - final_embeds_list[i].shape[0]
            
            # 左padding embeddings
            if pad_len > 0:
                pad_embeds = torch.zeros(pad_len, hidden_size, device=device, dtype=inputs_embeds.dtype)
                padded_emb = torch.cat([pad_embeds, final_embeds_list[i]], dim=0)
            else:
                padded_emb = final_embeds_list[i]
            padded_embeds.append(padded_emb)
            
            # 左padding attention mask
            if pad_len > 0:
                pad_mask = torch.zeros(pad_len, device=device, dtype=attention_mask.dtype)
                padded_mask = torch.cat([pad_mask, final_masks_list[i]], dim=0)
            else:
                padded_mask = final_masks_list[i]
            padded_masks.append(padded_mask)
            
            # 左padding labels
            if labels is not None:
                if pad_len > 0:
                    pad_labels = torch.full((pad_len,), -100, device=device, dtype=labels.dtype)
                    padded_label = torch.cat([pad_labels, final_labels_list[i]], dim=0)
                else:
                    padded_label = final_labels_list[i]
                padded_labels.append(padded_label)
        
        # Stack成batch tensors
        final_embeds = torch.stack(padded_embeds, dim=0)  # [batch_size, max_len, hidden_size]
        final_masks = torch.stack(padded_masks, dim=0)     # [batch_size, max_len]
        final_labels = torch.stack(padded_labels, dim=0) if labels is not None else None
        
        # 生成position_ids：根据attention_mask计算累积位置
        # 关键：position_ids要反映实际的序列位置，考虑padding
        position_ids = torch.zeros_like(final_masks, dtype=torch.long)
        for i in range(batch_size):
            # 找到非padding部分
            non_pad_mask = final_masks[i] == 1
            non_pad_indices = non_pad_mask.nonzero(as_tuple=True)[0]
            if len(non_pad_indices) > 0:
                # 从0开始递增
                position_ids[i, non_pad_indices] = torch.arange(
                    len(non_pad_indices), 
                    device=device, 
                    dtype=torch.long
                )
        
        # 调用base_model进行前向传播
        outputs = self.base_model(
            inputs_embeds=final_embeds,
            attention_mask=final_masks,
            position_ids=position_ids,
            labels=final_labels,
            **kwargs
        )
        
        return outputs
            
