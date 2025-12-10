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
from transformers import PretrainedConfig, PreTrainedModel

import random
from typing import Tuple, Optional, List, Union
import logging
from pathlib import Path
import json
from utils.render import text_to_images
from deepseek_ocr_encoder import DeepSeekOCREncoder


class MEMConfig(PretrainedConfig):
    model_type = "mem_model"
    
    def __init__(
        self, 
        base_model_name: str = "",
        ocr_model_name: str = "",
        vision_embedding_size: int = 1024,
        context_threshold: int = 2048,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.base_llm_model_name = base_model_name
        self.ocr_model_name = ocr_model_name
        self.vision_embedding_size = vision_embedding_size
        self.context_threshold = context_threshold
        
        # 修复：在加载预训练模型时允许空参数，但在实例化新模型时进行验证
        # 这避免了from_pretrained时的不必要警告


class MEMModel(PreTrainedModel):
    """
    带有视觉记忆压缩的模型
    
    在每个user消息位置检查历史长度，如果超过阈值则进行视觉压缩增强。
    支持DeepSpeed ZeRO优化和HuggingFace Trainer。
    """
    config_class = MEMConfig
    base_model_prefix = "base_model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["proj"]
    _skip_keys_device_placement = ["past_key_values"]

    def __init__(self, config: MEMConfig):
        super().__init__(config)
        
        # 验证配置参数（修复：避免运行时的配置错误）
        if not config.base_llm_model_name:
            raise ValueError(
                "config.base_llm_model_name 不能为空！\n"
                "请在创建MEMConfig时指定base_model_name参数。"
            )
        if not config.ocr_model_name:
            raise ValueError(
                "config.ocr_model_name 不能为空！\n"
                "请在创建MEMConfig时指定ocr_model_name参数。"
            )
        
        # 基础语言模型
        self.base_llm_model = AutoModelForCausalLM.from_pretrained(
            config.base_llm_model_name, 
            torch_dtype=torch.bfloat16, 
            attn_implementation="flash_attention_2"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(config.base_llm_model_name)
        self.base_llm_model_config = self.base_llm_model.config
        
        # 全局配置参数
        self.context_threshold = config.context_threshold

        # 文本到图像渲染函数
        self.render_func = text_to_images
        
        # OCR视觉编码器
        self.ocr_embed = DeepSeekOCREncoder.from_pretrained(config.ocr_model_name)
        
        # 冻结base_model和ocr_embed
        for param in self.base_llm_model.parameters():
            param.requires_grad = False
        for param in self.ocr_embed.parameters():
            param.requires_grad = False
        
        # 投影层：将视觉特征投影到语言模型的隐藏空间
        # 使用与base_model相同的dtype确保一致性
        mlp_hidden_dim = max(config.vision_embedding_size, self.base_llm_model_config.hidden_size)
        self.proj = nn.Sequential(
            nn.Linear(config.vision_embedding_size, mlp_hidden_dim, dtype=torch.bfloat16),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, self.base_llm_model_config.hidden_size, dtype=torch.bfloat16)
        )
        
        # 初始化投影层权重
        self._init_proj_weights()

    def _init_proj_weights(self):
        """初始化投影层权重"""
        for module in self.proj:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def get_input_embeddings(self):
        """返回输入嵌入层"""
        return self.base_llm_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        """设置输入嵌入层"""
        self.base_llm_model.set_input_embeddings(value)

    def get_output_embeddings(self):
        """返回输出嵌入层（用于语言模型头）"""
        return self.base_llm_model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        """设置输出嵌入层"""
        self.base_llm_model.set_output_embeddings(new_embeddings)

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """启用梯度检查点以节省内存"""
        self.base_llm_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs)

    def gradient_checkpointing_disable(self):
        """禁用梯度检查点"""
        self.base_llm_model.gradient_checkpointing_disable()

    @property
    def is_gradient_checkpointing(self) -> bool:
        """检查是否启用了梯度检查点"""
        return getattr(self.base_llm_model, "is_gradient_checkpointing", False)

    def train(self, mode: bool = True):
        """
        设置训练模式
        
        重要说明：
        虽然base_model和ocr_embed的参数被冻结(requires_grad=False)，
        但它们仍需要保持在train模式以正确传播梯度到可训练的proj层。
        将冻结模型设为eval()会完全断开梯度计算图。
        
        正确做法：
        1. 参数冻结通过requires_grad=False实现
        2. 所有模块保持train模式以维持计算图
        3. 如需特殊行为(如禁用dropout)，在forward中单独处理
        """
        super().train(mode)
        # 关键修复：不要将base_model设为eval模式！
        # 冻结参数(requires_grad=False)已经足够防止参数更新
        # 设为eval()会断开整个计算图的梯度传播
        if mode:
            self.base_llm_model.train()  # 保持train模式以传播梯度
            self.ocr_embed.train()       # 保持train模式以传播梯度
            self.proj.train()
        else:
            self.base_llm_model.eval()
            self.ocr_embed.eval()
            self.proj.eval()
        return self

    def get_trainable_parameters(self) -> List[nn.Parameter]:
        """返回可训练参数列表（用于优化器）"""
        return [p for p in self.proj.parameters() if p.requires_grad]

    def print_trainable_parameters(self):
        """打印可训练参数统计和详细信息"""
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        all_params = sum(p.numel() for p in self.parameters())
        
        print(
            f"trainable params: {trainable_params:,} || "
            f"all params: {all_params:,} || "
            f"trainable%: {100 * trainable_params / all_params:.4f}%"
        )
        
        # 验证冻结状态
        base_trainable = sum(p.numel() for p in self.base_llm_model.parameters() if p.requires_grad)
        ocr_trainable = sum(p.numel() for p in self.ocr_embed.parameters() if p.requires_grad)
        proj_trainable = sum(p.numel() for p in self.proj.parameters() if p.requires_grad)
        
        print(f"\nDetailed breakdown:")
        print(f"  base_llm_model trainable params: {base_trainable:,} (should be 0)")
        print(f"  ocr_embed trainable params: {ocr_trainable:,} (should be 0)")
        print(f"  proj trainable params: {proj_trainable:,}")
        
        # 警告：如果冻结的模块有可训练参数
        if base_trainable > 0:
            print(f"  ⚠️ WARNING: base_llm_model should be frozen but has {base_trainable:,} trainable params!")
        if ocr_trainable > 0:
            print(f"  ⚠️ WARNING: ocr_embed should be frozen but has {ocr_trainable:,} trainable params!")

    @classmethod
    def from_pretrained(
        cls, 
        pretrained_model_name_or_path: str, 
        *model_args,
        config: Optional[MEMConfig] = None,
        **kwargs
    ):
        """
        从预训练路径加载模型
        
        Args:
            pretrained_model_name_or_path: 模型保存路径
            config: 可选的配置对象，如果提供则使用此配置
            **kwargs: 传递给模型初始化的其他参数
        """
        pretrained_path = Path(pretrained_model_name_or_path)
        config_path = pretrained_path / "config.json"
        
        # 加载或使用提供的配置
        if config is None:
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config_dict = json.load(f)
                config = MEMConfig(**config_dict)
            else:
                raise ValueError(f"Config file not found at {config_path}")
        
        # 处理kwargs中的设备和dtype参数
        device_map = kwargs.pop("device_map", None)
        torch_dtype = kwargs.pop("torch_dtype", None)
        
        # 创建模型实例
        model = cls(config)
        
        # 加载投影层权重
        proj_path = pretrained_path / "projection_layer.pt"
        if proj_path.exists():
            proj_state_dict = torch.load(proj_path, map_location="cpu", weights_only=True)
            model.proj.load_state_dict(proj_state_dict)
            logging.info(f"Loaded projection layer from {proj_path}")
        else:
            logging.warning(f"Projection layer weights not found at {proj_path}, using initialized weights")
        
        # 处理设备映射（DeepSpeed兼容）
        if device_map is not None:
            model = model.to(device_map) if isinstance(device_map, (str, torch.device)) else model
        
        # 处理dtype转换
        if torch_dtype is not None:
            model.proj = model.proj.to(dtype=torch_dtype)
        
        return model

    def save_pretrained(
        self, 
        save_directory: str, 
        is_main_process: bool = True,
        state_dict: Optional[dict] = None,
        save_function: callable = torch.save,
        **kwargs
    ):
        """
        保存模型到指定目录
        
        Args:
            save_directory: 保存目录路径
            is_main_process: 是否为主进程（DeepSpeed/分布式训练）
            state_dict: 可选的state_dict，如果提供则使用此dict
            save_function: 保存函数，默认为torch.save
            **kwargs: 其他参数
        """
        if not is_main_process:
            return
        
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)
        
        # 保存配置（包含model_type）
        config_dict = {
            "model_type": self.config.model_type,
            "base_llm_model_name": self.config.base_llm_model_name,
            "ocr_model_name": self.config.ocr_model_name,
            "vision_embedding_size": self.config.vision_embedding_size,
            "context_threshold": self.config.context_threshold,
        }
        
        with open(save_directory / "config.json", 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        # 只保存投影层的权重（base_model和ocr_embed保持冻结）
        if state_dict is not None:
            # 从完整state_dict中提取proj相关的权重
            proj_state_dict = {
                k.replace("proj.", ""): v 
                for k, v in state_dict.items() 
                if k.startswith("proj.")
            }
        else:
            proj_state_dict = self.proj.state_dict()
        
        save_function(proj_state_dict, save_directory / "projection_layer.pt")
        
        logging.info(f"Model saved to {save_directory}")

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
        
        梯度流说明：
        - input_ids -> embedding层(冻结) -> inputs_embeds [有grad_fn]
        - 历史文本 -> OCR编码器(冻结) -> vision_features [有grad_fn] 
        - vision_features -> proj层(可训练) -> vision_embeds [有grad_fn]
        - 拼接后 -> base_llm(冻结) -> logits [有grad_fn] -> loss [有grad_fn]
        
        关键点：
        - 虽然base_llm和ocr_embed参数冻结，但forward必须在计算图中
        - 只有proj层的参数会更新，但梯度需要流经整个网络
        - 不能使用torch.no_grad()、detach()或eval()模式
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        hidden_size = self.base_llm_model_config.hidden_size
        
        # 获取原始输入嵌入 - 此操作在计算图中，即使embedding层冻结
        inputs_embeds = self.base_llm_model.get_input_embeddings()(input_ids)
        
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
            
            # 2. 将文本渲染为图像（可能返回多张图片的列表）
            images = self.render_func(history_text)
            if not isinstance(images, list):
                images = [images]
            
            # 3. 通过OCR编码器提取每张图像的视觉特征并拼接
            # 
            # 关键问题：DeepSeekOCREncoder 所有方法都使用了 @torch.inference_mode() 装饰器
            # 包括 encode(), _forward_core() 等,这会完全断开计算图！
            # 
            # 解决方案：完全绕过所有装饰的方法，直接调用底层的 SAM + CLIP 组件
            # 手动实现前向传播逻辑，确保梯度可以流动
            vision_features_list = []
            for img in images:
                # 预处理图像 (使用OCR encoder的预处理)
                if isinstance(img, str):
                    from PIL import Image as PILImage
                    img_pil = PILImage.open(img).convert("RGB")
                else:
                    img_pil = img.convert("RGB")
                
                # 使用OCR encoder的预处理pipeline
                if hasattr(self.ocr_embed, '_preproc_1024') and self.ocr_embed._preproc_1024 is not None:
                    x_cpu = self.ocr_embed._preproc_1024(img_pil).unsqueeze(0)
                else:
                    # 如果没有预处理pipeline，使用默认的
                    import torchvision.transforms as T
                    preprocess = T.Compose([
                        T.Resize((1024, 1024)),
                        T.ToTensor(),
                        T.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                  std=(0.26862954, 0.26130258, 0.27577711))
                    ])
                    x_cpu = preprocess(img_pil).unsqueeze(0)
                
                # 移到设备并设置为channels_last格式
                x = x_cpu.to(self.ocr_embed.device, dtype=self.ocr_embed.dtype)
                x = x.to(memory_format=torch.channels_last)
                
                # ==========================================
                # 手动实现 _forward_core 的逻辑,但不使用装饰器
                # 这样可以保持梯度流动
                # ==========================================
                
                # Step 1: SAM encoder
                sam_out = self.ocr_embed.sam(x)  # [1, 1024, Hs, Ws]
                B, C, Hs, Ws = sam_out.shape
                
                # Step 2: Flatten and transpose
                tokens = sam_out.flatten(2).transpose(1, 2).contiguous()  # [1, N, 1024]
                
                # Step 3: Add position embeddings
                # 使用预计算的位置编码（如果可用）
                if Hs == 16 and Ws == 16 and hasattr(self.ocr_embed, '_pos_fixed_16') and self.ocr_embed._pos_fixed_16 is not None:
                    pos = self.ocr_embed._pos_fixed_16
                else:
                    # 动态计算位置编码
                    import math
                    table = self.ocr_embed.clip_pos_table
                    grid_size = int(math.isqrt(table.size(0) - 1))
                    base_grid = table[1: 1 + grid_size * grid_size].view(grid_size, grid_size, self.ocr_embed.embed_dim)
                    base_grid = base_grid.permute(2, 0, 1).unsqueeze(0)  # [1,1024,G,G]
                    if (Hs, Ws) != (grid_size, grid_size):
                        base_grid = torch.nn.functional.interpolate(
                            base_grid, size=(Hs, Ws), mode="bicubic", align_corners=False
                        )
                    pos = base_grid.flatten(2).transpose(1, 2).contiguous()  # [1,N,1024]
                
                # Step 4: CLIP processing
                tokens_plus = tokens + pos
                x_tok = self.ocr_embed.clip_pre(tokens_plus)
                x_tok = self.ocr_embed.clip_tr(x_tok)
                img_features = tokens + x_tok  # [1, N, 1024]
                
                # 处理不同维度情况，最终得到 [N, 1024] 形状
                if img_features.dim() == 3:
                    # [1, N, 1024] -> [N, 1024] (去除batch维度)
                    img_features = img_features.squeeze(0)
                elif img_features.dim() == 2:
                    # 已经是 [N, 1024]，无需处理
                    pass
                elif img_features.dim() == 1:
                    # [1024] -> [1, 1024] (单token情况)
                    img_features = img_features.unsqueeze(0)
                vision_features_list.append(img_features)
            
            # 拼接所有图像的特征: [total_tokens, 1024]
            vision_features = torch.cat(vision_features_list, dim=0)
            
            # 4. 投影到语言模型的隐藏空间: [total_tokens, hidden_size]
            vision_embeds = self.proj(vision_features)
            
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
                # 关键修复：不设置requires_grad=False！
                # 虽然padding本身是常数，但不能断开梯度链
                # torch.cat会自动保持requires_grad=True（如果任一输入有梯度）
                pad_embeds = torch.zeros(
                    pad_len, hidden_size, 
                    device=device, 
                    dtype=inputs_embeds.dtype
                )
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
        outputs = self.base_llm_model(
            inputs_embeds=final_embeds,
            attention_mask=final_masks,
            position_ids=position_ids,
            labels=final_labels,
            **kwargs
        )
        
        return outputs