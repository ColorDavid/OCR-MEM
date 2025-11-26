"""
Multi-turn Conversation Dataset and DataLoader Implementation
按照 assistant 回复逐步分割对话，只对最后一个 assistant 回复计算 loss
"""

import json
from typing import List, Dict, Optional
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizerBase


class MultiTurnConversationDataset(Dataset):
    """
    多轮对话数据集，将每个对话按 assistant 回复分割成多个训练样本
    
    对于每个样本：
    - 前面的对话（包括前面的 assistant 回复）作为上下文，不计算 loss
    - 最后一个 assistant 回复计算 loss
    """
    
    def __init__(
        self,
        jsonl_path: str,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 2048,
        return_dict: bool = True
    ):
        """
        Args:
            jsonl_path: JSONL 文件路径，每行包含一个 messages 字段
            tokenizer: HuggingFace tokenizer
            max_length: 最大序列长度
            return_dict: 是否返回字典格式（包含原始 messages）
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.return_dict = return_dict
        
        # 读取并处理数据
        self.samples = []
        self._load_and_process_data(jsonl_path)
    
    def _load_and_process_data(self, jsonl_path: str):
        """读取 jsonl 文件并按 assistant 回复分割"""
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line_idx, line in enumerate(f):
                try:
                    data = json.loads(line.strip())
                    messages = data.get('messages', [])
                    
                    if not messages:
                        continue
                    
                    # 找到所有 assistant 消息的位置
                    assistant_indices = [
                        i for i, msg in enumerate(messages) 
                        if msg.get('role') == 'assistant'
                    ]
                    
                    if not assistant_indices:
                        continue
                    
                    # 为每个 assistant 回复创建一个训练样本
                    for assistant_idx in assistant_indices:
                        # 截取到当前 assistant 为止的所有消息
                        sample_messages = messages[:assistant_idx + 1]
                        
                        self.samples.append({
                            'messages': sample_messages,
                            'source_line': line_idx,
                            'assistant_position': assistant_idx
                        })
                        
                except json.JSONDecodeError as e:
                    print(f"Warning: Failed to parse line {line_idx}: {e}")
                    continue
        
        print(f"Loaded {len(self.samples)} samples from {jsonl_path}")
    
    def _create_labels(
        self, 
        input_ids: torch.Tensor, 
        messages: List[Dict[str, str]]
    ) -> torch.Tensor:
        """
        创建 labels，只对最后一个 assistant 的回复计算 loss
        
        Args:
            input_ids: tokenized input ids
            messages: 原始 messages
            
        Returns:
            labels: 与 input_ids 相同形状，最后一个 assistant 部分保留，其他部分为 -100
        """
        # 初始化 labels 全部为 -100（不计算 loss）
        labels = torch.full_like(input_ids, -100)
        
        # 找到最后一个 assistant 消息
        last_assistant_idx = None
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get('role') == 'assistant':
                last_assistant_idx = i
                break
        
        if last_assistant_idx is None:
            return labels
        
        # 使用 tokenizer 的 chat template 来定位最后一个 assistant 的位置
        # 方法：分别 tokenize 到倒数第二个 message 和到最后一个 message
        messages_before_last = messages[:last_assistant_idx]
        
        if messages_before_last:
            # Tokenize 到倒数第二个消息
            tokens_before = self.tokenizer.apply_chat_template(
                messages_before_last,
                tokenize=True,
                add_generation_prompt=True,  # 添加 assistant 开始标记
                return_tensors='pt'
            )
            context_len = tokens_before.shape[1]
        else:
            # 如果只有一个 assistant 消息，从头开始
            # 需要找到 assistant 开始的位置
            # 通过 tokenize system 和 user 来确定
            context_len = 0
            if messages[0].get('role') == 'system':
                system_tokens = self.tokenizer.apply_chat_template(
                    [messages[0]],
                    tokenize=True,
                    add_generation_prompt=False,
                    return_tensors='pt'
                )
                context_len = system_tokens.shape[1]
            
            # 加上 assistant 提示符的长度
            assistant_prompt = self.tokenizer.apply_chat_template(
                [],
                tokenize=True,
                add_generation_prompt=True,
                return_tensors='pt'
            )
            context_len += assistant_prompt.shape[1]
        
        # 最后一个 assistant 的回复部分需要计算 loss
        # 从 context_len 到结尾（不包括 eos）
        seq_len = input_ids.shape[0]
        
        # 找到第一个 eos token 的位置（如果有的话）
        eos_positions = (input_ids == self.tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
        if len(eos_positions) > 0:
            eos_pos = eos_positions[0].item()
            # 包括 eos token
            labels[context_len:eos_pos + 1] = input_ids[context_len:eos_pos + 1]
        else:
            labels[context_len:] = input_ids[context_len:]
        
        return labels
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        messages = sample['messages']
        
        # 使用 chat template 进行 tokenization
        # add_generation_prompt=False 因为最后一个是 assistant 消息
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        
        # Tokenize
        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding=False,
            return_tensors='pt'
        )
        
        input_ids = encoded['input_ids'].squeeze(0)
        attention_mask = encoded['attention_mask'].squeeze(0)
        
        # 创建 labels（只对最后一个 assistant 计算 loss）
        labels = self._create_labels(input_ids, messages)
        
        result = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
        }
        
        # 可选：返回原始 messages 用于调试
        if self.return_dict:
            result['messages'] = messages
            result['source_line'] = sample['source_line']
        
        return result


def collate_fn(batch: List[Dict[str, torch.Tensor]], 
               tokenizer: PreTrainedTokenizerBase) -> Dict[str, torch.Tensor]:
    """
    自定义 collate function，处理变长序列的 padding
    
    Args:
        batch: list of samples from dataset
        tokenizer: tokenizer for padding
        
    Returns:
        batched tensors with padding
    """
    # 提取所有字段
    input_ids_list = [sample['input_ids'] for sample in batch]
    attention_mask_list = [sample['attention_mask'] for sample in batch]
    labels_list = [sample['labels'] for sample in batch]
    
    # Padding
    # 左侧 padding 用于 input_ids 和 attention_mask
    input_ids_padded = torch.nn.utils.rnn.pad_sequence(
        input_ids_list,
        batch_first=True,
        padding_value=tokenizer.pad_token_id
    )
    
    attention_mask_padded = torch.nn.utils.rnn.pad_sequence(
        attention_mask_list,
        batch_first=True,
        padding_value=0
    )
    
    # labels 使用 -100 padding
    labels_padded = torch.nn.utils.rnn.pad_sequence(
        labels_list,
        batch_first=True,
        padding_value=-100
    )
    
    result = {
        'input_ids': input_ids_padded,
        'attention_mask': attention_mask_padded,
        'labels': labels_padded,
    }
    
    # 可选：保留 messages 信息
    if 'messages' in batch[0]:
        result['messages'] = [sample['messages'] for sample in batch]
        result['source_line'] = [sample['source_line'] for sample in batch]
    
    return result


def create_dataloader(
    jsonl_path: str,
    tokenizer: PreTrainedTokenizerBase,
    batch_size: int = 8,
    max_length: int = 2048,
    shuffle: bool = True,
    num_workers: int = 0,
    return_dict: bool = True
) -> DataLoader:
    """
    创建 DataLoader
    
    Args:
        jsonl_path: JSONL 文件路径
        tokenizer: HuggingFace tokenizer
        batch_size: batch size
        max_length: 最大序列长度
        shuffle: 是否打乱数据
        num_workers: 数据加载的进程数
        return_dict: 是否在 batch 中包含原始 messages
        
    Returns:
        DataLoader
    """
    # 确保 tokenizer 有 pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # 创建 dataset
    dataset = MultiTurnConversationDataset(
        jsonl_path=jsonl_path,
        tokenizer=tokenizer,
        max_length=max_length,
        return_dict=return_dict
    )
    
    # 创建 dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=lambda batch: collate_fn(batch, tokenizer),
        pin_memory=True
    )
    
    return dataloader


# ===== 示例使用代码 =====

def example_usage():
    """示例：如何使用这个 dataset 和 dataloader"""
    from transformers import AutoTokenizer
    
    # 1. 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    
    # 2. 创建 dataloader
    dataloader = create_dataloader(
        jsonl_path="path/to/your/data.jsonl",
        tokenizer=tokenizer,
        batch_size=4,
        max_length=2048,
        shuffle=True,
        num_workers=4,
        return_dict=True
    )
    
    # 3. 迭代数据
    for batch_idx, batch in enumerate(dataloader):
        input_ids = batch['input_ids']  # [batch_size, seq_len]
        attention_mask = batch['attention_mask']  # [batch_size, seq_len]
        labels = batch['labels']  # [batch_size, seq_len]
        
        print(f"Batch {batch_idx}:")
        print(f"  input_ids shape: {input_ids.shape}")
        print(f"  attention_mask shape: {attention_mask.shape}")
        print(f"  labels shape: {labels.shape}")
        
        # 检查 labels 中非 -100 的位置（需要计算 loss 的位置）
        for i in range(len(labels)):
            non_ignore_mask = labels[i] != -100
            num_tokens_to_predict = non_ignore_mask.sum().item()
            print(f"  Sample {i}: {num_tokens_to_predict} tokens to predict")
        
        # 只处理几个 batch 作为示例
        if batch_idx >= 2:
            break


def create_sample_jsonl(output_path: str = "sample_data.jsonl"):
    """创建一个示例 JSONL 文件用于测试"""
    sample_data = [
        {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello, how are you?"},
                {"role": "assistant", "content": "I'm doing well, thank you! How can I help you today?"},
                {"role": "user", "content": "Can you explain what Python is?"},
                {"role": "assistant", "content": "Python is a high-level programming language known for its simplicity and readability."},
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "What's 2+2?"},
                {"role": "assistant", "content": "2+2 equals 4."},
                {"role": "user", "content": "And 3+3?"},
                {"role": "assistant", "content": "3+3 equals 6."},
            ]
        }
    ]
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in sample_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"Created sample JSONL file at: {output_path}")


if __name__ == "__main__":
    # 创建示例数据
    create_sample_jsonl("sample_data.jsonl")
    
    # 运行示例
    print("\n" + "="*50)
    print("Running example usage...")
    print("="*50 + "\n")
    
    # 注意：你需要有实际的 tokenizer 才能运行这个示例
    # example_usage()
    
    print("\nTo use this in your code:")
    print("1. Import: from data import create_dataloader")
    print("2. Create dataloader with your JSONL file")
    print("3. Iterate and train your model")