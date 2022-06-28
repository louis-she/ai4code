import random
import torch
from typing import Dict
from transformers import AutoTokenizer

from ai4code.datasets import Sample


class RankDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data: Dict[str, Sample],
        tokenizer: AutoTokenizer,
        cell_token_size,
        cell_stride,
        context_cells_token_size,
        context_stride,
        max_len,
        ordered_context_ratio,
        shuffle_markdowns=True,
        with_code_cell=False,
        english_only=False,
    ):
        self.read_count = 0
        self.data = data
        self.ordered_context_ratio = ordered_context_ratio
        self.context_cells_token_size = context_cells_token_size
        self.context_stride = context_stride
        self.shuffle_markdowns = shuffle_markdowns
        self.cell_token_size = cell_token_size
        self.cell_stride = cell_stride
        self.max_len = max_len
        self.english_only = english_only
        self.all_cells = []

        for sample in list(self.data.values()):
            for cell_key in sample.cell_keys:
                if sample.cell_types[cell_key] == "markdown":
                    self.all_cells.append((sample.id, cell_key))

        self.tokenizer = tokenizer
        self.hash_id = self.tokenizer.encode("#", add_special_tokens=False)[0]

    def __len__(self):
        return len(self.all_cells)

    def preprocess(self, ids):
        if self.english_only:
            ids = [
                input_id
                for input_id in ids
                if (input_id >= 1997 and input_id <= 29612) or input_id == self.hash_id
            ]
        return ids

    def __getitem__(self, index: int):
        sample_id, cell_key = self.all_cells[index]
        sample = self.data[sample_id]

        anchor_encode = self.preprocess(sample.cell_encodes[cell_key])

        # 对于 anchor_encode，不要通过 stride 来过滤 # 字符（token 为 1001）
        anchor_encode = [
            x
            for k, x in enumerate(anchor_encode)
            if ((k % self.cell_stride) == 0 or x == self.hash_id)
            and k < (self.cell_token_size * self.cell_stride)
        ]

        input_ids = [self.tokenizer.cls_token_id] + anchor_encode

        # 将 context 分为两种，按概率随机选择其中的一种进行训练
        use_ordered_context = random.random() < self.ordered_context_ratio
        if not use_ordered_context:
            # 1. anchor + code cells
            context_cell_keys = [
                key for key in sample.cell_keys if sample.cell_types[key] == "code"
            ]
        else:
            # 2. anchor + ordered cells (这里不加 anchor 本身)
            context_cell_keys = [key for key in sample.orders if key != cell_key]

        for context_cell_key in context_cell_keys:
            cell_encode = self.preprocess(sample.cell_encodes[context_cell_key])
            context_encode = cell_encode[
                0 : self.context_cells_token_size
                * self.context_stride : self.context_stride
            ]
            input_ids += [self.tokenizer.sep_token_id] + context_encode

        input_ids += [self.tokenizer.sep_token_id]

        input_ids = input_ids[: self.max_len]
        pad_len = self.max_len - len(input_ids)
        input_ids += [self.tokenizer.pad_token_id] * pad_len
        attention_mask = [1] * (self.max_len - pad_len) + [0] * pad_len
        label = sample.cell_ranks_normed[cell_key]

        return (
            torch.tensor(input_ids).long(),
            torch.tensor(attention_mask).long(),
            torch.tensor([label]),
            torch.tensor([sample.code_cell_count, sample.markdown_cell_count]),
            sample_id,
            cell_key,
        )