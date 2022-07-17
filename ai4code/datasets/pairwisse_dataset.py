from collections import defaultdict
import random
from typing import Dict, List
import torch
from transformers import AutoTokenizer

from ai4code.datasets import Sample
from ai4code.datasets.types import SpecialTokenID
from ai4code.utils import adjust_sequences


class PairwiseDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data: Dict[str, Sample],
        special_tokens: SpecialTokenID,
        val: bool = False,
        max_len: int = 256,
        negative_ratio: float = 0.5,
        code_span: int = 1,
    ):
        self.val = val
        self.data = data
        self.special_tokens = special_tokens
        self.max_len = max_len
        self.all_cell_keys = []
        self.negative_ratio = negative_ratio
        self.code_span = code_span
        self.sample_ids = []

        # 获得 cluster 的 markdown
        self.markdown_segments = {}
        for sample in list(self.data.values()):
            processed_list = []
            tmp_code_list = []
            for cell_key in sample.orders:
                # 第一次遍历，去掉连续小于 code_span 中的所有 code
                cell_type = sample.cell_types[cell_key]
                if cell_type == "code":
                    tmp_code_list.append(cell_key)
                if cell_type == "markdown":
                    if len(tmp_code_list) > self.code_span:
                        # 跨度足够大的 code 段，加到处理后的数组中
                        processed_list += tmp_code_list
                    tmp_code_list = []
                    processed_list.append(cell_key)

            # processed_list 是符合条件的 code 以及 markdown
            # 直接将 markdown 的段取出来
            markdown_seg = []
            has_valid_seg = False
            for cell_key in processed_list:
                cell_type = sample.cell_types[cell_key]
                if cell_type == "markdown":
                    markdown_seg.append(cell_key)
                if cell_type == "code":
                    if len(markdown_seg) > 1:
                        if sample.id not in self.markdown_segments:
                            self.markdown_segments[sample.id] = []
                        self.markdown_segments[sample.id].append(markdown_seg)
                        has_valid_seg = True
                    markdown_seg = []
            if len(markdown_seg) != 0:
                if sample.id not in self.markdown_segments:
                    self.markdown_segments[sample.id] = []
                self.markdown_segments[sample.id].append(markdown_seg)
                has_valid_seg = True
            if has_valid_seg:
                self.sample_ids.append(sample.id)

    def __len__(self):
        if self.val:
            return len(self.sample_ids) * 5
        else:
            return len(self.sample_ids)

    def __getitem__(self, index):
        sample = self.data[self.sample_ids[index]]
        markdown_segments = self.markdown_segments[sample.id]

        if random.random() < self.negative_ratio and len(markdown_segments) > 1:
            # negative sample
            label = False
            mdsegs = random.sample(markdown_segments, k=2)
            keys_seqs = [random.sample(mdseg, k=1)[0] for mdseg in mdsegs]
        else:
            label = True
            mdseg = random.sample(markdown_segments, k=1)[0]
            anchor_index = random.sample(range(len(mdseg)), k=1)[0]
            if anchor_index == len(mdseg) - 1:
                pair_index = anchor_index - 1
            else:
                pair_index = anchor_index + 1
            keys_seqs = [mdseg[anchor_index], mdseg[pair_index]]

        seqs = [sample.cell_encodes[key] for key in keys_seqs]
        # cls + anchor + pad + pair + pad
        seqs, _ = adjust_sequences(seqs, self.max_len - 3)
        input_ids = (
            [self.special_tokens.cls_token_id]
            + seqs[0]
            + [self.special_tokens.sep_token_id]
            + seqs[1]
            + [self.special_tokens.sep_token_id]
        )

        input_ids = input_ids[: self.max_len]
        pad_len = self.max_len - len(input_ids)
        input_ids += [self.special_tokens.pad_token_id] * pad_len
        attention_mask = [1] * (self.max_len - pad_len) + [0] * pad_len

        return torch.tensor(input_ids).long(), torch.tensor(attention_mask).long(), torch.tensor([label]).float()
