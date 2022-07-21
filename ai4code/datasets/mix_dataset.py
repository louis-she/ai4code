from collections import Counter
import math
import pickle
import random
from typing import Dict
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
from ai4code import utils

from ai4code.datasets.types import Sample, SpecialTokenID


class MixedDatasetWithSplits(torch.utils.data.Dataset):
    def __init__(
        self,
        data: Dict[str, Sample],
        special_tokens: SpecialTokenID,
        anchor_size,
        max_len,
        split_len,
        distil_context=None,
        shuffle_markdowns=True,
        only_task_data=False,
        encode_key=None,
        reverse=False,
        global_keywords=None,
        keywords_thres=5,
    ):
        self.read_count = 0
        self.only_task_data = only_task_data
        self.data = data
        self.split_len = split_len
        self.anchor_size = anchor_size
        self.shuffle_markdowns = shuffle_markdowns
        self.max_len = max_len
        self.all_cells = []
        self.feature_scaler = StandardScaler()
        self.encode_key = encode_key
        self.reverse = reverse
        self.distil_context = distil_context
        self.global_keywords = global_keywords
        self.keywords_thres = keywords_thres

        if self.global_keywords:
            with open(self.global_keywords, "rb"):
                self.global_keywords = pickle.load(self.global_keywords)

        if self.distil_context:
            self.context_cells_keys = pickle.load(open(self.distil_context, "rb"))
        else:
            self.context_cells_keys = {
                sample.id: [
                    key for key in sample.cell_keys if sample.cell_types[key] == "code"
                ]
                for sample in self.data.values()
            }
        for sample in list(self.data.values()):
            context_cell_keys = self.context_cells_keys[sample.id]
            available_splits_num = max(1, math.ceil(len(context_cell_keys) / self.split_len))
            for split_id in range(available_splits_num):
                # 获取所有之前的 context_keys, 找出最后一个 code cell 的 rank，作为该 split 的 rank_offset
                rank_offset = 0
                previous_context_keys = context_cell_keys[
                    0 : self.split_len * split_id
                ]
                previous_code_context_keys = [
                    key
                    for key in previous_context_keys
                    if sample.cell_types[key] == "code"
                ]
                if len(previous_code_context_keys) > 0:
                    rank_offset = sample.cell_ranks[previous_code_context_keys[-1]]
                for cell_key in sample.cell_keys:
                    if sample.cell_types[cell_key] == "markdown":
                        self.all_cells.append(
                            (sample.id, cell_key, split_id, rank_offset)
                        )

            sample_keywords = set()
            if self.global_keywords:
                keywords_counter = Counter()
                # 获取 sample keywords encodes
                markdown_encode_set = set()
                code_encode_list = []
                for cell_key in sample.cell_keys:
                    cell_encode = sample.cell_encodes[cell_key]
                    cell_type = sample.cell_types[cell_key]
                    if cell_type == "code":
                        code_encode_list += cell_encode
                    elif cell_type == "markdown":
                        markdown_encode_set.update(cell_encode)
                for code_encode in code_encode_list:
                    if code_encode in markdown_encode_set:
                        keywords_counter[code_encode] += 1
                for key, apperance in keywords_counter.most_common()[::-1]:
                    if key in self.global_keywords_encode or apperance > self.keywords_thres:
                        continue
                    sample_keywords.add(key)

        if not self.only_task_data:
            self.all_cell_pairs = []
            for sample in list(self.data.values()):
                for cell_key, next_cell_key, previous_cell_key in zip(
                    sample.orders[1:-1], sample.orders[2:], sample.orders[:-2]
                ):
                    self.all_cell_pairs.append(
                        (sample.id, cell_key, next_cell_key, previous_cell_key)
                    )

        self.special_tokens = special_tokens

    def __len__(self):
        return len(self.all_cells)

    def get_encode(self, sample, cell_key):
        if self.encode_key:
            encode = sample.cell_encodes[self.encode_key][cell_key]
        else:
            encode = sample.cell_encodes[cell_key]
        if self.reverse:
            encode = encode[::-1]
        return encode

    def get_task_data(self, index):
        sample_id, cell_key, split_id, rank_offset = self.all_cells[index]
        sample = self.data[sample_id]

        if isinstance(self.anchor_size, tuple):
            anchor_size = random.sample(range(*self.anchor_size), k=1)[0]
        else:
            anchor_size = self.anchor_size
        anchor_encode = self.get_encode(sample, cell_key)[: anchor_size]

        input_ids = (
            [self.special_tokens.cls_token_id]
            + anchor_encode
            + [self.special_tokens.sep_token_id]
        )

        context_cell_keys = self.context_cells_keys[sample.id]
        context_cell_keys = context_cell_keys[
            split_id * self.split_len : (split_id + 1) * self.split_len
        ]

        context_encodes = []
        context_types = []
        for context_cell_key in context_cell_keys:
            context_encode = self.get_encode(sample, context_cell_key)
            cell_type = sample.cell_types[context_cell_key]
            context_encodes.append(context_encode)
            context_types.append(cell_type)

        _, length_of_seqs = utils.adjust_sequences(
            context_encodes, self.max_len - len(anchor_encode) - self.split_len - 2
        )

        if self.global_keywords:
            # 如果 keywords，则需重新抽取 context_encodes
            pass

        for i, (
            context_encode,
            cell_type,
        ) in enumerate(zip(context_encodes, context_types)):
            sep_token = (
                self.special_tokens.sep_token_id
                if cell_type == "code"
                else self.special_tokens.unk_token_id
            )
            input_ids += context_encode + [sep_token]

        input_ids = input_ids[: self.max_len]
        pad_len = self.max_len - len(input_ids)
        input_ids += [self.special_tokens.pad_token_id] * pad_len
        attention_mask = [1] * (self.max_len - pad_len) + [0] * pad_len

        rank = sample.cell_ranks[cell_key] - rank_offset
        rank_normed = math.log(rank) if rank > 0 else -10
        in_split = rank > 0 and rank < self.split_len + 1

        return (
            torch.tensor(input_ids).long(),
            torch.tensor(attention_mask).long(),
            torch.tensor([in_split, rank_normed]).float(),
            sample_id,
            cell_key,
            split_id,
            rank_offset,
        )

    def get_pair_data(self, index: int):
        index = index % len(self.all_cell_pairs)
        sample_id, anchor_key, next_key, previous_key = self.all_cell_pairs[index]
        sample = self.data[sample_id]

        anchor_encode = sample.cell_encodes[anchor_key]
        positive = random.random() > 0.5
        if positive:
            previous_encode = sample.cell_encodes[previous_key]
            next_encode = sample.cell_encodes[next_key]
        else:
            try:
                previous_encode, next_encode = random.sample(
                    [
                        sample.cell_encodes[x]
                        for x in sample.cell_keys
                        if x not in [anchor_key, next_key, previous_key]
                    ],
                    k=2,
                )
            except ValueError:
                # fallback to positive
                positive = True
                previous_encode = sample.cell_encodes[previous_key]
                next_encode = sample.cell_encodes[next_key]

        (previous_encode, anchor_encode, next_encode), _ = utils.adjust_sequences(
            [previous_encode, anchor_encode, next_encode], self.max_len - 4
        )

        input_ids = (
            [self.special_tokens.cls_token_id]
            + previous_encode
            + [self.special_tokens.sep_token_id]
            + anchor_encode
            + [self.special_tokens.sep_token_id]
            + next_encode
            + [self.special_tokens.sep_token_id]
        )

        input_ids += (self.max_len - len(input_ids)) * [
            self.special_tokens.pad_token_id
        ]
        mask = [1] * len(input_ids) + [0] * (self.max_len - len(input_ids))
        return (
            torch.tensor(input_ids).long(),
            torch.tensor(mask).long(),
            torch.tensor([positive]).float(),
        )

    def __getitem__(self, index: int):
        (
            input_ids,
            mask,
            targets,
            sample_id,
            cell_key,
            split_id,
            rank_offset,
        ) = self.get_task_data(index)
        if not self.only_task_data:
            lm_input_ids, lm_mask, lm_targets = self.get_pair_data(index)
            return (
                input_ids,
                mask,
                lm_input_ids,
                lm_mask,
                targets,
                lm_targets,
                sample_id,
                cell_key,
                split_id,
                rank_offset,
            )
        return (
            input_ids,
            mask,
            targets,
            sample_id,
            cell_key,
            split_id,
            rank_offset,
        )
