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

        self.distil_context = distil_context

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
            available_splits_num = math.ceil(len(context_cell_keys) / self.split_len)
            for split_id in range(available_splits_num):
                for cell_key in sample.cell_keys:
                    if sample.cell_types[cell_key] == "markdown":
                        self.all_cells.append((sample.id, cell_key, split_id))

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
            return sample.cell_encodes[self.encode_key][cell_key]
        else:
            return sample.cell_encodes[cell_key]

    def get_task_data(self, index):
        sample_id, cell_key, split_id = self.all_cells[index]
        sample = self.data[sample_id]

        anchor_encode = self.get_encode(sample, cell_key)[: self.anchor_size]

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

        context_encodes, indices = utils.adjust_sequences(
            context_encodes, self.max_len - len(anchor_encode) - self.split_len - 2
        )
        context_types = [context_types[i][:k] for i, k in enumerate(indices)]

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

        # start_rank: 1 or 9 or 17 ...
        offset = int(sample.cell_ranks[context_cell_keys[0]])
        rank = sample.cell_ranks[cell_key] - offset
        rank_normed = math.log(rank)

        in_split = float(rank_normed > 0 and rank_normed < 1)

        # 8 + 6 * 11 = 74
        context_feature = np.array([
            sample.markdown_cell_count,
            sample.code_cell_count,
            *sample.percentile_cell_lens,
            sample.mean_cell_lens,
            *sample.percentile_markdown_lens,
            sample.mean_markdown_lens,
            *sample.percentile_code_lens,
            sample.mean_code_lens,
            # *sample.percentile_cell_ids_lens,
            # sample.mean_cell_ids_lens,
            # *sample.percentile_markdown_ids_lens,
            # sample.mean_markdown_ids_lens,
            # *sample.percentile_code_ids_lens,
            # sample.mean_code_ids_lens
        ])

        context_feature[context_feature == 0] = 1e-5
        context_feature = np.log2(context_feature)

        return (
            torch.tensor(input_ids).long(),
            torch.tensor(attention_mask).long(),
            torch.tensor([in_split, rank_normed]),
            torch.tensor(context_feature).float(),
            sample_id,
            cell_key,
            split_id,
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
        input_ids, mask, targets, context_feature, sample_id, cell_key, split_id  = self.get_task_data(index)
        if not self.only_task_data:
            lm_input_ids, lm_mask, lm_targets = self.get_pair_data(index)
            return input_ids, mask, lm_input_ids, lm_mask, targets, lm_targets, context_feature, sample_id, cell_key, split_id
        return input_ids, mask, targets, context_feature, sample_id, cell_key, split_id
