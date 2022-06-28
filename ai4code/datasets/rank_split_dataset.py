import math
import pickle
from typing import Dict
import torch

from ai4code.datasets import Sample, SpecialTokenID


class RankDatasetWithSplits(torch.utils.data.Dataset):
    def __init__(
        self,
        data: Dict[str, Sample],
        special_tokens: SpecialTokenID,
        cell_token_size,
        cell_stride,
        context_stride,
        max_len,
        split_len,
        distil_context,
        shuffle_markdowns=True,
    ):
        self.read_count = 0
        self.data = data
        self.split_len = split_len
        self.context_stride = context_stride
        self.shuffle_markdowns = shuffle_markdowns
        self.cell_token_size = cell_token_size
        self.cell_stride = cell_stride
        self.max_len = max_len
        self.all_cells = []
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
        self.special_tokens = special_tokens

    def __len__(self):
        return len(self.all_cells)

    def __getitem__(self, index: int):
        sample_id, cell_key, split_id = self.all_cells[index]
        sample = self.data[sample_id]

        anchor_encode = sample.cell_encodes[cell_key]
        # 对于 anchor_encode，不要通过 stride 来过滤 # 字符（token 为 1001）
        # 对于不同的 tokenizer 这里
        anchor_encode = [
            x
            for k, x in enumerate(anchor_encode)
            if ((k % self.cell_stride) == 0 or x == self.hash_id)
            and k < (self.cell_token_size * self.cell_stride)
        ]

        input_ids = (
            [self.special_tokens.cls_token_id]
            + anchor_encode
            + [self.special_tokens.sep_token_id]
        )
        input_stride_ids = anchor_encode + [-100] + [self.special_tokens.sep_token_id]

        context_cell_keys = self.context_cells_keys[sample.id]
        context_cell_keys = context_cell_keys[
            split_id * self.split_len : (split_id + 1) * self.split_len
        ]

        context_encodes = []
        context_stride_encodes = []
        context_lens = []
        context_types = []
        for context_cell_key in context_cell_keys:
            cell_encode = sample.cell_encodes[context_cell_key]
            cell_type = sample.cell_types[context_cell_key]
            context_encode = cell_encode[0 :: self.context_stride]

            # self.context_stride should always be 2 here
            context_stride_encode = cell_encode[1 :: self.context_stride]
            context_stride_encode += [self.special_tokens.pad_token_id] * (
                len(context_encode) - len(context_stride_encode)
            )

            context_encodes.append(context_encode)
            context_stride_encodes.append(context_stride_encode)
            context_lens.append(len(context_encode))
            context_types.append(cell_type)

        current_total_length = sum(context_lens)
        cut_off_number = (
            current_total_length
            - self.max_len
            + len(anchor_encode)
            + self.split_len
            + 2
        )
        if cut_off_number > 0:
            for _ in range(cut_off_number):
                max_index = context_lens.index(max(context_lens))
                context_lens[max_index] -= 1
        for i, (
            context_encode,
            context_len,
            context_stride_encode,
            cell_type,
        ) in enumerate(
            zip(context_encodes, context_lens, context_stride_encodes, context_types)
        ):
            sep_token = (
                self.special_tokens.sep_token_id
                if cell_type == "code"
                else self.special_tokens.unk_token_id
            )
            input_ids += context_encode[:context_len] + [sep_token]
            input_stride_ids += context_stride_encode[:context_len] + [sep_token]

        input_ids = input_ids[: self.max_len]
        input_stride_ids = input_stride_ids[: self.max_len]
        pad_len = self.max_len - len(input_ids)
        input_ids += [self.special_tokens.pad_token_id] * pad_len
        input_stride_ids += [-100] * pad_len
        attention_mask = [1] * (self.max_len - pad_len) + [0] * pad_len

        # start_rank: 1 or 9 or 17 ...
        offset = sample.cell_ranks[context_cell_keys[0]]
        rank = sample.cell_ranks[cell_key] + 1 - offset
        rank_normed = rank / (self.split_len + 1)
        in_split = float(rank_normed > 0 and rank_normed < 1)

        return (
            torch.tensor(input_ids).long(),
            torch.tensor(input_stride_ids).long(),
            torch.tensor(attention_mask).long(),
            torch.tensor([in_split, rank_normed]),
            torch.tensor([sample.code_cell_count, sample.markdown_cell_count]),
            sample_id,
            cell_key,
            split_id,
        )