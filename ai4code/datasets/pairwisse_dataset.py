import random
from typing import Dict
import torch
from transformers import AutoTokenizer

from ai4code.datasets import Sample


class PairwiseDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data: Dict[str, Sample],
        tokenizer: AutoTokenizer,
        max_len: int,
        negative_ratio: float,
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.all_cell_keys = []
        self.negative_ratio = negative_ratio
        for sample in list(self.data.values()):
            for cell_key in sample.cell_keys:
                if sample.cell_types[cell_key] == "markdown":
                    self.all_cell_keys.append((cell_key, sample.id))

    def __len__(self):
        return len(self.all_cell_keys)

    @staticmethod
    def ensemble(ori_ids, pair_ids, max_len, cls_token_id, sep_token_id, pad_token_id):
        # adjust the length of ori_ids and pair_ids
        # pattern is [cls] + ori_ids + [sep] + pair_ids + [sep]
        ori_len, pair_len = len(ori_ids), len(pair_ids)
        min_len = min([ori_len, pair_len])
        left_space = (max_len - 3) - min_len * 2

        min_len = min(min_len, (max_len - 3) // 2)
        if ori_len > pair_len:
            ori_ids = ori_ids[: min_len + max(left_space, 0)]
            pair_ids = pair_ids[:min_len]
        else:
            ori_ids = ori_ids[:min_len]
            pair_ids = pair_ids[: min_len + max(left_space, 0)]
        input_ids = (
            [cls_token_id] + ori_ids + [sep_token_id] + pair_ids + [sep_token_id]
        )
        input_ids_len = len(input_ids)
        assert input_ids_len <= max_len
        input_ids += [pad_token_id] * (max_len - input_ids_len)
        attention_masks = [1] * input_ids_len + [0] * (max_len - input_ids_len)
        return input_ids, attention_masks

    def __getitem__(self, index):
        cell_key, sample_id = self.all_cell_keys[index]
        sample = self.data[sample_id]
        n_sample_cells = sample.markdown_cell_count + sample.code_cell_count
        cell_order_index = sample.orders.index(cell_key)

        positive = random.random() > self.negative_ratio
        try:
            next_cell_key = sample.orders[cell_order_index + 1]
        except IndexError:
            assert cell_order_index + 1 == n_sample_cells
            next_cell_key = None

        if positive:
            pair_source = "" if next_cell_key is None else sample.sources[next_cell_key]
        else:
            # random pick a source(not self and self + 1) as negative source
            # TODO: try to pick hard neg sample
            candinates = [
                o for o in sample.orders if o not in (next_cell_key, cell_key)
            ]
            if len(candinates) < 1:
                # TODO:
                # 这些样本应该直接 drop 掉
                pair_source_key = cell_key
            else:
                pair_source_key = random.sample(candinates, k=1)[0]
            pair_source = sample.sources[pair_source_key]

        # concatenate the source and pair source
        source = sample.sources[cell_key]
        ori_ids, pair_ids = self.tokenizer.batch_encode_plus(
            [source, pair_source],
            add_special_tokens=False,
        )["input_ids"]

        input_ids, attention_masks = self.ensemble(
            ori_ids,
            pair_ids,
            self.max_len,
            self.tokenizer.cls_token_id,
            self.tokenizer.sep_token_id,
            self.tokenizer.pad_token_id,
        )

        return (
            torch.tensor(input_ids).long(),
            torch.tensor(attention_masks).long(),
            torch.tensor([positive]).float(),
            torch.tensor([sample.code_cell_count, sample.markdown_cell_count]),
            cell_key,
            sample.id,
        )