from copy import copy
import math
import random
from typing import Dict, List, Optional
import re
from markdown import markdown
from tokenizers import Tokenizer
import torch
from transformers import (
    AutoTokenizer,
    DistilBertTokenizer,
    DistilBertTokenizerFast,
    LongformerTokenizer,
    LongformerTokenizerFast,
)
from dataclasses import dataclass
from nltk.stem import WordNetLemmatizer
import gc


stemmer = WordNetLemmatizer()


def common_preprocess(text):
    # replace links
    text = re.sub("https?:\/\/[^\s]+", "link", text)
    # remove all single characters
    text = re.sub(r"\s+[a-zA-Z]\s+", " ", text)
    text = re.sub(r"\^[a-zA-Z]\s+", " ", text)
    text = re.sub(r"\s+[a-zA-Z]$", " ", text)
    # substituting multiple spaces with single space
    text = re.sub(r"\s+", " ", text, flags=re.I)
    text = text.lower()

    # Lemmatization
    tokens = text.split()
    tokens = [stemmer.lemmatize(word) for word in tokens]

    return " ".join(tokens)


def code_preprocess(code):
    # replace links
    code = re.sub("https?:\/\/[^\s]+", "link", code)
    # replace [....], {...} to empty
    code = re.sub(r"\s+", " ", code)
    code = re.sub(r"\[.*?\]", " ", code)
    code = re.sub(r"\{.*?\}", " ", code)
    # replace all numbers to number
    code = re.sub(r"[1-9]+", " number ", code)
    code = re.sub(r"[\.\-\_\#\(\)\[\]\{\}\,\:\"\=']", " ", code)

    code = common_preprocess(code)
    return code


def markdown_preprocess(code):
    code = common_preprocess(code)
    return code


@dataclass
class Sample:
    id: str
    sources: Dict[str, str]
    ancestor: str
    parent: str
    orders: List[str]
    markdown_cell_count: int
    code_cell_count: int
    content_sha1: str
    content_len: int
    cell_keys: Dict[str, str]
    cell_sha1s: Dict[str, str]
    cell_lens: Dict[str, int]
    cell_ranks: Dict[str, float]
    cell_ranks_normed: Dict[str, float]
    cell_types: Dict[str, str]
    cell_encodes: Dict[str, List[int]]
    fold: Optional[int] = None


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
        english_only=False
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
            ids = [input_id for input_id in ids if (input_id >= 1997 and input_id <= 29612) or input_id == self.hash_id]
        return ids

    def __getitem__(self, index: int):
        sample_id, cell_key = self.all_cells[index]
        sample = self.data[sample_id]

        anchor_encode = self.preprocess(sample.cell_encodes[cell_key])

        # 对于 anchor_encode，不要通过 stride 来过滤 # 字符（token 为 1001）
        # 对于不同的 tokenizer 这里
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
            context_cell_keys = [key for key in sample.cell_keys if sample.cell_types[key] == "code"]
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


class LongFormerDataset(torch.utils.data.Dataset):
    def __init__(self, data: Dict[str, Sample]):
        self.data = data
        self.tokenizer = LongformerTokenizer.from_pretrained(
            "allenai/longformer-base-4096"
        )
        self.all_cells = []
        for sample in list(self.data.values()):
            for k, cell_id in enumerate(sample.cell_keys):
                self.all_cells.append((sample.id, cell_id, k))

    def __len__(self):
        return len(self.all_cells)

    def __getitem__(self, index: int):
        sample_id, cell_id, cell_index = self.all_cells[index]
        sample = self.data[sample_id]
        content = sample.sources[cell_index] + "\n\n\n\n" + "\n\n".join(sample.sources)
        label = torch.tensor(sample.orders.index(cell_id) / len(sample.orders))
        inputs = self.tokenizer.encode_plus(
            content,
            None,
            add_special_tokens=True,
            padding="max_length",
            return_token_type_ids=True,
            truncation=True,
        )
        ids = torch.tensor(inputs["input_ids"]).long()
        mask = torch.tensor(inputs["attention_mask"]).long()
        return ids, mask, label, sample_id, cell_id


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


class NewDataset(RankDataset):
    def __getitem__(self, index):
        sample_id, cell_id, cell_index = self.all_markdown_cells[index]
        sample = self.data[sample_id]

        cell_content = sample.sources[cell_index]
        cell_input = self.tokenizer.encode_plus(
            cell_content,
            add_special_tokens=True,
            max_length=64,
            padding="max_length",
            return_token_type_ids=True,
            truncation=True,
        )

        context_content = [
            source for source in sample.sources[: sample.code_cell_count]
        ]
        context_inputs = self.tokenizer.batch_encode_plus(
            context_content,
            add_special_tokens=True,
            max_length=23,
            padding="max_length",
            truncation=True,
        )

        ids = cell_input["input_ids"]
        for x in context_inputs["input_ids"]:
            ids.extend(x[:-1])
        ids = ids[:512]
        if len(ids) != 512:
            ids = ids + [
                self.tokenizer.pad_token_id,
            ] * (512 - len(ids))
        ids = torch.LongTensor(ids)

        mask = cell_input["attention_mask"]
        for x in context_inputs["attention_mask"]:
            mask.extend(x[:-1])
        mask = mask[:512]
        if len(mask) != 512:
            mask = mask + [
                self.tokenizer.pad_token_id,
            ] * (512 - len(mask))
        mask = torch.LongTensor(mask)
        label = torch.tensor(sample.orders.index(cell_id) / len(sample.orders))
        return ids, mask, torch.tensor([label]), sample_id, cell_id


# class RankDatasetWithSplits(torch.utils.data.Dataset):
#     def __init__(
#         self,
#         data: Dict[str, Sample],
#         tokenizer: AutoTokenizer,
#         cell_token_size,
#         cell_stride,
#         context_cells_token_size,
#         context_stride,
#         max_len,
#         ordered_context_ratio,
#         split_len,
#         shuffle_markdowns=True,
#         with_code_cell=False,
#     ):
#         self.read_count = 0
#         self.data = data
#         self.split_len = split_len
#         self.ordered_context_ratio = ordered_context_ratio
#         self.context_cells_token_size = context_cells_token_size
#         self.context_stride = context_stride
#         self.shuffle_markdowns = shuffle_markdowns
#         self.cell_token_size = cell_token_size
#         self.cell_stride = cell_stride
#         self.max_len = max_len
#         self.all_cells = []

#         for sample in list(self.data.values()):
#             for cell_key in sample.cell_keys:
#                 if sample.cell_types[cell_key] == "markdown":
#                     self.all_cells.append((sample.id, cell_key))

#         self.tokenizer = tokenizer
#         self.hash_id = self.tokenizer.encode("#", add_special_tokens=False)[0]

#     def __len__(self):
#         return len(self.all_cells)

#     def __getitem__(self, index: int):
#         sample_id, cell_key = self.all_cells[index]
#         sample = self.data[sample_id]

#         anchor_encode = sample.cell_encodes[cell_key]
#         # 对于 anchor_encode，不要通过 stride 来过滤 # 字符（token 为 1001）
#         # 对于不同的 tokenizer 这里
#         anchor_encode = [
#             x
#             for k, x in enumerate(anchor_encode)
#             if ((k % self.cell_stride) == 0 or x == self.hash_id)
#             and k < (self.cell_token_size * self.cell_stride)
#         ]

#         input_ids = [self.tokenizer.cls_token_id] + anchor_encode

#         # 将 context 分为两种，按概率随机选择其中的一种进行训练
#         use_ordered_context = random.random() < self.ordered_context_ratio
#         if not use_ordered_context:
#             # 1. anchor + code cells
#             context_cell_keys = [key for key in sample.cell_keys if sample.cell_types[key] == "code"]
#         else:
#             # 2. anchor + ordered cells (这里不加 anchor 本身)
#             context_cell_keys = [key for key in sample.orders if key != cell_key]

#         available_splits_num = math.ceil(len(context_cell_keys) / self.split_len)
#         split_selected = random.sample(range(available_splits_num), k=1)[0]
#         context_cell_keys = context_cell_keys[split_selected*self.split_len:(split_selected+1)*self.split_len]

#         for context_cell_key in context_cell_keys:
#             cell_encode = sample.cell_encodes[context_cell_key]
#             context_encode = cell_encode[
#                 0 : self.context_cells_token_size
#                 * self.context_stride : self.context_stride
#             ]
#             input_ids += [self.tokenizer.sep_token_id] + context_encode

#         input_ids += [self.tokenizer.sep_token_id]
#         input_ids = input_ids[: self.max_len]
#         pad_len = self.max_len - len(input_ids)
#         input_ids += [self.tokenizer.pad_token_id] * pad_len
#         attention_mask = [1] * (self.max_len - pad_len) + [0] * pad_len

#         start_rank = sample.cell_ranks[context_cell_keys[0]]

#         rank = sample.cell_ranks[cell_key] + 1 - start_rank
#         rank_normed = rank / min(self.split_len, len(context_cell_keys))
#         in_split = float(rank_normed > 0 and rank_normed < 1)

#         return (
#             torch.tensor(input_ids).long(),
#             torch.tensor(attention_mask).long(),
#             torch.tensor([in_split, rank_normed]),
#             torch.tensor([sample.code_cell_count, sample.markdown_cell_count]),
#             sample_id,
#             cell_key,
#             split_selected
#         )


@dataclass
class SpecialTokenID:
    hash_id: int
    cls_token_id: int
    sep_token_id: int
    pad_token_id: int


class RankDatasetWithSplits(torch.utils.data.Dataset):
    def __init__(
        self,
        data: Dict[str, Sample],
        special_tokens: SpecialTokenID,
        cell_token_size,
        cell_stride,
        context_cells_token_size,
        context_stride,
        max_len,
        ordered_context_ratio,
        split_len,
        shuffle_markdowns=True,
        with_code_cell=False,
    ):
        self.read_count = 0
        self.data = data
        self.split_len = split_len
        self.ordered_context_ratio = ordered_context_ratio
        self.context_cells_token_size = context_cells_token_size
        self.context_stride = context_stride
        self.shuffle_markdowns = shuffle_markdowns
        self.cell_token_size = cell_token_size
        self.cell_stride = cell_stride
        self.max_len = max_len
        self.all_cells = []

        for sample in list(self.data.values()):
            context_cell_keys = [key for key in sample.cell_keys if sample.cell_types[key] == "code"]
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

        input_ids = [self.special_tokens.cls_token_id] + anchor_encode

        context_cell_keys = [key for key in sample.cell_keys if sample.cell_types[key] == "code"]
        context_cell_keys = context_cell_keys[split_id*self.split_len:(split_id+1)*self.split_len]

        for context_cell_key in context_cell_keys:
            cell_encode = sample.cell_encodes[context_cell_key]
            context_encode = cell_encode[
                0 : self.context_cells_token_size
                * self.context_stride : self.context_stride
            ]
            input_ids += [self.special_tokens.sep_token_id] + context_encode

        input_ids += [self.special_tokens.sep_token_id]
        input_ids = input_ids[: self.max_len]
        pad_len = self.max_len - len(input_ids)
        input_ids += [self.special_tokens.pad_token_id] * pad_len
        attention_mask = [1] * (self.max_len - pad_len) + [0] * pad_len

        # start_rank: 1 or 9 or 17 ...
        offset = sample.cell_ranks[context_cell_keys[0]]
        rank = sample.cell_ranks[cell_key] + 1 - offset
        rank_normed = rank / (self.split_len + 1)
        in_split = float(rank_normed > 0 and rank_normed < 1)

        return (
            torch.tensor(input_ids).long(),
            torch.tensor(attention_mask).long(),
            torch.tensor([in_split, rank_normed]),
            torch.tensor([sample.code_cell_count, sample.markdown_cell_count]),
            sample_id,
            cell_key,
            split_id
        )
