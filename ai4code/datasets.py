import ast
from copy import copy
from ai4code import utils
import math
import pickle
import random
import numpy as np
from typing import Dict, List, Optional
import re
from markdown import markdown
from tokenizers import Tokenizer
import torch
from operator import itemgetter
import tokenize
import io
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


def links_to_word(text):
    return re.sub("https?:\/\/[^\s]+", " link ", text)


def no_char(text):
    text = re.sub(r"\s+[a-zA-Z]\s+", " ", text)
    text = re.sub(r"\^[a-zA-Z]\s+", " ", text)
    text = re.sub(r"\s+[a-zA-Z]$", " ", text)
    return text


def no_markdown_special(text):
    return re.sub(r"[\.\*\+\-\_\>\<\~\(\)\[\]]", " ", text)


def no_html_tags(text):
    return re.sub("<.*?>", " ", text)


def no_multi_spaces(text):
    return re.sub(r"\s+", " ", text, flags=re.I)


def lemmatize(text):
    tokens = text.split()
    tokens = [stemmer.lemmatize(word) for word in tokens]
    return " ".join(tokens)


def underscore_to_space(text: str):
    text = text.replace("_", " ")
    text = text.replace("-", " ")
    return text


def code_preprocess_v4(code):
    code = links_to_word(code)
    code = underscore_to_space(code)
    code = no_char(code)
    code = no_multi_spaces(code)
    code = lemmatize(code)
    return code


def code_preprocess_v7(code):
    # code 最小修改
    code = links_to_word(code)
    code = lemmatize(code)
    return code


def markdown_preprocess_v4(code):
    code = links_to_word(code)
    code = no_markdown_special(code)
    code = no_html_tags(code)
    code = no_char(code)
    code = no_multi_spaces(code)
    code = lemmatize(code)
    return code


def no_markdown_special_v2(text):
    # 保留顶格的 + - * >, 删除其他的
    text = text[0] + re.sub(r"(?<!\n)[\*\+\-\>]", " ", text[1:])

    # 删除 ( ) [ ] ` ~ |
    text = re.sub(r"\(\)\[\]\{\}\<\>\~\|\`\.", " ", text)
    return text


def markdown_preprocess_v6(code):
    """compare to v4:
    1. 不删除单个字符
    2. 保留部分顶格的特殊字符
    3. 先删除 html tags ，再删 markdown 记号
    """
    code = links_to_word(code)
    code = no_html_tags(code)
    code = no_markdown_special_v2(code)
    code = no_multi_spaces(code)
    code = lemmatize(code)
    return code


def code_preprocess_v5(code):
    """
    仅保留顶层代码，丢弃所有 nested 的代码
    """
    lines = code.split("\n")
    outputs = []
    for i, line in enumerate(lines):
        if not line.startswith(" "):
            outputs.append(line)

    return code_preprocess_v4("\n".join(outputs))


def code_preprocess_v6(code):
    """Tokenizer"""
    try:
        code_text = tokenize.generate_tokens(io.StringIO(code).readline)
        return " ".join(
            [tok.string for tok in code_text if tok.type == 1 or tok.type == 55]
        )
    except:
        # 有可能会失败，失败的话 fallback 到 code_preprocess_v4
        return code_preprocess_v4(code)


def preprocessor_v4(text, type):
    """follow mine mind version : )"""
    return dict(code=code_preprocess_v4, markdown=markdown_preprocess_v4)[type](text)


def preprocessor_v5(text, type):
    """代码仅保留最外层
    掉分！
    """
    return dict(code=code_preprocess_v5, markdown=markdown_preprocess_v4)[type](text)


def preprocessor_v6(text, type):
    return dict(code=code_preprocess_v4, markdown=markdown_preprocess_v6)[type](text)


def preprocessor_v7(text, type):
    return dict(code=code_preprocess_v6, markdown=markdown_preprocess_v6)[type](text)


def preprocessor_v8(text, type):
    return dict(code=code_preprocess_v7, markdown=markdown_preprocess_v6)[type](text)


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


@dataclass
class SpecialTokenID:
    hash_id: int
    cls_token_id: int
    sep_token_id: int
    pad_token_id: int
    unk_token_id: int


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
        with_code_cell=False,
        with_stride_id=False,
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
        self.with_stride_id = with_stride_id
        self.distil_context = distil_context

        if self.with_stride_id and self.context_stride != 2:
            raise RuntimeError("with stride id only support context id equals to 2")

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


class PairLMDataset(torch.utils.data.Dataset):
    def __init__(
        self, data: Dict[str, Sample], special_tokens: SpecialTokenID, max_len: int
    ):
        super().__init__()
        self.data = data
        self.special_tokens = special_tokens
        self.max_len = max_len
        self.all_cell_pairs = []
        for sample in list(self.data.values()):
            for cell_key, next_cell_key, previous_cell_key in zip(
                sample.orders[1:-1], sample.orders[2:], sample.orders[:-2]
            ):
                self.all_cell_pairs.append(
                    (sample.id, cell_key, next_cell_key, previous_cell_key)
                )

    def __getitem__(self, index):
        sample_id, anchor_key, next_key, previous_key = self.all_cell_pairs[index]
        sample = self.data[sample_id]

        anchor_encode = sample.cell_encodes[anchor_key]
        positive = random.random() > 0.5
        if positive:
            previous_encode = sample.cell_encodes[previous_key]
            next_encode = sample.cell_encodes[next_key]
        else:
            previous_encode, next_encode = random.sample(
                [
                    sample.cell_encodes[x]
                    for x in sample.cell_keys
                    if x not in [anchor_key, next_key, previous_key]
                ],
                k=2,
            )

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

    def __len__(self):
        return len(self.all_cell_pairs)


class MixedDatasetWithSplits(torch.utils.data.Dataset):
    def __init__(
        self,
        data: Dict[str, Sample],
        special_tokens: SpecialTokenID,
        anchor_size,
        max_len,
        split_len,
        distil_context,
        shuffle_markdowns=True,
        only_task_data=False,
    ):
        self.read_count = 0
        self.only_task_data = only_task_data
        self.data = data
        self.split_len = split_len
        self.anchor_size = anchor_size
        self.shuffle_markdowns = shuffle_markdowns
        self.max_len = max_len
        self.all_cells = []
        self.all_cell_pairs = []
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

    def get_task_data(self, index):
        sample_id, cell_key, split_id = self.all_cells[index]
        sample = self.data[sample_id]

        anchor_encode = sample.cell_encodes[cell_key][: self.anchor_size]

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
            context_encode = sample.cell_encodes[context_cell_key]
            cell_type = sample.cell_types[context_cell_key]
            context_encodes.append(context_encode)
            context_types.append(cell_type)

        context_encodes, indices = utils.adjust_sequences(
            context_encodes, self.max_len - len(anchor_encode) + self.split_len + 2
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
        input_ids, mask, targets, _, sample_id, cell_key, split_id  = self.get_task_data(index)
        if not self.only_task_data:
            lm_input_ids, lm_mask, lm_targets = self.get_pair_data(index)
            return input_ids, mask, lm_input_ids, lm_mask, targets, lm_targets, sample_id, cell_key, split_id
        return input_ids, mask, targets, sample_id, cell_key, split_id

