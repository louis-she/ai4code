from copy import copy
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
        shuffle_markdowns=True,
        with_code_cell=False,
    ):
        self.read_count = 0
        self.data = data
        self.context_cells_token_size = context_cells_token_size
        self.context_stride = context_stride
        self.shuffle_markdowns = shuffle_markdowns
        self.cell_token_size = cell_token_size
        self.cell_stride = cell_stride
        self.max_len = max_len
        self.all_cells = []

        for sample in list(self.data.values()):
            for cell_key in sample.cell_keys:
                if sample.cell_types[cell_key] == "markdown":
                    self.all_cells.append((sample.id, cell_key))

        self.tokenizer = tokenizer
        self.hash_id = self.tokenizer.encode("#", add_special_tokens=False)[0]

    def __len__(self):
        return len(self.all_cells)

    def __getitem__(self, index: int):
        sample_id, cell_key = self.all_cells[index]
        sample = self.data[sample_id]

        # cell_content = (
        #     code_preprocess(sample.sources[cell_index])
        #     if cell_type == "code"
        #     else markdown_preprocess(sample.sources[cell_index])
        # )
        # cell_input = self.tokenizer.encode_plus(
        #     cell_content,
        #     None,
        #     add_special_tokens=True,
        #     max_length=self.cell_token_size * self.cell_stride,
        #     padding="do_not_pad",
        #     return_token_type_ids=True,
        #     truncation=True,
        # )

        # context_content = [
        #     code_preprocess(source)
        #     if cell_type == "code"
        #     else markdown_preprocess(source)
        #     for (cell_type, source) in zip(sample.cell_types, sample.sources)
        # ]
        # context_inputs = self.tokenizer.batch_encode_plus(
        #     context_content,
        #     add_special_tokens=True,
        #     max_length=self.context_cells_token_size * self.context_stride,
        #     padding="do_not_pad",
        #     truncation=True,
        # )

        anchor_encode = sample.cell_encodes[cell_key]
        # 对于 anchor_encode，不要通过 stride 来过滤 # 字符（token 为 1001）
        # 对于不同的 tokenizer 这里
        anchor_encode = [
            x
            for k, x in enumerate(anchor_encode)
            if (k % self.cell_stride) == 0 or x == self.hash_id
        ]

        input_ids = [self.tokenizer.cls_token_id] + anchor_encode
        for context_cell_key, cell_encode in sample.cell_encodes.items():
            ctype = sample.cell_types[context_cell_key]
            if ctype == "markdown":
                continue
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

        # 对于 BertTokenizer 将句子处理为：
        # <cls> + markdown + <seq> + code + <seq> + ... + code + <seq>
        input_ids = cell_input["input_ids"][
            0 : self.cell_token_size * self.cell_stride : self.cell_stride
        ]
        # cell content mask
        length_of_cell_id = len(input_ids)
        cell_mask = torch.tensor(
            [1] * length_of_cell_id + [0] * (self.max_len - length_of_cell_id)
        ).long()
        for context_input_id in context_inputs["input_ids"]:
            context_id = context_input_id[
                0 : self.context_cells_token_size
                * self.context_stride : self.context_stride
            ]
            input_ids.extend(context_id[1:])

        input_ids = input_ids[: self.max_len]
        input_len = len(input_ids)
        pad_size = self.max_len - input_len
        input_ids = input_ids + [self.tokenizer.pad_token_id] * pad_size

        mask = torch.cat([torch.ones(input_len), torch.zeros(pad_size)]).long()
        ids = torch.tensor(input_ids).long()

        label = torch.tensor(sample.cell_ranks_normed[cell_index]).float()

        return (
            ids,
            mask,
            cell_mask,
            torch.tensor([sample.code_cell_count, sample.markdown_cell_count]),
            torch.tensor([label]),
            sample_id,
            cell_id,
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
