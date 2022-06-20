import json
from os import unlink
import pickle
import shutil
import tempfile
from tqdm import tqdm
from torch.utils.data import DataLoader
from ai4code import datasets, utils
from pathlib import Path

from transformers import AutoModel, AutoTokenizer, BertTokenizer, DistilBertTokenizer
from ai4code.datasets import MixedDatasetWithSplits, RankDataset, LongFormerDataset
import time

# mocking data
data = {
    "source": {
        # code
        "0000": "code " * 10, # 3642
        "0001": "code " * 15,
        "0002": "code " * 5,
        "0003": "code " * 9,
        "0004": "code " * 5,
        "0005": "code " * 3,
        "0006": "code " * 100,

        # markdown
        "0007": "mark " * 100, # 2928
        "0008": "mark " * 5,
        "0009": "mark " * 2,
        "0010": "mark " * 3,
        "0011": "mark " * 6,
        "0012": "mark " * 8,
    },
    "cell_type": {
        # code
        "0000": "code",
        "0001": "code",
        "0002": "code",
        "0003": "code",
        "0004": "code",
        "0005": "code",
        "0006": "code",

        # markdown
        "0007": "markdown",
        "0008": "markdown",
        "0009": "markdown",
        "0010": "markdown",
        "0011": "markdown",
        "0012": "markdown",
    }
}

orders_dict = {"000001": ["0007", "0000", "0001", "0002", "0008", "0009", "0010", "0003", "0004", "0011", "0005", "0006", "0012"]}

mock_filename = "/tmp/000001.json"
tokenizer = AutoTokenizer.from_pretrained("/home/featurize/distilbert-base-uncased/distilbert-base-uncased")

def load_processed_data():
    file = open(mock_filename, "w")
    json.dump(data, file)
    file.close()

    utils.ancestors_dict = None
    utils.orders_dict = orders_dict
    utils.tokenizer = tokenizer
    utils.processor_suffix = "preprocessor_v8"

    return {"000001": utils.process(Path(mock_filename))}


def test_mixed_dataset():
    mock_data = load_processed_data()
    sample = list(mock_data.values())[0]
    split_len = 3
    max_len = 56
    anchor_size = 30

    tokens = datasets.SpecialTokenID(
        hash_id=tokenizer.encode("#", add_special_tokens=False)[0],
        cls_token_id=tokenizer.cls_token_id,
        sep_token_id=tokenizer.sep_token_id,
        pad_token_id=tokenizer.pad_token_id,
        unk_token_id=tokenizer.unk_token_id,
    )
    tokens.mark_id = tokenizer.encode("mark", add_special_tokens=False)[0]
    tokens.code_id = tokenizer.encode("code", add_special_tokens=False)[0]

    dataset = datasets.MixedDatasetWithSplits(
        mock_data,
        tokens,
        anchor_size=anchor_size,
        max_len=max_len,
        split_len=split_len,
        distil_context=None,
    )

    assert len(dataset) == (sample.markdown_cell_count * split_len)

    input_ids, mask, lm_input_ids, lm_mask, targets, lm_targets, sample_id, cell_key, split_id = dataset[1]
    print(input_ids)


def test_dataset_getitem():
    tokenizer = DistilBertTokenizer.from_pretrained(
        "/home/featurize/distilbert-base-uncased/distilbert-base-uncased",
        do_lower_case=True,
        use_fast=True
    )
    special_tokens = datasets.SpecialTokenID(
        hash_id=tokenizer.encode("#", add_special_tokens=False)[0],
        cls_token_id=tokenizer.cls_token_id,
        sep_token_id=tokenizer.sep_token_id,
        pad_token_id=tokenizer.pad_token_id,
        unk_token_id=tokenizer.unk_token_id,
    )

    dataset = MixedDatasetWithSplits(
        data,
        special_tokens=special_tokens,
        anchor_size=64,
        split_len=10,
        max_len=256,
        distil_context=None,
    )

    input_ids, mask, lm_input_ids, lm_mask, targets, lm_targets, sample_id, cell_key, split_id = dataset[0]

    assert True


def test_tokenize():
    tokenizer = AutoTokenizer.from_pretrained("/home/featurize/codebert-base")
    tokenizer_base = DistilBertTokenizer.from_pretrained(
        "/home/featurize/distilbert-base-uncased/distilbert-base-uncased", do_lower_case=True
    )
    model = AutoModel.from_pretrained("/home/featurize/codebert-base")

    # tokenizer.encode_plus( "def a(): print \"xxxx\"",)
    text = "hello world nice good"
    token = tokenizer.tokenize(text)
    token_base = tokenizer_base.tokenize(text)

    assert True
