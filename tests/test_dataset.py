import pickle
from tqdm import tqdm
from torch.utils.data import DataLoader
from ai4code import datasets

from transformers import AutoModel, AutoTokenizer, DistilBertTokenizer
from ai4code.datasets import MixedDatasetWithSplits, RankDataset, LongFormerDataset
import time

data = pickle.load(open("/home/featurize/work/ai4code/data/v8/mini.v8.pkl", "rb"))


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
