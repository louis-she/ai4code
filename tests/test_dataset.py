import pickle
from tqdm import tqdm
from torch.utils.data import DataLoader

from transformers import AutoModel, AutoTokenizer, DistilBertTokenizer
from ai4code.datasets import RankDataset, LongFormerDataset
import time

fold4 = pickle.load(open("data/all_dict_data_4fold_mini.pkl", "rb"))


def test_dataset_getitem():
    fold10 = pickle.load(open("/home/featurize/work/ai4code/data/all_dict_data_10fold_mini.pkl", "rb"))
    dataset = RankDataset(
        fold10,
        tokenizer=DistilBertTokenizer.from_pretrained(
            "/home/featurize/distilbert-base-uncased/distilbert-base-uncased",
            do_lower_case=True,
            use_fast=True
        ),
        cell_token_size=64,
        cell_stride=1,
        context_cells_token_size=14,
        context_stride=2,
        max_len=512,
    )
    res = dataset[0]
    # loader = DataLoader(dataset, num_workers=8, batch_size=32, shuffle=True)
    # for _ in tqdm(loader):
    #     continue

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
