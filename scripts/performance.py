import time
import torch
import fire
import os
import pickle
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import fire
import torch
from aim.pytorch_ignite import AimLogger
from ignite.contrib.handlers import ProgressBar
from ignite.engine import Engine, Events
from ignite.handlers import Checkpoint, DiskSaver, global_step_from_engine
from ignite.metrics import RunningAverage
from torch import optim
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AdamW, AutoModel, AutoTokenizer

from ai4code import datasets, metrics, models
from ai4code.utils import SerializableDict

LOG_DIR = Path("/home/featurize/ai4code")
DEVICE = torch.device("cuda")

os.environ["TOKENIZERS_PARALLELISM"] = "True"


def main(
    pretrained_path: str,
    anchor_size: int = 64,
    max_len: int = 256,
    split_len: int = 10,
):
    data = pickle.load(open("/home/featurize/work/ai4code/data/v8/mini.v8.pkl", "rb"))
    data = {k: v for k, v in list(data.items())[:100]}

    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_path, do_lower_case=True, use_fast=True
    )
    special_tokens = datasets.SpecialTokenID(
        hash_id=tokenizer.encode("#", add_special_tokens=False)[0],
        cls_token_id=tokenizer.cls_token_id,
        sep_token_id=tokenizer.sep_token_id,
        pad_token_id=tokenizer.pad_token_id,
        unk_token_id=tokenizer.unk_token_id,
    )
    dataset = datasets.MixedDatasetWithSplits(
        data,
        special_tokens,
        anchor_size=anchor_size,
        max_len=int(max_len),
        split_len=int(split_len),
        distil_context=None,
        only_task_data=True,
    )
    loader = DataLoader(dataset, num_workers=2, batch_size=32, shuffle=False)
    model = AutoModel.from_pretrained(pretrained_path)
    model = models.MultiHeadModel(pretrained_path)
    model.cuda()

    now = time.time()
    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader)):
            input_ids, mask, _, _, _, _, _ = batch
            _ = model(input_ids.cuda(), mask.cuda(), lm=False)
    spent = time.time() - now
    print(spent / 10.64, "小时")


fire.Fire(main)
