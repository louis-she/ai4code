import fire
import pandas as pd
from pathlib import Path
import pickle
import numpy as np
import json
from tqdm import tqdm
import hashlib
import os
from sklearn.model_selection import KFold
from ai4code.datasets import Sample
from ai4code import datasets
from transformers import AutoTokenizer
import multiprocessing


def get_ranks(cell_types, cell_orders, cell_keys):
    code_cells_num = len([cell_type for cell_type in cell_types if cell_type == "code"])
    code_cell_keys = cell_keys[:code_cells_num]

    code_bins = {(i, i+1): [] for i in range(code_cells_num + 1)}
    code_cell_orders = [cell_orders.index(cell_key) for cell_key in code_cell_keys]

    cell_ranks = {}
    if len(code_cell_orders) != 0:
        for k, (cell_type, cell_order, cell_key) in enumerate(zip(cell_types, cell_orders, cell_keys)):
            cell_order = cell_orders.index(cell_key)
            if cell_type == "code":
                cell_ranks[cell_key] = k + 1
                continue
            for i, code_cell_order in enumerate(code_cell_orders):
                if cell_order < code_cell_order:
                    code_bins[(i, i+1)].append((cell_order, cell_key))
                    break
            else:
                code_bins[(i+1, i+2)].append((cell_order, cell_key))
    else:
        for k, (cell_order, cell_key) in enumerate(zip(cell_orders, cell_keys)):
            code_bins[(0, 1)].append((cell_order, cell_key))

    for bins, values in code_bins.items():
        markdowns_sorted = sorted(values, key=lambda x: x[0])
        step = 1 / (len(markdowns_sorted) + 1)
        for j, (markdown_cell_order, markdown_cell_key) in enumerate(markdowns_sorted):
            cell_ranks[markdown_cell_key] = bins[0] + step * (j + 1)

    return cell_ranks


orders_dict, ancestors_dict, tokenizer, processor_suffix = None, None, None, None


def process(file):
    global orders_dict, ancestors_dict, tokenizer

    id = file.stem
    content = file.read_text()
    body = json.loads(content)

    content_len = len(content)
    markdown_count = 0
    code_count = 0

    cell_keys = list(body['cell_type'].keys())
    cell_lens = {}
    cell_ranks = {}
    cell_types = body['cell_type']
    cell_orders = orders_dict[id]

    for key in cell_keys:
        type = body['cell_type'][key]
        if type == "code":
            code_count += 1
        elif type == "markdown":
            markdown_count += 1
        else:
            print(f"Unknown type {type}, ignore")
        source = body['source'][key]
        cell_lens[key] = len(source)

    cell_ranks = get_ranks([cell_types[k] for k in cell_keys], cell_orders, cell_keys)
    cell_ranks_norm_factor = code_count + 1
    cell_ranks_normed = {cell_id: (rank / cell_ranks_norm_factor) for cell_id, rank in cell_ranks.items()}
    ancestor = ancestors_dict[id][0] if id in ancestors_dict and isinstance(ancestors_dict[id][0], str) else None
    parent = ancestors_dict[id][1] if id in ancestors_dict and isinstance(ancestors_dict[id][1], str) else None

    cell_encodes = {}
    for cell_key, value in body["source"].items():
        processor = getattr(datasets.preprocessor, processor_suffix)
        value = processor(value, cell_types[cell_key])
        cell_encodes[cell_key] = tokenizer.encode(value, add_special_tokens=False)

    code_ratio = code_count / (code_count + markdown_count)

    cell_lens_dict = {key: len(value) for key, value in body['source'].items()}

    cell_lens = np.array(list(cell_lens_dict.values()))
    percentile_cell_lens = [np.percentile(cell_lens, percentile) for percentile in range(0, 101, 10)]
    mean_cell_lens = cell_lens.mean()

    cell_lens = np.array([len(x) for x in body['source'].values()])

    sample = Sample(
        id=id,
        sources=body['source'],
        ancestor=ancestor,
        parent=parent,
        orders=orders_dict[id],
        markdown_cell_count=markdown_count,
        code_cell_count=code_count,
        content_len=content_len,
        cell_keys=list(cell_keys),
        cell_lens=cell_lens,
        cell_ranks=cell_ranks,
        cell_ranks_normed=cell_ranks_normed,
        cell_types=cell_types,
        cell_encodes=cell_encodes,
        code_ratio=code_ratio,
        percentile_cell_lens=percentile_cell_lens,
        mean_cell_lens=mean_cell_lens,
    )
    return sample


def main(
    suffix: str,
    pretrained_tokenizer: str,
    processor: str = None,
    root: str = "/home/featurize/data"
):
    global orders_dict, ancestors_dict, tokenizer, processor_suffix
    if processor is None:
        processor = suffix
    processor_suffix = f"preprocessor_{processor}"
    dataset_root = Path(root)
    try:
        ancestors = pd.read_csv(dataset_root / "train_ancestors.csv")
        ancestors_dict = {}
        for _, item in ancestors.iterrows():
            ancestors_dict[item.id] = (item.ancestor_id, item.parent_id)
    except:
        ancestors_dict = {}


    orders = pd.read_csv(dataset_root / "train_orders.csv")
    orders.fillna('', inplace=True)

    orders_dict = {}
    for _, item in orders.iterrows():
        orders_dict[item.id] = item.cell_order.split(" ")

    tokenizer = AutoTokenizer.from_pretrained(pretrained_tokenizer, do_lower_case=True, use_fast=True)

    with multiprocessing.Pool(processes=8) as pool:
        results = list(
            tqdm(
                pool.imap(process, (dataset_root / "train").glob("*.json")),
                total=len(list((dataset_root / "train").glob("*.json"))),
            )
        )

    all_data = {sample.id: sample for sample in results}

    if len(ancestors_dict) > 0:
        ancestors = list(set(map(lambda x: x.ancestor, all_data.values())))
        kf = KFold(n_splits=10, shuffle=True, random_state=777)
        for fold, (train_inds, val_inds) in enumerate(kf.split(ancestors)):
            val_ancestors = set(ancestors[ind] for ind in val_inds)
            for id, sample in all_data.items():
                if sample.ancestor in val_ancestors:
                    all_data[id].fold = fold
    else:
        for k, (id, sample) in enumerate(all_data.items()):
            all_data[id].fold = k // 10000

    os.makedirs(f"/home/featurize/work/ai4code/data/{suffix}", exist_ok=True)
    i = 0
    while True:
        fold_data = {k: v for k, v in list(all_data.items()) if v.fold == i}
        if len(fold_data) == 0:
            break
        pickle.dump(fold_data, open(f"/home/featurize/work/ai4code/data/{suffix}/{i}.pkl", "wb"))
        i += 1

    mini_data = {}
    for fold in range(10):
        mini_fold = [sample for sample in all_data.values() if sample.fold == fold][:100]
        for sample in mini_fold:
            mini_data[sample.id] = sample

    pickle.dump(mini_data, open(f"/home/featurize/work/ai4code/data/{suffix}/mini.{suffix}.pkl", "wb"))

    pickle.dump(dict(
        hash_id=tokenizer.encode("#", add_special_tokens=False)[0],
        cls_token_id=tokenizer.cls_token_id,
        sep_token_id=tokenizer.sep_token_id,
        pad_token_id=tokenizer.pad_token_id,
        unk_token_id=tokenizer.unk_token_id,
    ), open(f"/home/featurize/work/ai4code/data/{suffix}/special_tokens.pkl", "wb"))



if __name__ == "__main__":
    fire.Fire(main)
