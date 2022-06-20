import hashlib
import json
import random
from collections import OrderedDict
from ignite.base.mixins import Serializable
from ai4code import datasets
import torch


class SerializableDict(Serializable):

    def __init__(self, state):
        self._state = state

    def state_dict(self):
        return self._state

    def load_state_dict(self, state):
        self._state = state

    def __getitem__(self, key):
        return self._state[key]


def adjust_sequences(sequences, max_len):
    length_of_seqs = [len(seq) for seq in sequences]
    total_len = sum(length_of_seqs)
    cut_off = total_len - max_len
    if cut_off <= 0:
        return sequences, length_of_seqs

    for _ in range(cut_off):
        max_index = length_of_seqs.index(max(length_of_seqs))
        length_of_seqs[max_index] -= 1
    sequences = [sequences[i][:l] for i, l in enumerate(length_of_seqs)]

    return sequences, length_of_seqs


def shuffle_batch(tensor):
    len_of_tensor = tensor.shape[0]
    shuffled_indices = random.sample(list(range(len_of_tensor)), len_of_tensor)
    unshuffled_indices = [shuffled_indices.index(k) for k, i in enumerate(shuffled_indices)]
    return shuffled_indices, unshuffled_indices


def get_ranks(cell_types, cell_orders, cell_keys):
    code_cells_num = len([cell_type for cell_type in cell_types if cell_type == "code"])
    code_cell_keys = cell_keys[:code_cells_num]

    code_bins = {(i, i+1): [] for i in range(code_cells_num + 1)}
    code_cell_orders = [cell_orders.index(cell_key) for cell_key in code_cell_keys]

    cell_ranks = {}
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

    for bins, values in code_bins.items():
        markdowns_sorted = sorted(values, key=lambda x: x[0])
        step = 1 / (len(markdowns_sorted) + 1)
        for j, (markdown_cell_order, markdown_cell_key) in enumerate(markdowns_sorted):
            cell_ranks[markdown_cell_key] = bins[0] + step * (j + 1)

    return cell_ranks


orders_dict, ancestors_dict, tokenizer, processor_suffix = None, None, None, None


def process(file):
    global orders_dict, ancestors_dict, tokenizer, processor_suffix

    id = file.stem
    content = file.read_text()
    body = json.loads(content)
    hash_object = hashlib.sha1(content.encode())

    content_sha1 = hash_object.hexdigest()
    content_len = len(content)
    markdown_count = 0
    code_count = 0

    cell_keys = list(body['cell_type'].keys())
    cell_sha1s = {}
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
        hash_object = hashlib.sha1(source.encode())
        cell_sha1s[key] = hash_object.hexdigest()
        cell_lens[key] = len(source)

    cell_ranks = get_ranks([cell_types[k] for k in cell_keys], cell_orders, cell_keys)
    cell_ranks_norm_factor = code_count + 1
    cell_ranks_normed = {cell_id: (rank / cell_ranks_norm_factor) for cell_id, rank in cell_ranks.items()}
    ancestor = ancestors_dict[id][0] if ancestors_dict is not None and isinstance(ancestors_dict[id][0], str) else None
    parent = ancestors_dict[id][1] if ancestors_dict is not None and isinstance(ancestors_dict[id][1], str) else None

    cell_encodes = {}
    for cell_key, value in body["source"].items():
        processor = getattr(datasets, processor_suffix)
        value = processor(value, cell_types[cell_key])
        cell_encodes[cell_key] = tokenizer.encode(value, add_special_tokens=False)

    sample = datasets.Sample(
        id=id,
        sources=body['source'],
        ancestor=ancestor,
        parent=parent,
        orders=orders_dict[id],
        markdown_cell_count=markdown_count,
        code_cell_count=code_count,
        content_sha1=content_sha1,
        content_len=content_len,
        cell_keys=list(cell_keys),
        cell_sha1s=cell_sha1s,
        cell_lens=cell_lens,
        cell_ranks=cell_ranks,
        cell_ranks_normed=cell_ranks_normed,
        cell_types=cell_types,
        cell_encodes=cell_encodes
    )
    return sample