import random
from collections import OrderedDict
from ignite.base.mixins import Serializable
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
