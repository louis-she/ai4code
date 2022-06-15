from ignite.base.mixins import Serializable


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
