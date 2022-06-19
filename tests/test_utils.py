from pytest import approx
import torch
from ai4code import utils


def test_adjust_sequences():
    mock_seqs = [
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        [1, 2, 3, 4, 5],
        [1, 2, 3, 4, 5, 6, 7],
    ]

    assert utils.adjust_sequences(mock_seqs, 9999)[1] == [10, 5, 7]
    assert utils.adjust_sequences(mock_seqs, 15)[1] == [5, 5, 5]

    assert utils.adjust_sequences(mock_seqs, 15)[0][0] == [1, 2, 3, 4, 5]
    assert utils.adjust_sequences(mock_seqs, 15)[0][1] == [1, 2, 3, 4, 5]
    assert utils.adjust_sequences(mock_seqs, 15)[0][2] == [1, 2, 3, 4, 5]

    assert utils.adjust_sequences(mock_seqs, 16)[1] == [5, 5, 6]
    assert utils.adjust_sequences(mock_seqs, 17)[1] == [6, 5, 6]
    assert utils.adjust_sequences(mock_seqs, 19)[1] == [7, 5, 7]
    assert utils.adjust_sequences(mock_seqs, 20)[1] == [8, 5, 7]


def test_shuffle_tensor():
    data = (torch.rand((8, 4)) * 100).long()
    shuffle_indices, unshuffled_indices = utils.shuffle_batch(data)
    shuffled = data[shuffle_indices]
    original = shuffled[unshuffled_indices]

    for i in range(data.shape[0]):
        assert data[i].sum().item() == original[i].sum().item()

    assert all( [ data[i].sum().item() == original[i].sum().item() for i in range(data.shape[0]) ] )
    assert any( [ data[i].sum().item() != shuffled[i].sum().item() for i in range(data.shape[0]) ] )

    assert data.sum().item(), approx(shuffled.sum().item())
