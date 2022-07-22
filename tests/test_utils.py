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


def test_advanced_subsequence():
    sequence = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
    quota = 9
    anchors = {10}
    new_sequence = utils.advanced_subsequence(sequence, anchors, quota)
    assert new_sequence == [1,2,3,9,10,11,12,19,20]

    quota = 10
    anchors = {10}
    new_sequence = utils.advanced_subsequence(sequence, anchors, quota)
    assert new_sequence == [1,2,3,9,10,11,12,13,19,20]

    quota = 6
    anchors = {3,4,5,6,7,8}
    new_sequence = utils.advanced_subsequence(sequence, anchors, quota)
    assert new_sequence == [1,4,5,6,7,8,9,20]

    quota = 13
    anchors = {8, 10}
    new_sequence = utils.advanced_subsequence(sequence, anchors, quota)
    print(new_sequence)
