from ai4code.metrics import kendall_tau


def test_kendall_tau_perfect():
    res = kendall_tau(
        [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]],
        [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]],
    )
    assert res == 1


def test_kendall_tau_reverse_perfect():
    res = kendall_tau(
        [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]],
        [[5, 4, 3, 2, 1], [10, 9, 8, 7, 6]],
    )

    assert res == -1


def test_kendall_tau_middle():
    res = kendall_tau(
        [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]],
        [[1, 2, 4, 3, 5], [9, 7, 8, 6, 10]],
    )
    assert res < 1 and res > -1
