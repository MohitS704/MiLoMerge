import numpy as np
import pytest
import MiLoMerge
import helpers

def test_naive():
    h1 = np.zeros(30)
    h1[0] = 5
    h1[1] = 1

    h2 = np.zeros(30)
    h2[2] = 5
    h2[3] = 2
    
    h3 = np.zeros(30)
    h3[4] = 5
    h3[5] = 3

    merger = MiLoMerge.MergerLocal(
        range(31), #bin edges
        h1,
        h2,
        h3
    )
    new_edges, new_counts = merger.run(3, return_counts=True)
    assert np.count_nonzero(new_counts[0] == 6) == 1
    assert np.count_nonzero(new_counts[0] == 0) == 2
    assert np.all(new_counts[0][new_counts[1] == 7] == 0)
    assert np.all(new_counts[0][new_counts[2] == 8] == 0)

    assert np.count_nonzero(new_counts[1] == 7) == 1
    assert np.count_nonzero(new_counts[1] == 0) == 2
    assert np.all(new_counts[1][new_counts[0] == 6] == 0)
    assert np.all(new_counts[1][new_counts[2] == 8] == 0)

    assert np.count_nonzero(new_counts[2] == 8) == 1
    assert np.count_nonzero(new_counts[2] == 0) == 2
    assert np.all(new_counts[2][new_counts[0] == 6] == 0)
    assert np.all(new_counts[2][new_counts[1] == 7] == 0)

    assert np.array_equal(new_edges, [0,2,4,30])

def test_naive_comp_first():
    h1 = np.zeros(30)
    h1[0] = 5
    h1[1] = 1

    h2 = np.zeros(30)
    h2[2] = 5
    h2[3] = 2
    
    h3 = np.zeros(30)
    h3[4] = 5
    h3[5] = 3

    merger = MiLoMerge.MergerLocal(
        range(31), #bin edges
        h1,
        h2,
        h3,
        comp_to_first=True
    )
    new_edges, new_counts = merger.run(2, return_counts=True)
    
    nonzero_h1_mask = new_counts[0] == 6
    zero_h1_mask = new_counts[0] == 0
    assert new_counts[1][nonzero_h1_mask] == 0
    assert new_counts[1][zero_h1_mask] == 7

    assert new_counts[2][nonzero_h1_mask] == 0
    assert new_counts[2][zero_h1_mask] == 8

    assert np.array_equal(new_edges, [0,2,30])


def test_2_case():
    h1 = np.array([3,4,5,6,8,9,10], dtype=np.float64)
    h2 = np.array([5,4,3,8,21,9,2], dtype=np.float64)
    assert len(h1) == len(h2)

    merger = MiLoMerge.MergerLocal(
        range(len(h1) + 1), #bin edges
        h1,
        h2
    )

    _, (h1_MLM, h2_MLM) = merger.run(2, return_counts=True)

    h1_BF, h2_BF = helpers.brute_force(h1, h2, 2, local=True)

    assert np.array_equal(h1_BF.sort(), h1_MLM.sort())
    assert np.array_equal(h2_BF.sort(), h2_MLM.sort())

def test_invalid_binsize_1d():
    h1 = [1,2,3]
    h2 = [4,5,6]
    with pytest.raises(ValueError) as exc_info:
        MiLoMerge.MergerLocal(
            range(3),
            h1,
            h2
        )
    assert "len(counts) =" in str(exc_info.value)
    assert exc_info.type is ValueError

# if __name__ == "__main__":
#     test_2_case()