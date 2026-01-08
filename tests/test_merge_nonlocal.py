import numpy as np
import pytest
import MiLoMerge
import helpers

def test_naive():
    h1 = np.zeros(30)
    h1[0] = 5
    h1[3] = 1

    h2 = np.zeros(30)
    h2[1] = 5
    h2[4] = 2
    
    h3 = np.zeros(30)
    h3[2] = 5
    h3[5] = 3

    merger = MiLoMerge.MergerNonlocal(
        range(31), #bin edges
        h1,
        h2,
        h3
    )
    new_counts = merger.run(3)
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

def test_naive_comp_first():
    h1 = np.zeros(30)
    h1[0] = 5
    h1[3] = 1

    h2 = np.zeros(30)
    h2[1] = 5
    h2[4] = 2
    
    h3 = np.zeros(30)
    h3[2] = 5
    h3[5] = 3

    merger = MiLoMerge.MergerNonlocal(
        range(31), #bin edges
        h1,
        h2,
        h3,
        comp_to_first=True
    )
    new_counts = merger.run(2)
    
    nonzero_h1_mask = new_counts[0] == 6
    zero_h1_mask = new_counts[0] == 0
    assert new_counts[1][nonzero_h1_mask] == 0
    assert new_counts[1][zero_h1_mask] == 7

    assert new_counts[2][nonzero_h1_mask] == 0
    assert new_counts[2][zero_h1_mask] == 8

def test_2_case():
    h1 = np.array([3,4,5,6,8,9,10], dtype=np.float64)
    h2 = np.array([5,4,3,8,21,9,2], dtype=np.float64)
    assert len(h1) == len(h2)

    h1_BF, h2_BF = helpers.brute_force(h1, h2, 2)

    merger = MiLoMerge.MergerNonlocal(
        range(len(h1) + 1), #bin edges
        h1,
        h2
    )
    h1_MLM, h2_MLM = merger.run(2)
    assert np.array_equal(h1_BF.sort(), h1_MLM.sort())
    assert np.array_equal(h2_BF.sort(), h2_MLM.sort())

def test_invalid_binsize_1d():
    h1 = [1,2,3]
    h2 = [4,5,6]
    with pytest.raises(ValueError) as exc_info:
        MiLoMerge.MergerNonlocal(
            range(3),
            h1,
            h2
        )
    assert "Bin edges are of invalid size" in str(exc_info.value)
    assert exc_info.type is ValueError

def test_invalid_binsize_2d():
    h1 = np.array([
        [1,2,3],
        [4,5,6]
    ])
    h2 = np.array([
        [7,8,9],
        [10,11,12]
    ])
    with pytest.raises(ValueError) as exc_info:
        MiLoMerge.MergerNonlocal(
            (range(3), range(3)),
            h1,
            h2
        )
    assert "Bin edge for dimension 1" in str(exc_info.value)
    assert exc_info.type is ValueError

def test_invalid_number_of_bins():
    h1 = np.array([
        [1,2,3],
        [4,5,6]
    ])
    h2 = np.array([
        [7,8,9],
        [10,11,12]
    ])
    with pytest.raises(IndexError) as exc_info:
        MiLoMerge.MergerNonlocal(
            (range(3),),
            h1,
            h2
        )
    assert "No bin edges provided" in str(exc_info.value)
    assert exc_info.type is IndexError