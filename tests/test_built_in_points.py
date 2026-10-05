import numpy as np
import pytest

from simpletree import BalancedTree, get_uniform_level_2d


def test_uniform_level_2d_builds_single_tree_level():
    occupancy = 6
    box_length = 0.25

    XX = get_uniform_level_2d(box_length=box_length, occupancy=occupancy)
    tree = BalancedTree(XX, leaf_size=occupancy, c0=np.array([0.5, 0.5]), L0=1.0)

    leaves = tree.get_leaves()
    levels = np.array([tree.get_box_level(box) for box in leaves], dtype=int)
    lengths = np.array([tree.get_box_length(box) for box in leaves], dtype=float)
    counts = np.array([tree.get_box_inds(box).shape[0] for box in leaves], dtype=int)

    assert XX.shape == (occupancy * 16, 2)
    assert np.all(levels == 2)
    assert np.allclose(lengths, box_length)
    assert np.all(counts == occupancy)


def test_uniform_level_2d_rejects_non_quadtree_box_length():
    with pytest.raises(ValueError, match="power of two"):
        get_uniform_level_2d(box_length=1.0 / 3.0, occupancy=4)
