"""The faster coloring must preserve the original greedy colors."""

import numpy as np
import pytest

from simpletree import BalancedTree
from simpletree.abstract_tree import AbstractTree
from simpletree.coloring import greedy_colors


@pytest.mark.parametrize("ndim", [2, 3])
def test_adaptive_tree_colors_match_original(ndim):
    rng = np.random.default_rng(730 + ndim)
    points = rng.random((2000, ndim))
    points[:1500] = 0.5 + 0.003 * rng.normal(size=(1500, ndim))
    tree = BalancedTree(points, leaf_size=16)
    tree.get_boxes_level_color(0, 0)
    colors, count = tree.box_colors.copy(), tree.num_colors
    AbstractTree._color_boxes(tree)
    np.testing.assert_array_equal(colors, tree.box_colors)
    assert count == tree.num_colors


def test_full_color_palette_and_chunk_boundary():
    # A 55-box clique followed by a chain spanning the 4096-row chunk boundary.
    neighbors = np.full((4097, 54), -1, dtype=int)
    for box in range(55):
        neighbors[box, :box] = np.arange(box)
    neighbors[55:, 0] = np.arange(54, 4096)
    colors, count = greedy_colors(neighbors[:, :27], neighbors[:, 27:])
    np.testing.assert_array_equal(colors, np.r_[np.arange(55), np.arange(4042) % 2])
    assert count == 55
