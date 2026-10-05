"""Regression checks for stable point ranges and cached neighbors."""

from itertools import product

import numpy as np
import pytest

from simpletree import BalancedTree, morton


@pytest.mark.parametrize("ndim", [2, 3])
def test_stable_point_views_and_leaf_layout(ndim):
    axis = [0.0, 0.25, 0.5, 0.75, np.nextafter(1.0, 0.0)]
    points = np.asarray(list(product(axis, repeat=ndim)))
    points = np.concatenate((points[::-1], points[::7]))
    tree = BalancedTree(points, leaf_size=3, c0=np.full(ndim, 0.5), L0=1.0)
    depth = (morton.MAX_LEVEL2 if ndim == 2 else morton.MAX_LEVEL3) - 1
    anchors = morton.points_to_anchors(points, depth, tree.c0, tree.L0, ndim=ndim)
    order = np.argsort(morton.anchors_to_keys(anchors, ndim=ndim), kind="stable")
    for box in range(tree.nboxes):
        center, length = tree.get_box_center(box), tree.get_box_length(box)
        inside = np.all((points >= center - length / 2) &
                        (points < center + length / 2), axis=1)
        indices = tree.get_box_inds(box)
        np.testing.assert_array_equal(indices, order[inside[order]])
        assert np.shares_memory(indices, tree.point_order)
        assert not indices.flags.writeable
    assert tree._leaf_layout is None
    for entry, box in enumerate(tree.get_leaves()):
        start, end = tree.leaf_sep[entry:entry + 2]
        np.testing.assert_array_equal(tree.perm_leaf[start:end], tree.get_box_inds(box))
    np.testing.assert_array_equal(np.sort(tree.perm_leaf), np.arange(tree.N))


@pytest.mark.parametrize("ndim", [2, 3])
def test_fine_neighbors_and_rebuilding(ndim):
    rng = np.random.default_rng(627 + ndim)
    points = rng.random((200, ndim))
    points[:150] = 0.5 + 0.003 * rng.normal(size=(150, ndim))
    tree = BalancedTree(points, leaf_size=5)
    tree._ensure_neighbor_lists()
    assert np.any(tree.Llist_unique_tmp >= 0)
    for box in range(tree.nboxes):
        expected = np.flatnonzero(np.any(tree.Llist_unique_tmp == box, axis=1))
        np.testing.assert_array_equal(tree.get_box_fine_neigh(box), expected)
    tree.get_boxes_level_color(0, 0)
    tree._form_tree(tree.N)
    assert tree.nboxes == 1 and not hasattr(tree, "box_colors")
    assert tree.get_box_fine_neigh(0).size == 0
    np.testing.assert_array_equal(tree.get_boxes_level_color(0, 0), [0])
    np.testing.assert_array_equal(tree.leaf_sep, [0, tree.N])


def test_explicit_root_excludes_upper_boundary():
    with pytest.raises(ValueError):
        BalancedTree([[1.0, 0.5]], c0=[0.5, 0.5], L0=1.0)
