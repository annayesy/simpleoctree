import numpy as np

from simpletree import BinaryTree, get_point_dist


def check_points_in_box(tree, box):
    inds = tree.get_box_inds(box)
    pts = tree.XX[inds]
    center = tree.get_box_center(box)
    length = tree.get_box_length(box)

    assert inds.shape[0] > 0
    assert np.all(np.abs(pts - center) <= 0.5 * length + 1e-12)


def check_children_partition(tree, box):
    children = tree.get_box_children(box)
    parent_inds = np.sort(tree.get_box_inds(box))
    child_inds = np.sort(np.concatenate([tree.get_box_inds(child) for child in children]))

    assert children.shape[0] == 2
    assert np.array_equal(parent_inds, child_inds)


def run_tree_checks(tree):
    for box in range(tree.nboxes):
        check_points_in_box(tree, box)

        children = tree.get_box_children(box)
        if tree.is_leaf(box):
            assert children.shape[0] == 0
            assert tree.get_box_inds(box).shape[0] <= tree.leaf_size
        else:
            check_children_partition(tree, box)
            for child in children:
                assert tree.get_box_parent(child) == box
                assert tree.get_box_level(child) == tree.get_box_level(box) + 1


def test_binary_tree_square():
    XX = get_point_dist(int(2e3), "square")
    tree = BinaryTree(XX, leaf_size=50)
    run_tree_checks(tree)


def test_binary_tree_cube():
    XX = get_point_dist(int(2e3), "cube")
    tree = BinaryTree(XX, leaf_size=50)
    run_tree_checks(tree)


def test_binary_tree_degenerate_points_stop():
    XX = np.ones((8, 2)) * 0.5
    tree = BinaryTree(XX, leaf_size=1)

    assert tree.nboxes == 1
    assert tree.nlevels == 1
    assert tree.is_leaf(0)
    assert np.array_equal(tree.get_box_inds(0), np.arange(8))
