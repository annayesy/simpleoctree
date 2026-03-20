import numpy as np

from simpletree import BalancedTree, get_point_dist, morton
import simpletree.morton_tree as morton_tree


def old_unbalanced_tree(XX, c0, L0, max_bs):
    N = XX.shape[0]
    ndim = XX.shape[-1]
    if ndim == 2:
        max_level = morton.MAX_LEVEL2
    else:
        max_level = morton.MAX_LEVEL3

    active_bool = np.ones(N, dtype=bool)
    leaf_keys = np.zeros(N, int)
    acc_leaves = 0

    for lev in range(max_level):
        if np.sum(active_bool) == 0:
            break

        active_inds = np.where(active_bool)[0]
        anchors_points = morton.points_to_anchors(XX[active_inds], lev, c0, L0, ndim=ndim)
        keys_points = morton.anchors_to_keys(anchors_points, ndim=ndim)
        sorted_perm = np.argsort(keys_points)

        unique, indices, counts = np.unique(keys_points[sorted_perm], return_counts=True, return_index=True)
        indices = np.concatenate((indices, np.array([np.sum(active_bool)])))

        nleaf_lev = 0
        for j in range(unique.shape[0]):
            if counts[j] <= max_bs:
                tmp_inds = sorted_perm[indices[j] : indices[j + 1]]
                active_bool[active_inds[tmp_inds]] = 0
                nleaf_lev += 1

        leaf_keys[acc_leaves : acc_leaves + nleaf_lev] = unique[counts <= max_bs]
        acc_leaves += nleaf_lev

    leaf_keys = leaf_keys[:acc_leaves]
    return leaf_keys


def old_balance_tree(XX, c0, L0, leaf_keys):
    ndim = XX.shape[-1]

    depth = np.max(morton.get_level(leaf_keys))
    leaf_keys = set(leaf_keys)
    for lev in range(depth, 1, -1):
        tmp = np.array(list(leaf_keys), dtype=int)
        keys_lev = tmp[morton.get_level(tmp) == lev]
        if keys_lev.shape[0] == 0:
            continue

        colleagues_vec = morton.get_colleagues_vec(keys_lev, ndim=ndim)
        colleagues = colleagues_vec[colleagues_vec > 0]

        parents = morton.get_parent(colleagues, ndim=ndim)
        sibs = morton.get_siblings_vec(parents, ndim=ndim)

        for j in range(colleagues.shape[0]):
            if (colleagues[j] not in leaf_keys) and (parents[j] not in leaf_keys):
                leaf_keys.update(sibs[j])

    for lev in range(depth, 0, -1):
        tmp = np.array(list(leaf_keys), dtype=int)
        keys_lev = tmp[morton.get_level(tmp) == lev]
        if keys_lev.shape[0] == 0:
            continue

        anchors_points = morton.points_to_anchors(XX, lev, c0, L0, ndim=ndim)
        keys_points = set(morton.anchors_to_keys(anchors_points, ndim=ndim))

        ancestors = morton.get_ancestors_vec(keys_lev, ndim=ndim)
        leaf_keys.difference_update(set(ancestors.flatten()))

        diff = set(keys_lev).difference(keys_points)
        leaf_keys -= diff

    return np.sort(np.array(list(leaf_keys), dtype=int))


def old_complete_tree(leaf_keys, ndim):
    tree_keys = set(leaf_keys.copy())
    for leaf in leaf_keys:
        tree_keys.update(set(morton.get_ancestors(leaf, ndim=ndim)))
    return np.sort(np.array(list(tree_keys), dtype=int))


def check_two_to_one_balance(tree):
    leaves = tree.get_leaves()
    centers = np.array([tree.get_box_center(box) for box in leaves])
    lengths = np.array([tree.get_box_length(box) for box in leaves])
    levels = np.array([tree.get_box_level(box) for box in leaves])

    for j in range(leaves.shape[0]):
        delta = np.abs(centers[j + 1 :] - centers[j])
        thresh = 0.5 * (lengths[j + 1 :, None] + lengths[j]) + 1e-12
        adjacent = np.all(delta <= thresh, axis=1)
        if np.any(adjacent):
            assert np.all(np.abs(levels[j + 1 :][adjacent] - levels[j]) <= 1)


def test_balance_property_square():
    XX = get_point_dist(int(3e3), "square")
    tree = BalancedTree(XX, leaf_size=40)
    check_two_to_one_balance(tree)


def test_balance_property_curvy_annulus():
    XX = get_point_dist(int(3e3), "curvy_annulus")
    tree = BalancedTree(XX, leaf_size=40)
    check_two_to_one_balance(tree)


def test_balance_property_cube():
    XX = get_point_dist(int(2e3), "cube")
    tree = BalancedTree(XX, leaf_size=40)
    check_two_to_one_balance(tree)


def test_balanced_tree_matches_reference_small_random():
    rng = np.random.RandomState(0)

    for ndim in [2, 3]:
        XX = rng.rand(200, ndim)
        c0, L0 = morton.get_root_params(XX)

        unbalanced = old_unbalanced_tree(XX, c0, L0, max_bs=12)
        ref_leaf = old_balance_tree(XX, c0, L0, unbalanced.copy())
        ref_tree = old_complete_tree(ref_leaf, ndim=ndim)

        tree = BalancedTree(XX, leaf_size=12, c0=c0, L0=L0)

        assert np.array_equal(tree.leaf_keys, ref_leaf)
        assert np.array_equal(tree.tree_keys, ref_tree)


def test_balanced_tree_explicit_root_params_match_auto():
    rng = np.random.RandomState(1)

    for ndim in [2, 3]:
        XX = rng.rand(300, ndim)
        c0, L0 = morton.get_root_params(XX)

        tree_auto = BalancedTree(XX, leaf_size=20)
        tree_manual = BalancedTree(XX, leaf_size=20, c0=c0, L0=L0)

        assert np.array_equal(tree_auto.leaf_keys, tree_manual.leaf_keys)
        assert np.array_equal(tree_auto.tree_keys, tree_manual.tree_keys)


def test_balanced_tree_partial_root_params():
    rng = np.random.RandomState(3)

    for ndim in [2, 3]:
        XX = rng.rand(300, ndim)
        c0, L0 = morton.get_root_params(XX)

        tree_auto = BalancedTree(XX, leaf_size=20)
        tree_c0 = BalancedTree(XX, leaf_size=20, c0=c0)
        tree_L0 = BalancedTree(XX, leaf_size=20, L0=L0)

        assert np.array_equal(tree_auto.leaf_keys, tree_c0.leaf_keys)
        assert np.array_equal(tree_auto.tree_keys, tree_c0.tree_keys)
        assert np.array_equal(tree_auto.leaf_keys, tree_L0.leaf_keys)
        assert np.array_equal(tree_auto.tree_keys, tree_L0.tree_keys)


def test_balanced_tree_lazy_neighbors_and_colors():
    XX = get_point_dist(int(2e3), "curvy_annulus")
    tree = BalancedTree(XX, leaf_size=50)

    assert tree.Clist_tmp is None
    assert tree.Llist_tmp is None
    assert not hasattr(tree, "box_colors")

    neigh = tree.get_box_neighbors(tree.level_sep[2])
    assert neigh.shape[0] > 0
    assert tree.Clist_tmp is not None
    assert tree.Llist_tmp is not None

    boxes = tree.get_boxes_level_color(0, 0)
    assert boxes.shape[0] > 0
    assert hasattr(tree, "box_colors")


def test_balanced_tree_small_point_clouds():
    rng = np.random.RandomState(2)
    for XX in [rng.rand(2, 2), rng.rand(3, 2), rng.rand(2, 3), rng.rand(3, 3)]:
        tree = BalancedTree(XX, leaf_size=1)
        leaf_boxes = tree.get_leaves()

        assert leaf_boxes.shape[0] >= 1
        all_inds = np.sort(np.concatenate([tree.get_box_inds(box) for box in leaf_boxes]))
        assert np.array_equal(all_inds, np.arange(XX.shape[0]))


def test_balanced_tree_identical_points_supported():
    for XX in [np.ones((8, 2)) * 0.5, np.ones((8, 3)) * 0.5]:
        tree = BalancedTree(XX, leaf_size=1)
        leaf_boxes = tree.get_leaves()

        assert leaf_boxes.shape[0] == 1
        assert np.array_equal(tree.get_box_inds(leaf_boxes[0]), np.arange(XX.shape[0]))
