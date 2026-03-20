import numpy as np

from simpletree.abstract_tree import AbstractTree


def _get_root_params(XX):
    min_bound = np.min(XX, axis=0)
    max_bound = np.max(XX, axis=0)

    center = 0.5 * (min_bound + max_bound)
    length = max_bound - min_bound

    # Keep a nonzero box size even for degenerate point clouds.
    pad = np.maximum(1e-3, 1e-3 * np.maximum(length, 1.0))
    return center, length + pad


class BinaryTree(AbstractTree):

    def __init__(self, XX, leaf_size=100, c0=None, L0=None):

        XX = np.asarray(XX, dtype=float)
        assert XX.ndim == 2
        assert XX.shape[-1] in (2, 3)
        if XX.shape[0] == 0:
            raise ValueError("XX must contain at least one point")

        self.XX = XX
        self.ndim = XX.shape[-1]
        self.leaf_size = leaf_size

        if c0 is None or L0 is None:
            center_auto, length_auto = _get_root_params(XX)
            if c0 is None:
                c0 = center_auto
            if L0 is None:
                L0 = length_auto

        self.root_center = np.asarray(c0, dtype=float)
        if self.root_center.shape != (self.ndim,):
            raise ValueError("c0 must have shape (ndim,)")

        root_length = np.asarray(L0, dtype=float)
        if root_length.ndim == 0:
            root_length = np.full(self.ndim, float(root_length))
        if root_length.shape != (self.ndim,):
            raise ValueError("L0 must be a scalar or have shape (ndim,)")
        if np.any(root_length <= 0):
            raise ValueError("L0 must be positive")
        self.root_length = root_length

        self._build_tree()

    def _build_tree(self):

        centers = [self.root_center]
        lengths = [self.root_length]
        point_lists = [np.arange(self.XX.shape[0], dtype=int)]
        levels = [0]
        parents = [-1]
        children = [[-1, -1]]
        isleaf = [False]

        box = 0
        while box < len(point_lists):
            point_inds = point_lists[box]
            level = levels[box]
            center = centers[box]
            length = lengths[box]

            split = self._split_box(point_inds, center, length, level)
            if split is None:
                isleaf[box] = True
                box += 1
                continue

            children[box] = []
            for child_inds, child_center, child_length in split:
                child_box = len(point_lists)
                point_lists.append(child_inds)
                centers.append(child_center)
                lengths.append(child_length)
                levels.append(level + 1)
                parents.append(box)
                children.append([-1, -1])
                isleaf.append(False)
                children[box].append(child_box)
            box += 1

        self.center_list = np.asarray(centers, dtype=float)
        self.length_list = np.asarray(lengths, dtype=float)
        self.level_list = np.asarray(levels, dtype=int)
        self.parent_list = np.asarray(parents, dtype=int)
        self.child_list = np.asarray(children, dtype=int)
        self.isleaf_list = np.asarray(isleaf, dtype=bool)
        self.point_lists = point_lists

        _, level_sep = np.unique(self.level_list, return_index=True)
        self.level_sep = np.concatenate((level_sep, np.array([len(point_lists)], dtype=int)))

    def _split_box(self, point_inds, center, length, level):
        if point_inds.shape[0] <= self.leaf_size:
            return None

        for offset in range(self.ndim):
            axis = (level + offset) % self.ndim
            coords = self.XX[point_inds, axis]
            left_mask = coords < center[axis]
            if np.all(left_mask) or np.all(~left_mask):
                continue

            child_length = length.copy()
            child_length[axis] *= 0.5

            left_center = center.copy()
            right_center = center.copy()
            shift = 0.25 * length[axis]
            left_center[axis] -= shift
            right_center[axis] += shift

            left_inds = point_inds[left_mask]
            right_inds = point_inds[~left_mask]
            return [
                (left_inds, left_center, child_length.copy()),
                (right_inds, right_center, child_length.copy()),
            ]

        return None

    @property
    def N(self):
        return self.XX.shape[0]

    @property
    def nboxes(self):
        return self.center_list.shape[0]

    @property
    def nlevels(self):
        return self.level_sep.shape[0] - 1

    def get_box_center(self, box):
        return self.center_list[box]

    def get_box_length(self, box):
        return self.length_list[box]

    def get_box_level(self, box):
        return self.level_list[box]

    def get_box_children(self, box):
        tmp = self.child_list[box]
        return tmp[tmp >= 0]

    def get_box_parent(self, box):
        return self.parent_list[box]

    def get_box_inds(self, box):
        return self.point_lists[box]

    def get_boxes_level(self, level):
        assert level < self.nlevels
        return np.arange(self.level_sep[level], self.level_sep[level + 1])

    def get_leaves(self):
        return np.where(self.isleaf_list)[0]

    def is_leaf(self, box):
        return self.isleaf_list[box]

    def get_box_coarse_neigh(self, box):
        return np.array([], dtype=int)

    def get_box_colleague_neigh(self, box):
        return np.array([box], dtype=int)
