import numpy as np
from simpletree import morton
from simpletree.abstract_tree import AbstractTree
from simpletree.coloring import greedy_colors

##############################################################################
######### Utilities for searching in sorted 1d numpy arrays ##################

def in_sorted_vec(sorted_list,entries):
    return findin_sorted_vec(sorted_list,entries) >= 0

def in_sorted(sorted_list,entry):
    return in_sorted_vec(sorted_list,np.array([entry],dtype=int))[0]

def findin_sorted_vec(sorted_list,entries):
    entries = np.asarray(entries)
    inds = np.searchsorted(sorted_list,entries)
    result = np.full(entries.shape,-1,dtype=int)
    valid = inds < len(sorted_list)
    match = np.zeros(entries.shape,dtype=bool)
    match[valid] = sorted_list[inds[valid]] == entries[valid]
    result[match] = inds[match]
    return result

def findin_sorted(sorted_list,entry):
    return findin_sorted_vec(sorted_list,np.array([entry],dtype=int))[0]

##########################################################################

def _max_level(ndim):
    if (ndim == 2):
        return morton.MAX_LEVEL2
    else:
        return morton.MAX_LEVEL3

class _PointIndex:
    """One finest-level Morton sort shared by refinement and every box query."""

    def __init__(self,XX,c0,L0):
        self.ndim = XX.shape[1]
        self.level = _max_level(self.ndim) - 1
        if L0 / (1 << self.level) < np.finfo(float).tiny:
            raise ValueError("L0 is too small to resolve the finest Morton cell")
        anchors = morton.points_to_anchors(XX,self.level,c0,L0,ndim=self.ndim)
        # Inputs have been checked against the half-open root. Subtraction can
        # round an interior point onto the upper face; keep that point inside.
        anchors[:,:self.ndim] = np.clip(anchors[:,:self.ndim],0,(1 << self.level)-1)
        raw_keys = morton.clear_level(morton.anchors_to_keys(anchors,ndim=self.ndim))
        self.order = np.argsort(raw_keys,kind='stable')
        self.raw_keys = raw_keys[self.order]
        self.order.flags.writeable = False

    def ranges(self,keys):
        """Return half-open point intervals for keys, including mixed levels."""
        shift = self.ndim * (self.level - morton.get_level(keys))
        raw = morton.clear_level(keys)
        lower = raw << shift
        upper = (raw + 1) << shift
        return (np.searchsorted(self.raw_keys,lower),
                np.searchsorted(self.raw_keys,upper))


def _unbalanced_tree(XX,c0,L0,max_bs,point_index=None):
    """Refine occupied octants using point intervals, stopping at the depth cap."""
    if point_index is None:
        point_index = _PointIndex(XX,c0,L0)
    ndim = point_index.ndim
    keys = np.array([0],dtype=int)
    starts = np.array([0],dtype=int)
    ends = np.array([XX.shape[0]],dtype=int)
    leaves = []
    child_offsets = np.arange(1 << ndim,dtype=int)

    for lev in range(point_index.level + 1):
        is_leaf = (ends - starts <= max_bs) | (lev == point_index.level)
        leaves.append(keys[is_leaf])
        parents = keys[~is_leaf]
        if parents.size == 0:
            break
        child_raw = (morton.clear_level(parents) << ndim)[:,None] | child_offsets
        keys = morton.set_level(child_raw.ravel(),lev + 1)
        starts,ends = point_index.ranges(keys)
        occupied = ends > starts
        keys,starts,ends = keys[occupied],starts[occupied],ends[occupied]

    return np.concatenate(leaves)


def _balance_tree(XX,c0,L0,leaf_keys,point_index=None):
    """
    Balance the tree using the bottom-up octant-set sweep from Algorithm 1 in
    Malhotra and Biros, Algorithm 967 (2016), Section 2.3, operating on octants only.
    https://users.flatironinstitute.org/~dmalhotra/files/pubs/pvfmm.pdf

    Starting from the parents of the unbalanced leaves, we collect non-leaf
    octants level by level from finest to coarsest. At each level we propagate
    the parents of all possible colleagues upward, which identifies the octants
    that must be non-leaf in the next coarser level for the 2:1 balance
    constraint to hold.

    Parameters:
    - XX, c0, L0: Point cloud and root geometry, used to exclude empty leaves.
    - leaf_keys: Morton keys for the leaves of the unbalanced tree.

    Returns:
    - balanced_leaf_keys: Morton keys for the leaves of the balanced tree.
    - balanced_tree_keys: Morton keys for all occupied nodes in the balanced tree.
    """

    ndim = XX.shape[-1]
    if (leaf_keys.shape[0] == 0):
        return leaf_keys.copy(),leaf_keys.copy()

    # Immediate parents suffice: the colleague stencil includes self, so the
    # downward loop propagates every ancestor without constructing it upfront.
    nonroot = leaf_keys[morton.get_level(leaf_keys) > 0]
    nonleaf_keys = np.unique(morton.get_parent(nonroot,ndim=ndim))

    if (nonleaf_keys.shape[0] == 0):
        return leaf_keys.copy(),np.sort(leaf_keys)

    nonleaf_levels = morton.get_level(nonleaf_keys)
    depth = np.max(nonleaf_levels)

    balanced_nonleaf_keys = []
    carry = np.array([],dtype=int)

    for lev in range(depth,-1,-1):
        current = nonleaf_keys[nonleaf_levels == lev]
        if (carry.shape[0] > 0):
            current = np.unique(np.concatenate((current,carry)))

        if (current.shape[0] == 0):
            carry = np.array([],dtype=int)
            continue

        balanced_nonleaf_keys.append(current)

        if (lev > 0):
            colleagues = morton.get_colleagues_vec(current,ndim=ndim).reshape(-1)
            colleagues = np.unique(colleagues[colleagues >= 0])
            carry = np.unique(morton.get_parent(colleagues,ndim=ndim))
        else:
            carry = np.array([],dtype=int)

    balanced_nonleaf_keys = np.unique(np.concatenate(balanced_nonleaf_keys))

    nsibs = 1 << ndim
    child_roots = morton.set_level(
        morton.clear_level(balanced_nonleaf_keys) << ndim,
        morton.get_level(balanced_nonleaf_keys) + 1,
    )
    children = child_roots.reshape(-1,1) | np.arange(nsibs,dtype=int).reshape(1,-1)

    candidate_tree_keys = np.unique(np.concatenate((np.array([0],dtype=int),children.reshape(-1))))
    candidate_leaf_keys = np.setdiff1d(candidate_tree_keys,balanced_nonleaf_keys,assume_unique=True)

    if point_index is None:
        point_index = _PointIndex(XX,c0,L0)
    starts,ends = point_index.ranges(candidate_leaf_keys)
    balanced_leaf_keys = candidate_leaf_keys[ends > starts]

    return balanced_leaf_keys,_get_complete_tree(balanced_leaf_keys,ndim=ndim)

def _get_leaf_keys(XX,c0,L0,max_bs,point_index=None):
    """
    Generates an unbalanced and balanced tree structure for a given set of points `XX`.
    The function first creates an unbalanced tree based on the specified maximum number
    of points per leaf (`max_bs`). If the resulting tree is unbalanced, it then balances
    the tree to satisfy the 2:1 balance constraint on the leaves.

    Parameters:
    - XX: A 2D NumPy array of shape (N, ndim), where N is the number of points and `ndim` is the number of dimensions
          (either 2 or 3). This array contains the coordinates of the points to be partitioned.
    - c0: A float or array-like object specifying the origin (or center) used for computing Morton codes.
    - L0: A float specifying the length of the bounding box used for computing Morton codes.
    - max_bs: An integer specifying the maximum number of points allowed in a leaf.

    Returns:
    - unbalanced_keys: A sorted 1D NumPy array of integers containing the Morton keys of the unbalanced leaf nodes.
    - leaf_keys: A sorted 1D NumPy array of integers containing the balanced leaf nodes.
    - tree_keys: A sorted 1D NumPy array of integers containing all occupied nodes in the balanced tree.
    """

    if point_index is None:
        point_index = _PointIndex(XX,c0,L0)
    unbalanced_keys = _unbalanced_tree(XX,c0,L0,max_bs,point_index)

    lev = morton.get_level(unbalanced_keys)
    if ( (np.max(lev) - np.min(lev)) > 1):
        leaf_keys,tree_keys = _balance_tree(XX,c0,L0,unbalanced_keys,point_index)
    else:
        leaf_keys = unbalanced_keys.copy()
        tree_keys = _get_complete_tree(leaf_keys,ndim=XX.shape[-1])
    return np.sort(unbalanced_keys),np.sort(leaf_keys),np.sort(tree_keys)

def _get_complete_tree(leaf_keys,ndim):
    """
    Return the ancestor closure of occupied leaves, including the root.

    This is a sparse occupied subtree, not a complete partition of the root
    domain: empty siblings are deliberately absent.

    Parameters:
    - leaf_keys: A 1D NumPy array of integers containing the Morton keys of the leaf nodes.
    - ndim: An integer specifying the number of dimensions (either 2 or 3).

    Returns:
    - tree_keys: A sorted 1D NumPy array of integers containing the Morton keys of all nodes in the complete tree,
                 including both leaf and tree nodes.
    """
    if (leaf_keys.shape[0] == 0):
        return leaf_keys.copy()

    keys_curr = np.unique(leaf_keys.astype(int))
    all_keys = [keys_curr]

    while (keys_curr.shape[0] > 0):
        positive = morton.get_level(keys_curr) > 0
        if (not np.any(positive)):
            break
        keys_curr = np.unique(morton.get_parent(keys_curr[positive],ndim=ndim))
        all_keys.append(keys_curr)

    return np.sort(np.unique(np.concatenate(all_keys)))

######################################  MORTON TREE CLASS #########################################

class MortonTree(AbstractTree):
    """Occupied 2:1-balanced quad/octree with level-major box numbering.

    Points use one stable finest-level Morton ordering. Each box, including
    internal nodes, owns an interval in that ordering. ``get_box_inds`` returns
    a read-only view; use ``.copy()`` for a mutable work array. Equal finest-cell
    keys preserve input order. Coordinates must not change after construction.

    Roots are half-open: c0 - L0/2 <= XX < c0 + L0/2 in every coordinate.
    Refinement stops at the Morton depth cap even if occupancy is still high.
    """

    def __init__(self,XX,leaf_size=100,c0=None,L0=None):
        XX = np.asarray(XX,dtype=float)
        if XX.ndim != 2 or XX.shape[1] not in (2,3):
            raise ValueError("XX must have shape (N, 2) or (N, 3)")
        if XX.shape[0] == 0 or not np.all(np.isfinite(XX)):
            raise ValueError("XX must contain at least one point and only finite coordinates")
        leaf_size_float = float(leaf_size)
        if (not np.isfinite(leaf_size_float) or leaf_size_float <= 0 or
                not leaf_size_float.is_integer()):
            raise ValueError("leaf_size must be a positive integer")
        leaf_size = int(leaf_size_float)

        if c0 is None or L0 is None:
            c0_auto,L0_auto = morton.get_root_params(XX)
            if c0 is None:
                c0 = c0_auto
            if L0 is None:
                L0 = L0_auto
        c0 = np.asarray(c0,dtype=float)
        if c0.shape != (XX.shape[1],) or not np.all(np.isfinite(c0)):
            raise ValueError("c0 must be a finite vector of shape (ndim,)")
        L0_array = np.asarray(L0,dtype=float)
        if L0_array.ndim != 0 or not np.isfinite(L0_array) or L0_array <= 0:
            raise ValueError("L0 must be a finite positive scalar")
        L0 = float(L0_array)
        lower,upper = c0 - 0.5*L0,c0 + 0.5*L0
        if (not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)) or
                np.any(upper <= lower)):
            raise ValueError("root bounds must be finite and resolvable in floating point")
        if np.any(XX < lower) or np.any(XX >= upper):
            raise ValueError("XX must lie inside the half-open root [c0-L0/2, c0+L0/2)")

        self.XX = XX
        self.c0 = c0
        self.L0 = L0
        self.ndim = XX.shape[1]
        self.leaf_size = leaf_size
        self.leaf_size_param = leaf_size
        self._form_tree(leaf_size)

    @property
    def N(self):
        return self.XX.shape[0]

    @property
    def nboxes(self):
        return self.tree_keys.shape[0]

    @property
    def nleaves(self):
        return self.leaf_keys.shape[0]

    @property
    def nlevels(self):
        return self.level_sep.shape[0]-1

    def _form_tree(self,leaf_size):
        point_index = _PointIndex(self.XX,self.c0,self.L0)
        self.unbalanced_keys,self.leaf_keys,self.tree_keys = _get_leaf_keys(
            self.XX,self.c0,self.L0,leaf_size,point_index
        )
        levels = morton.get_level(self.tree_keys)
        self.level_sep = np.r_[0,np.flatnonzero(np.diff(levels)) + 1,self.nboxes]

        self.point_order = point_index.order
        self.point_start,self.point_end = point_index.ranges(self.tree_keys)
        self.point_start.flags.writeable = False
        self.point_end.flags.writeable = False
        # Sorted raw codes are construction scratch; only the permutation and
        # per-box intervals survive, using O(N + nboxes) storage.
        self.parent_list = np.full(self.nboxes,-1,dtype=int)
        self.parent_list[1:] = findin_sorted_vec(
            self.tree_keys,morton.get_parent(self.tree_keys[1:],ndim=self.ndim)
        )
        self._leaf_layout = None
        self.Clist_tmp = None
        self.Llist_tmp = None
        self.Llist_unique_tmp = None
        self._fine_neighbors = None
        self._fine_offsets = None
        self.leaf_boxes_tmp = None
        self.leaf_entry_by_box_tmp = None
        self.box_is_leaf_tmp = None
        self.children_fullvec_tmp = None
        # Rebuilding a tree also invalidates inherited coloring caches.
        self.__dict__.pop('box_colors',None)
        self.__dict__.pop('num_colors',None)

    @property
    def perm_leaf(self):
        """Compatibility layout, grouping points in level-major leaf order."""
        if self._leaf_layout is None:
            self._leaf_layout = self._get_leaf_sep()
        return self._leaf_layout[0]

    @property
    def leaf_sep(self):
        """Offsets into perm_leaf; materialized together on first access."""
        if self._leaf_layout is None:
            self._leaf_layout = self._get_leaf_sep()
        return self._leaf_layout[1]

    def _get_leaf_sep(self):
        leaves = self.get_leaves()
        counts = self.point_end[leaves] - self.point_start[leaves]
        leaf_sep = np.r_[0,np.cumsum(counts)]
        perm_leaf = np.concatenate([self.get_box_inds(box) for box in leaves])
        perm_leaf.flags.writeable = False
        leaf_sep.flags.writeable = False
        return perm_leaf,leaf_sep

    def _get_CL_lists_vec(self):
        """
        Generates an internal data structures to store colleague and coarse neighbors for each box in the tree.

        Returns:
        - Clist: A 2D NumPy array where each row contains the colleague neighbors for each box.
        - Llist: A 2D NumPy array where each row contains the coarse neighbors for each box.
        """

        nmax  = 3**self.ndim
        Llist = np.ones((self.nboxes,nmax),dtype=int) * (-1)
        Clist = np.ones((self.nboxes,nmax),dtype=int) * (-1)

        for lev in range(1,self.nlevels):

            ## colleague neigh boxes
            keys_lev      = self.tree_keys[self.level_sep[lev]:self.level_sep[lev+1]]
            nboxes_lev    = keys_lev.shape[0]

            colleague_keys = morton.get_colleagues_vec(keys_lev,ndim=self.ndim)
            colleague_keys = colleague_keys.flatten()

            colleague_boxes = findin_sorted_vec(self.tree_keys,colleague_keys)
            Clist[self.level_sep[lev]:\
                  self.level_sep[lev+1]] = colleague_boxes.reshape(nboxes_lev,nmax)

            ## coarse neigh boxes
            ## coarse inds keeps track of which indices are candidates for being
            ## a coarse neighboring box
            coarse_inds = np.arange(nboxes_lev*nmax)
            coarse_inds = coarse_inds[colleague_keys >= 0]

            # if a colleague box does not exist, check if its parent box
            iskey_bool = colleague_boxes[coarse_inds] >= 0
            coarse_inds = coarse_inds[~iskey_bool]

            par_keys    = morton.get_parent(colleague_keys[coarse_inds],ndim=self.ndim)
            isleaf_bool = in_sorted_vec(self.leaf_keys, par_keys)
            coarse_inds = coarse_inds[isleaf_bool]

            # check if the parent box of colleague is a leaf box
            coarse_neigh_boxes = np.ones(nboxes_lev*nmax,dtype=int) * (-1)
            coarse_neigh_boxes[coarse_inds] = findin_sorted_vec(self.tree_keys,par_keys[isleaf_bool])
            Llist[self.level_sep[lev]:\
                  self.level_sep[lev+1]] = coarse_neigh_boxes.reshape(nboxes_lev,nmax)
        return Clist,Llist

    def _ensure_neighbor_lists(self):
        if (self.Clist_tmp is None) or (self.Llist_tmp is None):
            self.Clist_tmp, self.Llist_tmp = self._get_CL_lists_vec()
        if self.Llist_unique_tmp is None:
            # Sort short rows once and compact their unique valid entries.
            # nboxes is a sentinel larger than any valid box index.
            ordered = np.sort(np.where(self.Llist_tmp >= 0,self.Llist_tmp,self.nboxes),axis=1)
            keep = ordered < self.nboxes
            keep[:,1:] &= ordered[:,1:] != ordered[:,:-1]
            rows,cols = np.nonzero(keep)
            compact_cols = np.cumsum(keep,axis=1)[rows,cols] - 1
            self.Llist_unique_tmp = np.full_like(self.Llist_tmp,-1)
            self.Llist_unique_tmp[rows,compact_cols] = ordered[rows,cols]

    def _color_boxes(self):
        self._ensure_neighbor_lists()
        self.box_colors,self.num_colors = greedy_colors(self.Clist_tmp,self.Llist_unique_tmp)

    def _ensure_leaf_cache(self):
        if self.leaf_boxes_tmp is None:
            leaf_boxes = findin_sorted_vec(self.tree_keys,self.leaf_keys)
            self.leaf_boxes_tmp = leaf_boxes

            self.leaf_entry_by_box_tmp = np.ones(self.nboxes,dtype=int) * (-1)
            self.leaf_entry_by_box_tmp[leaf_boxes] = np.arange(leaf_boxes.shape[0])

            self.box_is_leaf_tmp = np.zeros(self.nboxes,dtype=bool)
            self.box_is_leaf_tmp[leaf_boxes] = True

    def _ensure_traversal_caches(self):
        if self.children_fullvec_tmp is None:
            nchildren = 1 << self.ndim
            children_fullvec = np.full((self.nboxes,nchildren),-1,dtype=int)
            child_boxes = np.arange(1,self.nboxes)
            child_slots = self.tree_keys[1:] & (nchildren - 1)
            children_fullvec[self.parent_list[1:],child_slots] = child_boxes
            self.children_fullvec_tmp = children_fullvec


    def get_ordered_colleague_pairs(self,root_level=1):
        self._ensure_neighbor_lists()

        Clist = self.Clist_tmp; nmax   = Clist.shape[-1]
        unordered_blocks = []
        grouped_blocks = []
        ngrouped_ordered = []

        for lev in range(root_level+1,self.nlevels):

            boxes_lev = np.arange(self.level_sep[lev],self.level_sep[lev+1])
            Clist_lev = Clist[boxes_lev]

            for j in range(nmax):

                tmp_bool = Clist_lev[:,j] > 0
                targets = boxes_lev[tmp_bool]
                sources = Clist_lev[tmp_bool,j]

                tree_pair_bool = np.logical_and(np.logical_not(self.are_leaves(sources)),\
                                                np.logical_not(self.are_leaves(targets)))
                ordered_num = np.sum(tree_pair_bool)
                stacked_tar_scs = np.vstack((targets,sources)).T
                if ordered_num > 0:
                    grouped_blocks.append(stacked_tar_scs[tree_pair_bool])
                    ngrouped_ordered.append(ordered_num)
                if np.any(~tree_pair_bool):
                    unordered_blocks.append(stacked_tar_scs[~tree_pair_bool])

        unordered_list_pairs = (np.concatenate(unordered_blocks) if unordered_blocks else
                                np.empty((0,2),dtype=int))
        ngrouped_ordered = np.asarray(ngrouped_ordered,dtype=int)
        if grouped_blocks:
            grouped_list_pairs = np.zeros((len(grouped_blocks),int(ngrouped_ordered.max()),2),dtype=int)
            for j,pairs in enumerate(grouped_blocks):
                grouped_list_pairs[j,:len(pairs)] = pairs
        else:
            grouped_list_pairs = np.zeros((nmax*self.nlevels,0,2),dtype=int)

        return unordered_list_pairs,ngrouped_ordered,grouped_list_pairs


    ######################################################################################
    ## The functions below implement abstract methods of AbstractTree.
    ## For descriptions, see `abstract_tree.py`.

    def get_box_level(self,box):
        key = self.tree_keys[box]
        return morton.get_level(key)

    def get_box_parent(self,box):
        return self.parent_list[box]

    def get_box_children(self,box):
        self._ensure_traversal_caches()
        children = self.children_fullvec_tmp[box]
        return children[children >= 0]

    def get_box_children_fullvec(self,box):
        self._ensure_traversal_caches()
        return self.children_fullvec_tmp[box]

    def get_boxes_children_fullvec(self,boxes):
        self._ensure_traversal_caches()
        return self.children_fullvec_tmp[boxes]

    def get_box_colleague_neigh(self,box):
        self._ensure_neighbor_lists()

        box_list = self.Clist_tmp[box]
        return box_list[box_list >= 0]

    def get_boxes_colleague_neigh_fullvec(self,boxes):
        self._ensure_neighbor_lists()
        return self.Clist_tmp[boxes]

    def get_box_coarse_neigh(self,box):
        self._ensure_neighbor_lists()

        box_list = self.Llist_unique_tmp[box]
        return box_list[box_list >= 0]

    def get_boxes_coarse_neigh_fullvec(self,boxes):
        self._ensure_neighbor_lists()
        return self.Llist_unique_tmp[boxes]

    def get_box_fine_neigh(self,box):
        self._ensure_neighbor_lists()
        if self._fine_neighbors is None:
            # Reverse the unique coarse-neighbor relation once, instead of
            # scanning the entire neighbor matrix for every fine-neighbor query.
            rows,cols = np.nonzero(self.Llist_unique_tmp >= 0)
            coarse = self.Llist_unique_tmp[rows,cols]
            order = np.argsort(coarse,kind='stable')
            self._fine_neighbors = rows[order]
            self._fine_neighbors.flags.writeable = False
            self._fine_offsets = np.r_[0,np.cumsum(np.bincount(coarse,minlength=self.nboxes))]
        # Normalize negative indices just as NumPy indexing does for other getters.
        if box < 0:
            box += self.nboxes
        if not 0 <= box < self.nboxes:
            raise IndexError("box index out of range")
        start,end = self._fine_offsets[box],self._fine_offsets[box+1]
        rows = self._fine_neighbors[start:end]
        assert np.all(self.are_leaves(rows))
        return rows

    def get_box_center(self,box):
        key = self.tree_keys[box]

        return morton.get_key_params(key,self.c0,self.L0)[0]

    def get_boxes_centers(self,boxes):

        keys = self.tree_keys[boxes]
        return morton.get_keys_centers(keys,self.c0,self.L0)

    def get_box_length(self,box):
        return self.L0 / (1 << self.get_box_level(box))

    def get_leaves(self):
        self._ensure_leaf_cache()
        return self.leaf_boxes_tmp

    def get_boxes_level(self,lev):
        return np.arange(self.level_sep[lev],self.level_sep[lev+1])

    def get_leaves_above(self,lev):
        lev_leafs = morton.get_level(self.leaf_keys)
        keys      = self.leaf_keys[lev_leafs<lev]

        return findin_sorted_vec(self.tree_keys,keys)

    def get_box_inds(self,box):
        """Read-only point indices in stable Morton order; O(1) view creation."""
        return self.point_order[self.point_start[box]:self.point_end[box]]

    def is_leaf(self,box):
        self._ensure_leaf_cache()
        return self.box_is_leaf_tmp[box]

    are_leaves = is_leaf
