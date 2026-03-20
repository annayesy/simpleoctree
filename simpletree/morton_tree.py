import numpy as np
from simpletree import morton
from simpletree.abstract_tree import AbstractTree

##############################################################################
######### Utilities for searching in sorted 1d numpy arrays ##################

def in_sorted_vec(sorted_list,entries):
    n = sorted_list.shape[-1]

    inds = np.searchsorted(sorted_list,entries)
    bool_list = np.zeros(entries.shape[0],dtype=bool)
    bool_list[inds == n] = False
    bool_list[inds < n]  = (sorted_list[inds[inds < n]] == entries[inds<n])

    return bool_list

def in_sorted(sorted_list,entry):
    return in_sorted_vec(sorted_list,np.array([entry],dtype=int))[0]

def findin_sorted_vec(sorted_list,entries):
    n = sorted_list.shape[-1]

    result = np.zeros(entries.shape[0],dtype=int)
    bool_list = in_sorted_vec(sorted_list,entries)

    result[~bool_list] = -1
    result[bool_list]  = np.searchsorted(sorted_list,entries[bool_list])

    return result

def findin_sorted(sorted_list,entry):
    return findin_sorted_vec(sorted_list,np.array([entry],dtype=int))[0]

##########################################################################

def _max_level(ndim):
    if (ndim == 2):
        return morton.MAX_LEVEL2
    else:
        return morton.MAX_LEVEL3

def _sorted_deepest_raw_keys(XX,c0,L0,ndim):
    deepest_level = _max_level(ndim) - 1
    anchors_points = morton.points_to_anchors(XX,deepest_level,c0,L0,ndim=ndim)
    deepest_keys = morton.anchors_to_keys(anchors_points,ndim=ndim)
    return deepest_level,np.sort(morton.clear_level(deepest_keys))

def _unbalanced_tree(XX,c0,L0,max_bs):
    """
    Constructs an unbalanced tree structure for a given set of points `XX` using Morton codes.
    Iteratively partitions the points into leaves based on their spatial location until all points
    are assigned to leaves that contain no more than a specified maximum number of points (`max_bs`).

    Parameters:
    - XX: A 2D NumPy array of shape (N, ndim), where N is the number of points and `ndim` is the number
          of dimensions (either 2 or 3). This array contains the coordinates of the points to be partitioned.
    - c0: A float or array-like object specifying the origin (or center) used for computing Morton codes.
    - L0: A float specifying the length of the bounding box used for computing Morton codes.
    - max_bs: An integer specifying the maximum number of points allowed in a leaf.

    Returns:
    - leaf_keys: A 1D NumPy array of integers containing the Morton keys of the leaf nodes in the tree.
                 Each key corresponds to a unique leaf containing a subset of the input points `XX`.
    """

    N = XX.shape[0]; ndim = XX.shape[-1]
    MAX_LEVEL = _max_level(ndim)
    deepest_level,sorted_raw_keys = _sorted_deepest_raw_keys(XX,c0,L0,ndim)

    # active mask is stored in deepest-level Morton order. A coarse octant and
    # all of its descendants occupy a contiguous range in this ordering.
    active_sorted = np.ones(N,dtype=bool)
    leaf_keys   = np.zeros(N,int); acc_leaves = 0

    for lev in range(MAX_LEVEL):

        if (not np.any(active_sorted)):
            break

        active_inds = np.flatnonzero(active_sorted)
        shift = ndim * (deepest_level - lev)
        keys_points = sorted_raw_keys[active_inds] >> shift

        if (keys_points.shape[0] == 0):
            break

        change_inds = np.flatnonzero(np.diff(keys_points)) + 1
        starts = np.concatenate((np.array([0],dtype=int),change_inds))
        ends = np.concatenate((change_inds,np.array([keys_points.shape[0]],dtype=int)))
        counts = ends - starts

        leaf_group_bool = counts <= max_bs
        if (not np.any(leaf_group_bool)):
            continue

        keep_starts = starts[leaf_group_bool]
        keep_ends = ends[leaf_group_bool]
        keep_raw = keys_points[keep_starts]

        for start,end in zip(keep_starts,keep_ends):
            active_sorted[active_inds[start:end]] = False

        nleaf_lev = keep_raw.shape[0]
        leaf_keys[acc_leaves : acc_leaves + nleaf_lev] = morton.set_level(keep_raw,lev)
        acc_leaves += nleaf_lev

    if (np.any(active_sorted)):
        active_inds = np.flatnonzero(active_sorted)
        keep_raw = np.unique(sorted_raw_keys[active_inds])
        nleaf_lev = keep_raw.shape[0]
        leaf_keys[acc_leaves : acc_leaves + nleaf_lev] = morton.set_level(keep_raw,deepest_level)
        acc_leaves += nleaf_lev

    leaf_keys = leaf_keys[:acc_leaves]
    return leaf_keys

def _balance_tree(XX,c0,L0,leaf_keys):
    """
    Balance the tree using the bottom-up octant-set sweep from Algorithm 1 in
    Lashuk et al. (2016), operating on octants only.

    Starting from the complete unbalanced tree, we collect the set of non-leaf
    octants level by level from finest to coarsest. At each level we propagate
    the parents of all possible colleagues upward, which identifies the octants
    that must be non-leaf in the next coarser level for the 2:1 balance
    constraint to hold.

    Parameters:
    - XX: Point cloud used only to infer the spatial dimension.
    - c0: Unused, retained for API compatibility.
    - L0: Unused, retained for API compatibility.
    - leaf_keys: Morton keys for the leaves of the unbalanced tree.

    Returns:
    - balanced_leaf_keys: Morton keys for the leaves of the balanced tree.
    - balanced_tree_keys: Morton keys for all occupied nodes in the balanced tree.
    """

    ndim = XX.shape[-1]
    if (leaf_keys.shape[0] == 0):
        return leaf_keys.copy(),leaf_keys.copy()

    unbalanced_tree_keys = _get_complete_tree(leaf_keys,ndim=ndim)
    nonleaf_keys = np.setdiff1d(unbalanced_tree_keys,leaf_keys,assume_unique=True)

    if (nonleaf_keys.shape[0] == 0):
        return leaf_keys.copy(),np.sort(unbalanced_tree_keys)

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

    deepest_level,deepest_raw_keys = _sorted_deepest_raw_keys(XX,c0,L0,ndim)

    keep_leaf_keys = []
    leaf_levels = morton.get_level(candidate_leaf_keys)
    raw_leaf_keys = morton.clear_level(candidate_leaf_keys)
    for lev in np.unique(leaf_levels):
        lev_mask = leaf_levels == lev
        keys_lev = candidate_leaf_keys[lev_mask]
        raw_lev = raw_leaf_keys[lev_mask]

        shift = ndim * (deepest_level - lev)
        lower = raw_lev << shift
        upper = (raw_lev + 1) << shift

        ind0 = np.searchsorted(deepest_raw_keys,lower,side='left')
        ind1 = np.searchsorted(deepest_raw_keys,upper,side='left')
        keep_leaf_keys.append(keys_lev[ind1 > ind0])

    if (len(keep_leaf_keys) == 0):
        balanced_leaf_keys = np.array([0],dtype=int)
    else:
        balanced_leaf_keys = np.sort(np.concatenate(keep_leaf_keys))

    return balanced_leaf_keys,_get_complete_tree(balanced_leaf_keys,ndim=ndim)

def _get_leaf_keys(XX,c0,L0,max_bs):
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

    unbalanced_keys = _unbalanced_tree(XX,c0,L0,max_bs)

    lev = morton.get_level(unbalanced_keys)
    if ( (np.max(lev) - np.min(lev)) > 1):
        leaf_keys,tree_keys = _balance_tree(XX,c0,L0,unbalanced_keys.copy())
    else:
        leaf_keys = unbalanced_keys.copy()
        tree_keys = _get_complete_tree(leaf_keys,ndim=XX.shape[-1])
    return np.sort(unbalanced_keys),np.sort(leaf_keys),np.sort(tree_keys)

def _get_complete_tree(leaf_keys,ndim):
    """
    Generates a complete tree structure from a given set of leaf nodes by adding all necessary
    tree nodes. This ensures that the resulting tree is a complete representation from the
    root to the given leaves.

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

    def __init__(self,XX,leaf_size = 100,c0=None,L0=None):

        """
        Initializes the tree structure for a given set of points `XX`. This method sets up the necessary
        parameters for the tree, including the origin, bounding box length, and leaf size. If the origin (`c0`)
        and bounding box length (`L0`) are not provided, they are automatically calculated.

        Parameters:
        - XX: A 2D NumPy array of shape (N, ndim), where N is the number of points and `ndim` is the number of dimensions
              (either 2 or 3). This array contains the coordinates of the points to be partitioned.
        - leaf_size: An integer specifying the maximum number of points allowed in a leaf. Default is 100.
        - c0: A float or array-like object specifying the origin (or center) used for computing Morton codes. If None,
              the origin is automatically calculated. Default is None.
        - L0: A float specifying the length of the bounding box used for computing Morton codes. If None, the bounding box
              length is automatically calculated. Default is None.
        """

        assert ( (XX.shape[-1] == 2) or (XX.shape[-1] == 3))

        if (c0 is None) or (L0 is None):
            c0_auto,L0_auto = morton.get_root_params(XX)
            if (c0 is None):
                c0 = c0_auto
            if (L0 is None):
                L0 = L0_auto
        self.XX        = XX
        self.c0        = c0
        self.L0        = L0
        self.ndim      = XX.shape[-1]
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
        return self.get_leaves().shape[0]

    @property
    def nlevels(self):
        return self.level_sep.shape[0]-1

    def _form_tree(self,leaf_size):

        """
        Forms the tree structure by generating unbalanced and balanced leaf nodes, and then completing
        the tree with all necessary ancestor nodes. This method also initializes various attributes for
        efficient tree traversal and operations.

        Parameters:
        - leaf_size: An integer specifying the maximum number of points allowed in a leaf.

        Attributes Set:
        - unbalanced_keys: A sorted 1D NumPy array of integers containing the Morton keys of the unbalanced leaf nodes.
        - leaf_keys: A sorted 1D NumPy array of integers containing the Morton keys of the balanced leaf nodes.
        - tree_keys: A sorted 1D NumPy array of integers containing the Morton keys of all nodes in the complete tree.
        - level_sep: A 1D NumPy array indicating the separation indices for nodes at each level in the tree.
        - perm_leaf: A permutation array for indices [1:N].
        - leaf_sep: Separation indices for indices [1:N].
        - Clist_tmp: An internal data structure for adjacent colleagues.
        - Llist_tmp: An internal data structure for coarse neighbors.
        """

        unbalanced_keys,leaf_keys,tree_keys = _get_leaf_keys(self.XX,self.c0,self.L0,leaf_size)

        nboxes    = tree_keys.shape[0]
        lev       = morton.get_level(tree_keys)

        _,level_sep = np.unique(lev,return_index=True)
        level_sep   = np.concatenate(( level_sep, np.array([nboxes],dtype=int) ))

        self.unbalanced_keys = unbalanced_keys # only for plotting purposes
        self.leaf_keys       = leaf_keys

        self.tree_keys       = tree_keys
        self.level_sep       = level_sep

        self.perm_leaf,self.leaf_sep = self._get_leaf_sep()

        self.Clist_tmp = None
        self.Llist_tmp = None


    def _get_leaf_sep(self):
        """
        Generates a permutation vector that groups indices belonging to the same leaf together and
        computes separation indices for the permutation.

        Returns:
        - perm_leaf: A permutation vector of indices such that indices for the same leaf are grouped together.
        - leaf_sep: An array indicating the separation indices for the permutation.
        """
        leaf_levels = morton.get_level(self.leaf_keys)
        max_lev = np.max(leaf_levels)

        anchors_points = morton.points_to_anchors(self.XX,max_lev,self.c0,self.L0,ndim=self.ndim)
        keys_points = morton.anchors_to_keys(anchors_points,ndim=self.ndim)
        raw_keys_points = morton.clear_level(keys_points)

        sorted_perm = np.argsort(raw_keys_points)
        sorted_raw_keys = raw_keys_points[sorted_perm]

        nleaf = self.leaf_keys.shape[0]
        perm_leaf = np.zeros(self.N,dtype=int)
        leaf_sep = np.zeros(nleaf+1,dtype=int)

        raw_leaf_keys = morton.clear_level(self.leaf_keys)
        acc = 0
        for j,key in enumerate(raw_leaf_keys):
            shift = self.ndim * (max_lev - leaf_levels[j])
            lower = key << shift
            upper = (key + 1) << shift

            ind0 = np.searchsorted(sorted_raw_keys,lower,side='left')
            ind1 = np.searchsorted(sorted_raw_keys,upper,side='left')

            nloc = ind1 - ind0
            perm_leaf[acc : acc + nloc] = sorted_perm[ind0:ind1]
            acc += nloc
            leaf_sep[j+1] = acc

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
            iskey_bool   = in_sorted_vec(self.tree_keys,colleague_keys[coarse_inds])
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


    def get_ordered_colleague_pairs(self,root_level=1):
        self._ensure_neighbor_lists()

        Clist = self.Clist_tmp; nmax   = Clist.shape[-1]

        offset_unordered = 0;
        count_ordered    = 0

        unordered_list_pairs   = np.zeros((nmax*self.nboxes,2),dtype=int)
        ngrouped_ordered       = np.zeros(nmax*self.nlevels,dtype=int)
        grouped_list_pairs     = np.zeros((nmax*self.nlevels,self.nboxes,2),dtype=int)

        for lev in range(root_level+1,self.nlevels):

            boxes_lev = np.arange(self.level_sep[lev],self.level_sep[lev+1])
            Clist_lev = Clist[boxes_lev]

            for j in range(nmax):

                tmp_bool = Clist_lev[:,j] > 0
                targets = boxes_lev[tmp_bool]
                sources = Clist_lev[tmp_bool,j]

                tree_pair_bool = np.logical_and(np.logical_not(self.are_leaves(sources)),\
                                                np.logical_not(self.are_leaves(targets)))

                ordered_num     = np.sum(tree_pair_bool)
                unordered_num   = targets.shape[0] - ordered_num

                stacked_tar_scs = np.vstack((targets,sources)).T

                if (ordered_num > 0):
                    grouped_list_pairs[count_ordered,:ordered_num] = stacked_tar_scs[tree_pair_bool]
                    ngrouped_ordered[count_ordered]                = ordered_num
                    count_ordered += 1

                unordered_list_pairs[offset_unordered : offset_unordered + unordered_num] = \
                stacked_tar_scs[np.logical_not(tree_pair_bool)]
                offset_unordered += unordered_num

        unordered_list_pairs = unordered_list_pairs[:offset_unordered]

        if (count_ordered > 0):
            ngrouped_ordered     = ngrouped_ordered[:count_ordered]
            grouped_list_pairs   = grouped_list_pairs[:count_ordered,:np.max(ngrouped_ordered)]
        else:
            ngrouped_ordered     = ngrouped_ordered[:0]
            grouped_list_pairs = np.zeros((nmax*self.nlevels,0,2),dtype=int)

        return unordered_list_pairs,ngrouped_ordered,grouped_list_pairs


    ######################################################################################
    ## The functions below implement abstract methods of AbstractTree.
    ## For descriptions, see `abstract_tree.py`.

    def get_box_level(self,box):
        key = self.tree_keys[box]
        return morton.get_level(key)

    def get_box_parent(self,box):
        key = self.tree_keys[box]
        par_key = morton.get_parent(key,ndim=self.ndim)

        return findin_sorted(self.tree_keys,par_key)

    def get_box_children(self,box):
        key = self.tree_keys[box]
        child_keys = morton.get_children(key,ndim=self.ndim)

        valid_keys = in_sorted_vec(self.tree_keys,child_keys)
        return findin_sorted_vec(self.tree_keys,child_keys[valid_keys])

    def get_box_children_fullvec(self,box):
        key = self.tree_keys[box]
        child_keys = morton.get_children(key,ndim=self.ndim)
        return findin_sorted_vec(self.tree_keys,child_keys)

    def get_box_colleague_neigh(self,box):
        self._ensure_neighbor_lists()

        box_list = self.Clist_tmp[box]
        return box_list[box_list >= 0]

    def get_box_coarse_neigh(self,box):
        self._ensure_neighbor_lists()

        box_list = self.Llist_tmp[box]
        return np.unique(box_list[box_list >= 0])

    def get_box_fine_neigh(self,box):
        self._ensure_neighbor_lists()
        rows, _ = np.where(self.Llist_tmp == box)
        rows = np.unique(rows)

        assert np.all(self.are_leaves(rows))
        return rows

    def get_box_center(self,box):
        key = self.tree_keys[box]

        return morton.get_key_params(key,self.c0,self.L0)[0]

    def get_boxes_centers(self,boxes):

        keys = self.tree_keys[boxes]
        return morton.get_keys_centers(keys,self.c0,self.L0)

    def get_box_length(self,box):
        key = self.tree_keys[box]
        return morton.get_key_params(key,self.c0,self.L0)[1]

    def get_leaves(self):
        return findin_sorted_vec(self.tree_keys,self.leaf_keys)

    def get_boxes_level(self,lev):
        return np.arange(self.level_sep[lev],self.level_sep[lev+1])

    def get_leaves_above(self,lev):
        lev_leafs = morton.get_level(self.leaf_keys)
        keys      = self.leaf_keys[lev_leafs<lev]

        return findin_sorted_vec(self.tree_keys,keys)

    def get_box_inds(self,box):
        key = self.tree_keys[box]

        leaf_entry = findin_sorted(self.leaf_keys,key)

        if ( leaf_entry > 0):
            return self.perm_leaf[ self.leaf_sep[leaf_entry] : self.leaf_sep[leaf_entry+1]]

        else:
            lev = morton.get_level(key)

            anchors_points = morton.points_to_anchors(self.XX,lev,self.c0,self.L0,ndim=self.ndim)
            keys_points    = morton.anchors_to_keys(anchors_points,ndim=self.ndim)
            sorted_perm    = np.argsort(keys_points)

            unique_keys,indices = np.unique(keys_points[sorted_perm],return_index=True)
            indices = np.concatenate(( indices, np.array([self.N]) ))
            j = findin_sorted(unique_keys,key)
            return sorted_perm[ indices[j] : indices[j+1]]

    def is_leaf(self,box):
        key = self.tree_keys[box]
        return in_sorted(self.leaf_keys,key)

    def are_leaves(self,box):
        key = self.tree_keys[box]
        return in_sorted_vec(self.leaf_keys,key)
