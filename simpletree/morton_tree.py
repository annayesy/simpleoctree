import numpy as np
from simpletree import morton
from time import time
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
    if (ndim == 2):
        MAX_LEVEL = morton.MAX_LEVEL2
    else:
        MAX_LEVEL = morton.MAX_LEVEL3

    # find leaves in tree
    active_bool = np.ones(N,dtype=bool)
    leaf_keys   = np.zeros(N,int); acc_leaves = 0

    for lev in range(MAX_LEVEL):

        if (np.sum(active_bool) == 0):
            break

        # morton codes for lev_curr
        active_inds = np.where(active_bool)[0]

        anchors_points = morton.points_to_anchors(XX[active_inds],lev,c0,L0,ndim=ndim)
        keys_points    = morton.anchors_to_keys(anchors_points,ndim=ndim)
        sorted_perm    = np.argsort(keys_points)

        unique,indices,counts = np.unique(keys_points[sorted_perm],return_counts=True,return_index=True)
        indices = np.concatenate(( indices, np.array([np.sum(active_bool)]) ))

        nleaf_lev = 0
        for j in range(unique.shape[0]):
            if (counts[j] <= max_bs):
                tmp_inds = sorted_perm[indices[j] : indices[j+1] ]
                active_bool[active_inds [tmp_inds]] = 0
                nleaf_lev += 1

        leaf_keys[acc_leaves : acc_leaves + nleaf_lev] = unique[counts <= max_bs]
        acc_leaves += nleaf_lev

    leaf_keys = leaf_keys[:acc_leaves]
    return leaf_keys

def _balance_tree(XX,c0,L0,leaf_keys):
    """
    Balances a tree structure by ensuring that the tree satisfies the 2:1 balance constraint on the leaves.
    The initial construction of a tree may contain coarse leaves which neighbor leaves much finer leaves,
    creating possibly large set of near-neigbor boxes. This procedure will refine these coarse leaf boxes
    so that two leaves which are adjacent are within one level of each other.

    Parameters:
    - XX: A 2D NumPy array of shape (N, ndim), where N is the number of points and `ndim` is the number of dimensions
          (either 2 or 3). This array contains the coordinates of the points used for partitioning.
    - c0: A float or array-like object specifying the origin (or center) used for computing Morton codes.
    - L0: A float specifying the length of the bounding box used for computing Morton codes.
    - leaf_keys: A 1D NumPy array of integers containing the Morton keys of the initial leaf nodes in the tree.

    Returns:
    - balanced_leaf_keys: A 1D NumPy array of integers containing the Morton keys of the balanced leaf nodes in the tree.
                          Each key corresponds to a unique leaf containing a subset of the input points `XX`.

    """

    ndim = XX.shape[-1]

    depth = np.max(morton.get_level(leaf_keys))
    leaf_keys = set(leaf_keys)
    for lev in range(depth,1,-1):

        tmp = np.array(list(leaf_keys),dtype=int)
        keys_lev = tmp[morton.get_level(tmp) == lev]
        if (keys_lev.shape[0] == 0):
            continue

        colleagues_vec  = morton.get_colleagues_vec(keys_lev,ndim=ndim)
        colleagues      = colleagues_vec[colleagues_vec > 0]

        parents = morton.get_parent(colleagues,ndim=ndim)
        sibs    = morton.get_siblings_vec(parents,ndim=ndim)

        for j in range(colleagues.shape[0]):
            if (not (colleagues[j] in leaf_keys) and not (parents[j] in leaf_keys)):
                leaf_keys.update(sibs[j])

    # process may have introduced redundant ancestor nodes
    # traverse from bottom level again and remove ancestors
    for lev in range(depth,0,-1):

        tmp = np.array(list(leaf_keys),dtype=int)
        keys_lev = tmp[morton.get_level(tmp) == lev]
        if (keys_lev.shape[0] == 0):
            continue

        anchors_points = morton.points_to_anchors(XX,lev,c0,L0,ndim=ndim)
        keys_points    = set(morton.anchors_to_keys(anchors_points,ndim=ndim))

        ancestors = morton.get_ancestors_vec(keys_lev,ndim=ndim)
        leaf_keys.difference_update(set(ancestors.flatten()))

        diff = set(keys_lev).difference(keys_points)
        leaf_keys -= diff

    return np.array(list(leaf_keys),dtype=int)

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
    - leaf_keys: A sorted 1D NumPy array of integers containing the Morton keys of the balanced leaf nodes.
    """

    tic = time()
    unbalanced_keys = _unbalanced_tree(XX,c0,L0,max_bs)
    toc_unbalanced = time() - tic

    tic = time()
    lev = morton.get_level(unbalanced_keys)
    if ( (np.max(lev) - np.min(lev)) > 1):
        leaf_keys  = _balance_tree(XX,c0,L0,unbalanced_keys.copy())
    else:
        leaf_keys = unbalanced_keys.copy()
    toc_balance = time() - tic
    return np.sort(unbalanced_keys),np.sort(leaf_keys)

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


    depth = np.max(morton.get_level(leaf_keys))
    tree_keys = set(leaf_keys.copy())

    for leaf in leaf_keys:
        tree_keys.update(set(morton.get_ancestors(leaf,ndim=ndim)))
    return np.sort(np.array(list(tree_keys),dtype=int))

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

        if ((c0 is None) and (L0 is None)):
            c0,L0          = morton.get_root_params(XX)
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

        tic = time()
        unbalanced_keys,leaf_keys = _get_leaf_keys(self.XX,self.c0,self.L0,leaf_size)
        toc_leaf = time() - tic

        tic = time()
        tree_keys = _get_complete_tree(leaf_keys,ndim=self.ndim)
        toc_complete = time() - tic

        nboxes    = tree_keys.shape[0]
        depth     = np.max(morton.get_level(leaf_keys))
        lev       = morton.get_level(tree_keys)

        _,level_sep = np.unique(lev,return_index=True)
        level_sep   = np.concatenate(( level_sep, np.array([nboxes],dtype=int) ))

        self.unbalanced_keys = unbalanced_keys # only for plotting purposes
        self.leaf_keys       = leaf_keys

        self.tree_keys       = tree_keys
        self.level_sep       = level_sep

        tic = time()
        self.perm_leaf,self.leaf_sep = self._get_leaf_sep()
        toc_sep = time() - tic

        tic = time()
        self.Clist_tmp, self.Llist_tmp = self._get_CL_lists_vec()
        toc_lists = time() - tic


    def _get_leaf_sep(self):
        """
        Generates a permutation vector that groups indices belonging to the same leaf together and
        computes separation indices for the permutation.

        Returns:
        - perm_leaf: A permutation vector of indices such that indices for the same leaf are grouped together.
        - leaf_sep: An array indicating the separation indices for the permutation.
        """

        lev_leaf = morton.get_level(self.leaf_keys)
        min_lev = np.min(lev_leaf); max_lev = np.max(lev_leaf)

        nleaf = self.leaf_keys.shape[0]

        perm_leaf  = np.zeros(self.N,dtype=int);
        leaf_sep   = np.zeros(nleaf+1,dtype=int)
        leaf_count = 0

        for lev in range(min_lev,max_lev+1):
            anchors_points = morton.points_to_anchors(self.XX,lev,self.c0,self.L0,ndim=self.ndim)
            keys_points    = morton.anchors_to_keys(anchors_points,ndim=self.ndim)
            sorted_perm_lev = np.argsort(keys_points)

            unique_keys,indices = np.unique(keys_points[sorted_perm_lev],return_index=True)
            indices = np.concatenate(( indices, np.array([self.N]) ))

            for j,key in enumerate(unique_keys):
                if (in_sorted(self.leaf_keys,key)):

                    inds_loc = sorted_perm_lev[indices[j]:indices[j+1]]
                    bnd1 = leaf_sep[leaf_count]
                    bnd2 = leaf_sep[leaf_count]+inds_loc.shape[0]
                    perm_leaf[bnd1 : bnd2] = inds_loc
                    leaf_count += 1; leaf_sep[leaf_count] = bnd2

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

    def get_box_colleague_neigh(self,box):

        box_list = self.Clist_tmp[box]
        return box_list[box_list >= 0]

    def get_box_coarse_neigh(self,box):

        box_list = self.Llist_tmp[box]
        return np.unique(box_list[box_list >= 0])

    def get_box_center(self,box):
        key = self.tree_keys[box]

        return morton.get_key_params(key,self.c0,self.L0)[0]

    def get_box_length(self,box):
        key = self.tree_keys[box]
        return morton.get_key_params(key,self.c0,self.L0)[1]

    def get_leaves(self):
        return findin_sorted_vec(self.tree_keys,self.leaf_keys)

    def get_boxes_level(self,lev):
        return np.arange(self.level_sep[lev],self.level_sep[lev+1])

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
