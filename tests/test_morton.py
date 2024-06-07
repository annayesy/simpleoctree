import pytest
import numpy as np
from pytreelib import morton, morton_tree

np.random.seed(0)

def test_morton_encode_helper():

    # 2d test
    x = np.arange(2**morton.MAX_LEVEL2)
    code = morton.encode_helper2d(x,morton.mortonLUT_encode2d)
    tmp  = morton.decode_helper2d(code,morton.mortonLUT_decode2d)
    assert np.all(x == tmp)

    # 3d test
    x = np.arange(2**morton.MAX_LEVEL3)

    code = morton.encode_helper3d(x,morton.mortonLUT_encode3d)
    tmp  = morton.decode_helper3d(code,morton.mortonLUT_decode3d)
    assert np.all(x == tmp)


def test_convert2d():

    N = int(1e5)
    XX = np.random.rand(N,2)

    c0,L0 = morton.get_root_params(XX)

    for lev in range(morton.MAX_LEVEL2):
        anchors = morton.points_to_anchors(XX,lev,c0,L0,ndim=2)
        keys    = morton.anchors_to_keys(anchors,ndim=2)

        anchors_orig = morton.keys_to_anchors(keys,ndim=2)
        err = anchors_orig - anchors
        assert np.linalg.norm(err.astype(float)) == 0

def test_convert3d():

    N = int(1e5)
    XX = np.random.rand(N,3)

    c0,L0 = morton.get_root_params(XX)

    for lev in range(morton.MAX_LEVEL3):
        anchors = morton.points_to_anchors(XX,lev,c0,L0,ndim=3)
        keys    = morton.anchors_to_keys(anchors,ndim=3)

        anchors_orig = morton.keys_to_anchors(keys,ndim=3)

        err = anchors_orig - anchors

        assert np.linalg.norm(err.astype(float)) == 0


def test_parent2d():

    anchors = np.array([[5,4,4],\
                        [15,12,4]],dtype=int)

    parent_anchors = np.array([[2,2,3],\
                               [7,6,3]],dtype=int)

    parent_keys = morton.get_parent(morton.anchors_to_keys(anchors,ndim=2),ndim=2)

    err = morton.keys_to_anchors(parent_keys,ndim=2) - parent_anchors

    assert np.linalg.norm(err.astype(float)) == 0


def test_parent3d():

    anchors = np.array([[5,4,3,4],\
                        [15,12,10,4]],dtype=int)

    parent_anchors = np.array([[2,2,1,3],\
                               [7,6,5,3]],dtype=int)

    parent_keys = morton.get_parent(morton.anchors_to_keys(anchors,ndim=3),ndim=3)

    err = morton.keys_to_anchors(parent_keys,ndim=3) - parent_anchors

    assert np.linalg.norm(err.astype(float)) == 0

def test_ancestors2d():

    anchors = np.array([[15,12,4]],dtype=int)
    ancestor_anchors = np.array([[0,0,0],\
                                 [1,1,1],\
                                 [3,3,2],\
                                 [7,6,3]])

    key = morton.anchors_to_keys(anchors,ndim=2)[0]
    ancestor_keys = morton.anchors_to_keys(ancestor_anchors,ndim=2)
    err = morton.get_ancestors(key,ndim=2) - ancestor_keys

    assert np.linalg.norm(err.astype(float)) == 0

def test_ancestors3d():

    anchors = np.array([[15,12,10,4]],dtype=int)
    ancestor_anchors = np.array([[0,0,0,0],\
                                 [1,1,1,1],\
                                 [3,3,2,2],\
                                 [7,6,5,3]])

    key = morton.anchors_to_keys(anchors,ndim=2)[0]
    ancestor_keys = morton.anchors_to_keys(ancestor_anchors,ndim=2)
    err = morton.get_ancestors(key,ndim=2) - ancestor_keys

    assert np.linalg.norm(err.astype(float)) == 0

def test_siblings2d():

    anchor = np.array([5,3,4],dtype=int)

    sibling_anchors = np.array([[4,2,4],\
                                [4,3,4],\
                                [5,2,4],\
                                [5,3,4]],dtype=int)

    key = morton.anchors_to_keys(anchor,ndim=2)[0]

    sibling_keys = morton.get_siblings(key,ndim=2)

    err = morton.keys_to_anchors(sibling_keys,ndim=2) - sibling_anchors
    assert np.linalg.norm(err.astype(float)) == 0

    anchors = np.array([[5,3,4],[9,2,4]],dtype=int)
    keys = morton.anchors_to_keys(anchors,ndim=2)
    sibling_keys = morton.get_siblings_vec(keys,ndim=2)

    for j in range(2):
        err = morton.get_siblings(keys[0],ndim=2) - sibling_keys[0]
        assert np.linalg.norm(err.astype(float)) == 0

def test_siblings3d():

    anchor = np.array([5,3,2,4])

    sibling_anchors = np.array([[4,2,2,4],\
                                [4,2,3,4],\
                                [4,3,2,4],\
                                [4,3,3,4],\
                                [5,2,2,4],\
                                [5,2,3,4],\
                                [5,3,2,4],\
                                [5,3,3,4]],dtype=int)

    key = morton.anchors_to_keys(anchor,ndim=3)[0]

    sibling_keys = morton.get_siblings(key,ndim=3)

    print(morton.keys_to_anchors(sibling_keys,ndim=3))

    err = morton.keys_to_anchors(sibling_keys,ndim=3) - sibling_anchors
    assert np.linalg.norm(err.astype(float)) == 0

def test_children2d():

    anchor = np.array([5,3,4])

    child_anchors = np.array([[10,6,5],\
                              [10,7,5],\
                              [11,6,5],\
                              [11,7,5]],dtype=int)

    key = morton.anchors_to_keys(anchor,ndim=2)[0]

    child_keys = morton.get_children(key,ndim=2)

    err = morton.keys_to_anchors(child_keys,ndim=2) - child_anchors
    assert np.linalg.norm(err.astype(float)) == 0

def test_children3d():

    anchor = np.array([5,3,2,4])

    child_anchors = np.array([[10,6,4,5],\
                              [10,6,5,5],\
                              [10,7,4,5],\
                              [10,7,5,5],\
                              [11,6,4,5],\
                              [11,6,5,5],\
                              [11,7,4,5],\
                              [11,7,5,5],\
                             ],dtype=int)

    key = morton.anchors_to_keys(anchor,ndim=3)[0]

    child_keys = morton.get_children(key,ndim=3)

    err = morton.keys_to_anchors(child_keys,ndim=3) - child_anchors
    assert np.linalg.norm(err.astype(float)) == 0

def test_increment2d():

    anchors = np.array([[6,7,5],[30,35,6]],dtype=int)
    keys = morton.anchors_to_keys(anchors,ndim=2)

    inc_anchors = morton.keys_to_anchors(morton.increment_x(keys,ndim=2),ndim=2)
    tmp_anchors = anchors.copy(); tmp_anchors[:,0] += 1

    err = inc_anchors - tmp_anchors

    assert np.linalg.norm(err.astype(float)) == 0

    inc_anchors = morton.keys_to_anchors(morton.increment_y(keys,ndim=2),ndim=2)
    tmp_anchors = anchors.copy(); tmp_anchors[:,1] += 1

    err = inc_anchors - tmp_anchors

    assert np.linalg.norm(err.astype(float)) == 0

def test_increment3d():

    anchors = np.array([[6,7,8,5],[30,35,9,6]],dtype=int)
    keys = morton.anchors_to_keys(anchors,ndim=3)

    inc_anchors = morton.keys_to_anchors(morton.increment_x(keys,ndim=3),ndim=3)
    tmp_anchors = anchors.copy(); tmp_anchors[:,0] += 1

    err = inc_anchors - tmp_anchors

    assert np.linalg.norm(err.astype(float)) == 0

    inc_anchors = morton.keys_to_anchors(morton.increment_y(keys,ndim=3),ndim=3)
    tmp_anchors = anchors.copy(); tmp_anchors[:,1] += 1

    err = inc_anchors - tmp_anchors

    assert np.linalg.norm(err.astype(float)) == 0

    inc_anchors = morton.keys_to_anchors(morton.increment_z(keys,ndim=3),ndim=3)
    tmp_anchors = anchors.copy(); tmp_anchors[:,2] += 1

    err = inc_anchors - tmp_anchors

    assert np.linalg.norm(err.astype(float)) == 0

def test_decrement2d():

    anchors = np.array([[6,7,5],[30,35,6]],dtype=int)
    keys = morton.anchors_to_keys(anchors,ndim=2)

    dec_anchors = morton.keys_to_anchors(morton.decrement_x(keys,ndim=2),ndim=2)
    tmp_anchors = anchors.copy(); tmp_anchors[:,0] -= 1

    err = dec_anchors - tmp_anchors

    assert np.linalg.norm(err.astype(float)) == 0

    dec_anchors = morton.keys_to_anchors(morton.decrement_y(keys,ndim=2),ndim=2)
    tmp_anchors = anchors.copy(); tmp_anchors[:,1] -= 1

    err = dec_anchors - tmp_anchors

    assert np.linalg.norm(err.astype(float)) == 0

def test_decrement3d():

    anchors = np.array([[6,7,8,5],[30,35,9,6]],dtype=int)
    keys = morton.anchors_to_keys(anchors,ndim=3)

    inc_anchors = morton.keys_to_anchors(morton.decrement_x(keys,ndim=3),ndim=3)
    tmp_anchors = anchors.copy(); tmp_anchors[:,0] -= 1

    err = inc_anchors - tmp_anchors

    assert np.linalg.norm(err.astype(float)) == 0

    inc_anchors = morton.keys_to_anchors(morton.decrement_y(keys,ndim=3),ndim=3)
    tmp_anchors = anchors.copy(); tmp_anchors[:,1] -= 1

    err = inc_anchors - tmp_anchors

    assert np.linalg.norm(err.astype(float)) == 0

    inc_anchors = morton.keys_to_anchors(morton.decrement_z(keys,ndim=3),ndim=3)
    tmp_anchors = anchors.copy(); tmp_anchors[:,2] -= 1

    err = inc_anchors - tmp_anchors

    assert np.linalg.norm(err.astype(float)) == 0

def colleague_test_helper(anchors, invalid_shifts,ndim):

    if (ndim == 2):
        diff_anchors = np.array([[-1,-1, 0],[-1, 0, 0],[-1, 1, 0],\
                                 [ 0,-1, 0],[ 0, 0, 0],[ 0, 1, 0],\
                                 [ 1,-1, 0],[ 1, 0, 0],[ 1, 1, 0]])
    else:
        diff_anchors = np.array([[-1,-1,-1, 0],[-1,-1, 0, 0],[-1,-1,+1, 0],\
                                 [-1, 0,-1, 0],[-1, 0, 0, 0],[-1, 0,+1, 0],\
                                 [-1, 1,-1, 0],[-1, 1, 0, 0],[-1, 1,+1, 0],\
                                 [ 0,-1,-1, 0],[ 0,-1, 0, 0],[ 0,-1,+1, 0],\
                                 [ 0, 0,-1, 0],[ 0, 0, 0, 0],[ 0, 0,+1, 0],\
                                 [ 0, 1,-1, 0],[ 0, 1, 0, 0],[ 0, 1,+1, 0],\
                                 [ 1,-1,-1, 0],[ 1,-1, 0, 0],[ 1,-1,+1, 0],\
                                 [ 1, 0,-1, 0],[ 1, 0, 0, 0],[ 1, 0,+1, 0],\
                                 [ 1, 1,-1, 0],[ 1, 1, 0, 0],[ 1, 1,+1, 0]])
    colleague_anchors = anchors + diff_anchors
    colleague_keys = morton.get_colleagues_vec(morton.anchors_to_keys(anchors,ndim=ndim),ndim=ndim)

    key_check  = morton.anchors_to_keys(colleague_anchors,ndim=ndim)
    key_check[invalid_shifts] = -1

    err = colleague_keys - key_check
    assert np.linalg.norm(err.astype(float)) == 0

def test_neighbors_validshift2d():

    lev = 5
    anchors = np.array([[2,15,lev]])

    colleague_test_helper(anchors,np.array([],dtype=int),ndim=2)

def test_neighbors_validshift3d():

    lev = 5
    anchors = np.array([[2,15,8,lev]])

    colleague_test_helper(anchors,np.array([],dtype=int),ndim=3)

def test_neighbors_invalidshift2d():

    lev = 4

    anchors = np.array([[2,15,lev]])
    invalid_shifts = np.array([2,5,8],dtype=int)
    colleague_test_helper(anchors,invalid_shifts,ndim=2)

    anchors = np.array([[0,14,lev]])
    invalid_shifts = np.array([0,1,2],dtype=int)
    colleague_test_helper(anchors,invalid_shifts,ndim=2)

    anchors = np.array([[0,0,lev]])
    invalid_shifts = np.array([0,1,2,3,6],dtype=int)
    colleague_test_helper(anchors,invalid_shifts,ndim=2)

def test_neighbors_invalidshift3d():

    lev = 4

    anchors = np.array([[15,1,1,lev]])
    invalid_shifts = np.arange(9)+18
    colleague_test_helper(anchors,invalid_shifts,ndim=3)

    anchors = np.array([[1,15,1,lev]])
    invalid_shifts = np.array([0,1,2,9,10,11,18,19,20],dtype=int)+6
    colleague_test_helper(anchors,invalid_shifts,ndim=3)

    anchors = np.array([[1,1,15,lev]])
    invalid_shifts = np.arange(0,27,3) + 2
    colleague_test_helper(anchors,invalid_shifts,ndim=3)

    anchors = np.array([[0,0,0,lev]])
    invalid_shifts = np.hstack((np.arange(9),\
                                np.array([0,1,2,9,10,11,18,19,20],dtype=int),\
                                np.arange(0,27,3)))
    invalid_shifts = np.unique(invalid_shifts)

    colleague_test_helper(anchors,invalid_shifts,ndim=3)

def test_sortedsearch():

    keys = np.unique(np.random.rand(100)*int(1e6)+5);
    keys = np.sort(keys.astype(int))

    entries = keys[:5].copy(); entries[1] = 1

    result  = np.ones(5); result[1] = 0
    err = morton_tree.in_sorted_vec(keys,entries) - result
    assert np.linalg.norm(err.astype(float)) == 0

    result = np.arange(5); result[1] = -1
    err = morton_tree.findin_sorted_vec(keys,entries) - result
    assert np.linalg.norm(err.astype(float)) == 0

    entries = keys[:5].copy(); entries[1] = np.max(keys) + 5
    result  = np.arange(5); result[1] = -1
    err = morton_tree.findin_sorted_vec(keys,entries) - result
