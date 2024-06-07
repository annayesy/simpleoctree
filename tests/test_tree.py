import pytest
import numpy as np
from pytreelib import get_point_dist, BalancedTree

np.random.seed(0)

def acc_inds(tree,boxes):
    inds = np.zeros(tree.N,dtype=int); acc = 0
    for box in boxes:
        tmp = tree.get_box_inds(box)
        inds[acc : acc + tmp.shape[0]] = tmp
        acc += tmp.shape[0]
    return inds[:acc]

def check_points_in_bounds(XX_B,c,boxlen):
    tmp = np.linalg.norm(XX_B - c,axis=1,ord=np.inf) < boxlen*0.5
    assert np.all(tmp)

def check_points_outof_bounds(XX_B,c,boxlen):
    tmp = np.linalg.norm(XX_B - c,axis=1,ord=np.inf) >= boxlen*0.5
    assert np.all(tmp)

def check_children(tree,box):
    box_inds = tree.get_box_inds(box)

    child_inds = acc_inds(tree,tree.get_box_children(box))

    assert child_inds.shape[0] == box_inds.shape[0]
    assert np.linalg.norm( (np.sort(box_inds) - np.sort(child_inds)).astype(float) ) < 1e-14

def check_neighbors(tree,box):
    XX  = tree.XX
    I_B = tree.get_box_inds(box)

    neigh_boxes = np.hstack((tree.get_box_colleague_neigh(box),\
                                         tree.get_box_coarse_neigh(box)))

    I_N = acc_inds(tree,neigh_boxes )

    I_F = np.setdiff1d(np.arange(tree.N),I_N)

    c = tree.get_box_center(box)
    L = tree.get_box_length(box)
    check_points_outof_bounds(XX[I_F],c,3*L)
    return I_N,I_F

def tree_check(tree):

    XX = tree.XX
    for box in range(tree.level_sep[2],tree.nboxes):
        c = tree.get_box_center(box)
        L = tree.get_box_length(box)

        I_B = tree.get_box_inds(box)
        check_points_in_bounds(XX[I_B],c,L)
        if (not tree.is_leaf(box)):
            check_children(tree,box)
        I_N,I_F = check_neighbors(tree,box)

def test_square():

    N = int(1e4)
    XX = get_point_dist(N,'square')
    tree = BalancedTree(XX,leaf_size=50)
    tree_check(tree)

def test_curvy_annulus():

    N = int(1e4)
    XX = get_point_dist(N,'curvy_annulus')
    tree = BalancedTree(XX,leaf_size=50)
    tree_check(tree)

def test_cube():

    N = int(1e4)
    XX = get_point_dist(N,'cube')
    tree = BalancedTree(XX,leaf_size=50)
    tree_check(tree)
