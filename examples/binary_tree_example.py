from simpletree import BinaryTree, get_point_dist
import numpy as np
from plotting_utils import *

N = int(3e3)
np.random.seed(0)

XX = get_point_dist(N,'curvy_annulus')
tree = BinaryTree(XX)

for b in range(tree.nboxes):
    children = tree.get_box_children(b)
    assert len(children) <= 2

##################################################################################
################    Example usage for box_inds and neigh_inds    ################

box = 0

children = tree.get_box_children(box)

if (len(children) == 2):

    fig,ax = plt.subplots()

    I_C0    = tree.get_box_inds(children[0])
    I_C1    = tree.get_box_inds(children[1])

    ax.scatter(XX[:,0], XX[:,1],s=1,color='tab:gray')
    ax.scatter(XX[I_C0,0],XX[I_C0,1],s=1,color='tab:orange')
    ax.scatter(XX[I_C1,0],XX[I_C1,1],s=1,color='tab:green')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title("Children (orange,green) of box %d" % box)
    plt.show()
