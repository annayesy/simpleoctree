from simpletree import BalancedTree, get_point_dist
import numpy as np
from plotting_utils import *

np.random.seed(0)  # Fix random seed for reproducibility

ntot = 1000  # Total number of points

XX = get_point_dist(ntot,'curvy_annulus')
tree = BalancedTree(XX,leaf_size=40)

fig,ax = plt.subplots(figsize=(8,8))

ax.scatter(XX[:,0], XX[:,1])

boxes = [12,23]

for box in boxes:

    add_patches(ax,tree,np.array([box]),edgecolor='green',\
                text_label=False,fontsize=14)


for box in tree.get_leaves():

    add_patches(ax,tree,np.array([box]),edgecolor='black',\
                text_label=True,fontsize=14)

ax.set_aspect('equal', adjustable='box')
plt.tight_layout()
plt.show()
