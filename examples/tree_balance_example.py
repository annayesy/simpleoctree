from simpletree import BalancedTree, get_point_dist
import numpy as np
from plotting_utils import *

N = int(2e3)
np.random.seed(0)

XX = get_point_dist(N,'curvy_annulus')
tree = BalancedTree(XX,leaf_size=20)

fig,(ax0,ax1) = plt.subplots(1,2)

unbalanced_keys = tree.unbalanced_keys
balanced_keys   = tree.leaf_keys

deleted_keys    = np.setdiff1d(unbalanced_keys,balanced_keys)
unmod_keys      = np.intersect1d(balanced_keys,unbalanced_keys)
added_keys      = np.setdiff1d(balanced_keys,unmod_keys)

#############################################################################
# This subplot shows a leaf box that adjacent to a leaf box two levels below.

ax0.scatter(XX[:,0], XX[:,1],s=5,alpha=0.5)

add_patches(ax0,tree,unmod_keys,keys=True,\
            edgecolor='tab:gray',facecolor='none',linewidth=1.0,alpha=0.8)

print("%d leaves violate the balance constraint" % (deleted_keys.shape[0]))

add_patches(ax0,tree,deleted_keys,keys=True,\
            edgecolor='tab:red',facecolor='tab:red',linewidth=1.0,alpha=0.5)

ax0.set_xlim([0.75,1.03])
ax0.set_ylim([0.5,0.78])

ax0.set_aspect('equal', adjustable='box')

#############################################################################
# This subplot shows how the leaf is refined in order to satisfy the 2:1
# balance constraint.

ax1.scatter(XX[:,0], XX[:,1],s=5,alpha=0.5)

add_patches(ax1,tree,unmod_keys,keys=True,\
            edgecolor='tab:gray',facecolor='none',linewidth=1.0,alpha=0.8)

add_patches(ax1,tree,added_keys,keys=True,\
            edgecolor='tab:green',facecolor='tab:green',linewidth=1.0,alpha=0.5)
ax1.set_aspect('equal', adjustable='box')
ax1.set_xlim([0.75,1.03])
ax1.set_ylim([0.5,0.78])

plt.savefig("examples/tree_balance.png",bbox_inches='tight')
