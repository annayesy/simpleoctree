from simpletree import BalancedTree, get_point_dist
import numpy as np
from plotting_utils import *

N = int(3e3)
np.random.seed(0)

XX = get_point_dist(N,'curvy_annulus')
tree = BalancedTree(XX)

fig,ax = plt.subplots()

leaf_keys   = tree.leaf_keys

##################################################################################
################    Example usage for box_inds and neigh_inds    ################

box = 68

I_B = tree.get_box_inds(box)
I_N = tree.get_neigh_inds(box)

c = tree.get_box_center(box)
L = tree.get_box_length(box)

ax.scatter(XX[:,0], XX[:,1],s=1,color='tab:blue')

ax.scatter(XX[I_B,0],XX[I_B,1],s=1,color='black')
ax.scatter(XX[I_N,0],XX[I_N,1],s=1,color='tab:green')

##################################################################################
################################ Plotting utils   ################################

add_patches(ax,tree,leaf_keys,keys=True,\
            edgecolor='tab:gray',facecolor='none',linewidth=1.0,alpha=0.5)

add_patches(ax,tree,np.setdiff1d(tree.get_box_colleague_neigh(box),np.array([box])),\
                                 edgecolor='tab:green',text_label=False,fontsize=14)
add_patches(ax,tree,tree.get_box_coarse_neigh(box),edgecolor='tab:green',\
            text_label=False,fontsize=14)
add_patches(ax,tree,np.array([box]),edgecolor='black',\
            text_label=False,fontsize=14)

##################################################################################

plt.gca().set_aspect('equal', adjustable='box')
plt.savefig("examples/curvy_annulus.png",bbox_inches='tight')
