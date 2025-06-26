from simpletree import BalancedTree, get_point_dist
import numpy as np
from plotting_utils import *
from matplotlib.patches import Patch


N = int(3e3)
np.random.seed(0)

XX = get_point_dist(N,'curvy_annulus')
tree = BalancedTree(XX)

fig,ax = plt.subplots()

leaf_keys   = tree.leaf_keys

##################################################################################
################    Example usage for box_inds and neigh_inds    ################

box = 60

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

add_patches(ax,tree,np.setdiff1d(tree.get_box_neighbors(box),np.array([box])),\
                                 edgecolor='tab:green',text_label=False,fontsize=14)

interaction_list = []
neigh_parent = tree.get_box_neighbors(tree.get_box_parent(box))
neigh_box    = tree.get_box_neighbors(box)
for neip in neigh_parent:
	for c in tree.get_box_children(neip):
		if not (c in neigh_box):
			interaction_list += [c]

Wlist = []
Xlist = []

if ( tree.is_leaf(box) ):

	for neigh in neigh_box:
		for c in tree.get_box_children(neigh):
			if not (c in neigh_box) and tree.get_box_parent(c) in neigh_box:
				Wlist += [c]

for neip in neigh_parent:

	if not (neip in neigh_box) and tree.is_leaf(neip):
		Xlist += [neip]

add_patches(ax,tree,np.setdiff1d(tree.get_box_neighbors(box),np.array([box])),\
                                 edgecolor='tab:green',text_label=False,fontsize=14)

add_patches(ax,tree,np.array(interaction_list),\
	edgecolor='tab:purple',text_label=False,fontsize=14)

add_patches(ax,tree,np.array(Wlist),\
	edgecolor='tab:orange',text_label=False,fontsize=14)

add_patches(ax,tree,np.array(Xlist),\
	edgecolor='tab:orange',text_label=False,fontsize=14)

add_patches(ax,tree,np.array([box]),edgecolor='black',\
            text_label=False,fontsize=14)

# Add legend with box patches

legend_elements = [
    Patch(facecolor='black', edgecolor='black', label='Box'),
    Patch(facecolor='tab:green', edgecolor='tab:green', label='Neighbor List'),
    Patch(facecolor='tab:purple', edgecolor='tab:purple', label='Interaction List')
]

if (len(Wlist) > 0):
	legend_elements += [Patch(facecolor='tab:orange', edgecolor='tab:orange', \
		label='W-List')]
elif (len(Xlist) > 0):
	legend_elements += [Patch(facecolor='tab:orange', edgecolor='tab:orange', \
		label='X-List')]

ax.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.05), 
          fancybox=True, shadow=False, ncol=len(legend_elements), frameon=False)


ax.axis('off')

##################################################################################

plt.gca().set_aspect('equal', adjustable='box')
plt.savefig("examples/interaction_list_curvy_annulus.png",bbox_inches='tight')
