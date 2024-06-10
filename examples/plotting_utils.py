import matplotlib.patches as patches
from simpletree import BalancedTree,morton

import matplotlib.pyplot as plt

def add_patches(ax,tree,box_list,keys=False,\
                edgecolor='black',facecolor='white',alpha=0.7,linewidth=2.0,text_label=False,fontsize=32):
    for box in box_list:
        if (keys):
            c,L = morton.get_key_params(box,tree.c0,tree.L0)
        else:
            c = tree.get_box_center(box)
            L = tree.get_box_length(box)
        rect = patches.Rectangle((c[0]-L/2, c[1]-L/2), L, L, \
                             linewidth=linewidth, edgecolor=edgecolor, \
                                 facecolor=facecolor,alpha=alpha)
        if (not keys and text_label):
            ax.text(c[0],c[1],"$\\mathcal{B}_{%d}$" % (box),horizontalalignment='center',fontsize=fontsize)
        ax.add_patch(rect)
