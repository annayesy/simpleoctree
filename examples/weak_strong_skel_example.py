from simpletree import BalancedTree, get_point_dist
import numpy as np
from plotting_utils import *
from scipy.linalg.interpolative import interp_decomp
from scipy.spatial.distance     import cdist 

plt.rc('text',usetex=True)
plt.rc('font',family='serif')
plt.rc('text.latex',preamble=r'\usepackage{amsfonts,bm}')

def get_near_neigh(box_list):
    neigh_list = [list(tree.get_box_neighbors(box)) for box in box_list]
    neigh_list = np.array(list({x for l in neigh_list for x in l}))
    return neigh_list

#####################################################################################

N = int(3e3)
np.random.seed(0)

XX = get_point_dist(N,'curvy_annulus')
tree = BalancedTree(XX)

box = 35

near_neigh  = get_near_neigh(tree.get_box_neighbors(box))
neigh_plus  = get_near_neigh(near_neigh)
leaf_boxes  = get_near_neigh(neigh_plus)

I_B = tree.get_box_inds(box)
I_N = tree.get_neigh_inds(box)

c = tree.get_box_center(box)
L = tree.get_box_length(box)

s = 5

I_Fstrong = np.setdiff1d(np.arange(tree.N),I_N)
I_Fweak   = np.setdiff1d(np.arange(tree.N),I_B)

A_fb_strong = np.log(cdist(XX[I_Fstrong],XX[I_B]))
A_fb_weak   = np.log(cdist(XX[I_Fweak]  ,XX[I_B]))

d_weak   = np.linalg.svd(A_fb_weak,compute_uv=False)
d_strong = np.linalg.svd(A_fb_strong,compute_uv=False)
d_weak  /= d_weak[0]
d_strong/= d_strong[0]

k_weak  = np.sum(d_weak   > 1e-5)-1
k_strong= np.sum(d_strong > 1e-5)-1


#####################################################################################
####################################  weak skel  ####################################

fig,ax = plt.subplots()

ax.scatter(XX[:,0], XX[:,1],s=s,color='tab:gray')
ax.scatter(XX[I_B,0],XX[I_B,1],s=s,color='black')

add_patches(ax,tree,leaf_boxes,text_label=False,\
    edgecolor='tab:gray',facecolor='none',linewidth=1.0,alpha=0.5)
add_patches(ax,tree,np.array([box]),edgecolor='blue',\
    alpha=0.2, linewidth=4)

idx,proj = interp_decomp(A_fb_weak,k_weak)
I_S        = I_B[idx[:k_weak]]
ax.scatter(XX[I_S,0],XX[I_S,1],s=4,color='red')

ax.text(c[0]-0.5*L,c[1]+1.75*L,"$k={%d}$" % (k_weak),horizontalalignment='center',fontsize=20)
ax.text(c[0],c[1]+0.2*L,"$\\sf{B}$",horizontalalignment='center',fontsize=20)
ax.text(c[0]+0.80*L,c[1]-0.35*L,"$\\sf{F}_{\\rm weak}$",horizontalalignment='left',fontsize=20)

ax.set_xlim([c[0]-3.00*L,c[0]+3.00*L])
ax.set_ylim([c[1]-0.50*L,c[1]+2.50*L])

ax.axis('off')

plt.gca().set_aspect('equal', adjustable='box')
plt.savefig("examples/curvy_annulus_weak.pdf",bbox_inches='tight')

##################################################################################
################################### strong skel ##################################

fig,ax = plt.subplots()

ax.scatter(XX[:,0], XX[:,1],s=s,color='tab:gray')
ax.scatter(XX[I_N,0],XX[I_N,1],s=s,color='tab:green')
ax.scatter(XX[I_B,0],XX[I_B,1],s=s,color='black')

add_patches(ax,tree,leaf_boxes,text_label=False,\
    edgecolor='tab:gray',facecolor='none',linewidth=1.0,alpha=0.5)
add_patches(ax,tree,np.array([box]),edgecolor='blue',alpha=0.2, linewidth=4)
add_patches(ax,tree,tree.get_box_neighbors(box)[1:],\
    edgecolor='tab:green',alpha=0.2,linewidth=4)

idx,proj = interp_decomp(A_fb_strong,k_strong)
I_S        = I_B[idx[:k_strong]]
ax.scatter(XX[I_S,0],XX[I_S,1],s=4,color='red')

ax.text(c[0]-0.5*L,c[1]+1.75*L,"$k={%d}$" % (k_strong),horizontalalignment='center',fontsize=20)

ax.text(c[0],c[1]+0.2*L,"$\\sf{B}$",horizontalalignment='center',fontsize=20)
ax.text(c[0]+L,c[1]-0.35*L,"$\\sf{N}$",horizontalalignment='center',fontsize=20)
ax.text(c[0]+2.0*L,c[1]+1.60*L,"$\\sf{F}_{\\rm strong}$",horizontalalignment='center',fontsize=20)

ax.set_xlim([c[0]-3.00*L,c[0]+3.00*L])
ax.set_ylim([c[1]-0.50*L,c[1]+2.50*L])

ax.axis('off')

plt.gca().set_aspect('equal', adjustable='box')
plt.savefig("examples/curvy_annulus_strong.pdf",bbox_inches='tight')

##################################################################################

plt.figure()
fig,ax = plt.subplots()

cutoff   = 55; fontsize=20
plt.semilogy(d_weak[:cutoff],color='black',linestyle='--')
plt.semilogy(d_strong[:cutoff],color='black',linestyle='-')

k_weak  = np.sum(d_weak   > 1e-5)-1
k_strong= np.sum(d_strong > 1e-5)-1
ax.set_ylim([1e-18,2])
ax.set_xlim([-1,cutoff])

plt.plot(np.array([k_weak,k_weak]),    np.array([1e-18,1e-5]),linestyle='-',color='tab:blue')
plt.plot(np.array([k_strong,k_strong]),np.array([1e-18,1e-5]),linestyle='-',color='tab:blue')

ax.text(k_weak+1,3e-18,"$k={%d}$" % (k_weak),horizontalalignment='left',fontsize=fontsize)
ax.text(k_strong+1,3e-18,"$k={%d}$" % (k_strong),horizontalalignment='left',fontsize=fontsize)

ax.text(k_weak  +2.0,1e-5,"weak" % (k_weak),horizontalalignment='left',fontsize=fontsize)
ax.text(k_weak  +5.5,8e-7,"admissibility" % (k_weak),horizontalalignment='left',fontsize=fontsize)
ax.text(k_strong+1.0,1e-5,"strong" % (k_strong),horizontalalignment='left',fontsize=fontsize)
ax.text(k_strong+2.7,6e-7,"admissibility" % (k_weak),horizontalalignment='left',fontsize=fontsize)

plt.savefig("examples/curvy_annulus_spectra.pdf",bbox_inches='tight')