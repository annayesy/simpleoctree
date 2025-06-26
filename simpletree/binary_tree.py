import numpy as np
from simpletree_dev.abstract_tree import AbstractTree
from simpletree_dev.morton_tree   import MortonTree

def childvec_helper(child_vec):

	if (child_vec.shape[0] == 4):
		[DL_exists,UL_exists,DR_exists,UR_exists] = child_vec > 0
		tmp = child_vec.reshape(2,2)
		L_exists = np.any(tmp[0]>=0)
		R_exists = np.any(tmp[1]>=0)

	else:
		#[BDL_exists,BUL_exists,BDR_exists,BUR_exists,\
		# FDL_exists,FUL_exists,FDR_exists,FUR_exists] = child_vec > 0
		tmp = child_vec.reshape(2,2,2)
		L_exists  = np.any(tmp[:,0,:]>=0)
		R_exists  = np.any(tmp[:,1,:]>=0)

		DL_exists = np.any(tmp[:,0,0]>=0)
		UL_exists = np.any(tmp[:,0,1]>=0)
		DR_exists = np.any(tmp[:,1,0]>=0)
		UR_exists = np.any(tmp[:,1,1]>=0)
	return L_exists,R_exists,\
	DL_exists,UL_exists,DR_exists,UR_exists

def get_nboxes_extra(mtree,lev):

	bin_offset = 0; quad_offset = 0; oct_offset = 0
	ndim       = mtree.ndim

	for mbox in mtree.get_boxes_level(lev):

		if (mtree.is_leaf(mbox)):
			continue

		child_vec = mtree.get_box_children_fullvec(mbox)

		L_exists,R_exists,\
		DL_exists,UL_exists,DR_exists,UR_exists = childvec_helper(child_vec)

		bin_offset  += int(L_exists) + int(R_exists)
		quad_offset += int(DL_exists)+ int(UL_exists) + int(DR_exists) + int(UR_exists)
		if (mtree.ndim  == 3):
			oct_offset += np.sum((child_vec > 0).astype(int))

	return bin_offset,quad_offset,oct_offset

def get_lev_offset(mtree):

	lev_offset   = np.zeros(mtree.nlevels * 2**(mtree.ndim -1) + 1, dtype=int)
	lev_offset[1]= 1
	nlevels      = 1

	for lev in range(mtree.nlevels):

		bin_offset,quad_offset,oct_offset = get_nboxes_extra(mtree,lev)

		if (bin_offset > 0):
			lev_offset[nlevels+1] = lev_offset[nlevels] + bin_offset
			nlevels += 1
		if (quad_offset > 0):
			lev_offset[nlevels+1] = lev_offset[nlevels] + quad_offset
			nlevels += 1
		if (oct_offset > 0):
			lev_offset[nlevels+1] = lev_offset[nlevels] + oct_offset
			nlevels += 1
	lev_offset = lev_offset[:nlevels+1]

	return lev_offset


class BinaryTree(AbstractTree):

	def __init__(self,XX,leaf_size = 100,c0=None,L0=None):

		self.XX   = XX
		self.ndim = XX.shape[-1]
		mtree = MortonTree(XX,leaf_size,c0,L0)

		lev_offset = get_lev_offset(mtree)

		nboxes = lev_offset[-1]

		# find what boxes in original tree map to binary tree
		# mbox_ref[new_box] gives a pointer to old box if applicable
		mbox_list    = np.ones(nboxes,dtype=int)     * (-1)
		mbox_list[0] = 0

		parent_list  = np.ones(nboxes,dtype=int)     * (-1)
		child_list   = np.ones((nboxes,2),dtype=int) * (-1)

		center_list  = np.zeros((nboxes,self.ndim))
		length_list  = np.zeros((nboxes,self.ndim))
		isleaf_list  = np.zeros(nboxes,dtype=bool)

		for lev in range(mtree.nlevels):

			if (lev < mtree.nlevels - 1):
				offset_lev0 = lev_offset[(self.ndim) * lev+1]
				offset_lev1 = lev_offset[(self.ndim) * lev+2]
			else:
				offset_lev0 = nboxes; offset_lev1 = nboxes

			if (self.ndim == 3 and lev < mtree.nlevels - 1):
				offset_lev2 = lev_offset[(self.ndim) * lev+3]
			else:
				offset_lev2 = nboxes

			for mbox in mtree.get_boxes_level(lev):

				bbox              = np.where(mbox_list == mbox)[0][0]

				mcenter           = mtree.get_box_center(mbox)
				Ldim              = mtree.get_box_length(mbox)
				center_list[bbox] = mcenter
				length_list[bbox] = Ldim

				if (mtree.is_leaf(mbox)):
					isleaf_list[bbox] = True
					continue

				child_vec = mtree.get_box_children_fullvec(mbox)

				L_exists,R_exists,\
				DL_exists,UL_exists,DR_exists,UR_exists = childvec_helper(child_vec)

				def shift_center_length(center,L,shift_array):

					new_center = center.copy(); new_L = L.copy()
					for (j,shift) in enumerate(shift_array):
						if (shift == 1):
							new_center[j] += L[j]/4
							new_L[j]       = L[j]/2
						elif (shift == -1):
							new_center[j] -= L[j]/4
							new_L[j]       = L[j]/2
					return new_center,new_L

				def assign_mbox_split_helper(child_vec_loc,offset_lev0_loc,offset_lev1_loc):

					tmp         = np.where(child_vec_loc > 0)[0]
					split_boxes = np.arange(tmp.shape[0]) + offset_lev1_loc

					child_list[offset_lev0_loc,tmp] = split_boxes
					parent_list[split_boxes]        = offset_lev0_loc

					mbox_list[split_boxes]         = child_vec_loc[tmp]
					return tmp.shape[0]

				if (L_exists):

					child_list [bbox,0]      = offset_lev0
					parent_list[offset_lev0] = bbox

					if (self.ndim == 2):
						center_list[offset_lev0],length_list[offset_lev0] = \
						shift_center_length( mcenter,Ldim,np.array([-1,0]) )

						#quad split
						offset_lev1 += assign_mbox_split_helper(child_vec[:2],offset_lev0,offset_lev1)
						offset_lev0 += 1

					else:
						center_list[offset_lev0],length_list[offset_lev0] = \
						shift_center_length(mcenter,Ldim,np.array([0,-1,0]) )

						# quad split
						if (DL_exists):

							child_list [offset_lev0,0] = offset_lev1
							parent_list[offset_lev1]   = offset_lev0

							center_list[offset_lev1],length_list[offset_lev1] = \
							shift_center_length( mcenter,Ldim,np.array([0,-1,-1]) )

							# assign mbox split helper here
							offset_lev2 += assign_mbox_split_helper(child_vec[np.array([0,4])],\
								offset_lev1,offset_lev2)

							offset_lev1 += 1

						if (UL_exists):
							child_list [offset_lev0,1] = offset_lev1
							parent_list[offset_lev1]   = offset_lev0

							center_list[offset_lev1],length_list[offset_lev1] = \
							shift_center_length( mcenter,Ldim,np.array([0,-1,+1]) )

							# assign mbox split helper here 
							offset_lev2 += assign_mbox_split_helper(child_vec[np.array([1,5])],\
								offset_lev1,offset_lev2)

							offset_lev1 += 1

						offset_lev0 += 1

				if (R_exists):

					child_list[bbox,1]       = offset_lev0
					parent_list[offset_lev0] = bbox

					if (self.ndim == 2):
						center_list[offset_lev0],length_list[offset_lev0] = \
						shift_center_length( mcenter,Ldim,np.array([+1,0]) )

						# quad split
						offset_lev1 += assign_mbox_split_helper(child_vec[2:],offset_lev0,offset_lev1)
						offset_lev0 += 1
					else:

						center_list[offset_lev0],length_list[offset_lev0] = \
						shift_center_length(mcenter,Ldim,np.array([0,+1,0]) )

						# quad split
						if (DR_exists):

							child_list [offset_lev0,0] = offset_lev1
							parent_list[offset_lev1]   = offset_lev0

							center_list[offset_lev1],length_list[offset_lev1] = \
							shift_center_length( mcenter,Ldim,np.array([0,+1,-1]) )

							# assign mbox split helper here 
							offset_lev2 += assign_mbox_split_helper(child_vec[np.array([2,6])],\
								offset_lev1,offset_lev2)

							offset_lev1 += 1

						if (UR_exists):
							child_list [offset_lev0,1] = offset_lev1
							parent_list[offset_lev1]   = offset_lev0

							center_list[offset_lev1],length_list[offset_lev1] = \
							shift_center_length( mcenter,Ldim,np.array([0,+1,+1]) )

							# assign mbox split helper here 
							offset_lev2 += assign_mbox_split_helper(child_vec[np.array([3,7])],\
								offset_lev1,offset_lev2)

							offset_lev1 += 1
						offset_lev0 += 1
					

		self.level_sep   = lev_offset
		self.center_list = center_list
		self.length_list = length_list
		self.isleaf_list = isleaf_list
		self.parent_list = parent_list
		self.child_list  = child_list
		self.mbox_list   = mbox_list
		self.mtree       = mtree

	@property
	def N(self):
		return self.XX.shape[0]

	@property
	def nboxes(self):
		return self.level_sep[-1]

	@property
	def nlevels(self):
		return self.level_sep.shape[0] - 1

	def get_box_center(self,box):
		return self.center_list[box]

	def get_box_length(self,box):
		return self.length_list[box]

	def get_box_level(self,box):
		return np.where((self.level_sep - box) > 0)[0][0] - 1 

	def get_box_children(self,box):
		tmp = self.child_list[box]
		return tmp [tmp > 0]

	def get_box_parent(self,box):
		return self.parent_list[box]

	def acc_inds(self,list_boxes):
		I_B = np.array([],dtype=int)
		for b_child in list_boxes:
			assert self.mbox_list[b_child] > 0
			I_B = np.hstack((I_B,self.mtree.get_box_inds(self.mbox_list[b_child])))
		return I_B

	def get_box_inds(self,box):
		if self.mbox_list[box] > 0:

			return self.mtree.get_box_inds(self.mbox_list[box])
		
		elif (self.ndim == 2 or (self.ndim == 3 and np.mod(self.get_box_level(box),3) == 2)):
			return self.acc_inds(self.get_box_children(box))

		elif (self.ndim == 3 and np.mod(self.get_box_level(box),3) == 1):

			ancestors = np.array([],dtype=int)
			for c in self.get_box_children(box):
				ancestors = np.hstack((ancestors,self.get_box_children(c)))

			return self.acc_inds(ancestors)

		else:
			raise ValueError


	def get_boxes_level(self,level):
		assert level < self.nlevels
		return np.arange(self.level_sep[level],self.level_sep[level+1])

	def get_leaves(self):
		return np.where(self.isleaf_list)[0]

	def is_leaf(self,box):
		return self.isleaf_list[box]

	def get_box_coarse_neigh(self,box):
		return np.array([],dtype=int)

	def get_box_colleague_neigh(self,box):
		return np.array([box],dtype=int)

