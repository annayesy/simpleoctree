import numpy as np
from abc import ABCMeta, abstractmethod,abstractproperty

###
# Abstract Tree class.
# To instantiate a class of this type, use syntax `class MyTreeClass(AbstractTree)`.
# All instantiated classes of this type MUST implement the abstract methods.
# Otherwise, you will encounter a TypeError.
class AbstractTree(metaclass = ABCMeta):
    
    ### get_nlevels
    # Returns: number [int] of levels.
    @abstractproperty
    def nlevels(self):
        pass
    
    ### get_num_boxes
    # Returns: number [int] of boxes.
    @abstractproperty
    def nboxes(self):
        pass
    
    ### get_num_points
    # Returns: number [int] of points.
    @abstractproperty
    def N(self):
        pass
    
    ### get_postorder_traversal
    @abstractmethod
    def get_box_level(self,box):
        pass
    
    ################## Getters for box attributes related to geometry ##################
    
    ### Adjacent boxes on the same level
    @abstractmethod
    def get_box_colleague_neigh(self,box):
        pass
    
    ### returns boxes A, where A is a leaf on a level above and A is adjacent to B
    @abstractmethod
    def get_box_coarse_neigh(self,box):
        pass

    def get_box_neighbors(self,box):
        coll   = self.get_box_colleague_neigh(box)
        coarse = self.get_box_coarse_neigh(box)
        
        neigh_list = np.concatenate((coll,coarse))
        tmp = np.setdiff1d(neigh_list,np.array([box]))
        neigh_list = np.concatenate((np.array([box]),tmp))
        
        return neigh_list
    
    
    ################# Getter and setters for box indices ##################
    # The process of skeletonization of a box separates indices into an 
    # active and inactive set.
    
    ### get_box_inds
    # Input: box [int].
    # Returns: inds [1d int numpy array] of points in box B .
    @abstractmethod
    def get_box_inds(self,box):
        pass
    
    @abstractmethod
    def get_box_parent(self,box):
        pass
    
    
    @abstractmethod
    def get_box_length(self,box):
        pass
    
    @abstractmethod
    def get_box_center(self,box):
        pass
    
    @abstractmethod
    def get_leaves(self):
        pass
    
    @abstractmethod
    def get_boxes_level(self,level):
        pass
    
    @abstractmethod
    def is_leaf(self,box):
        pass
    
    def get_neigh_inds(self,box):
        ind_func = self.get_box_inds

        box_neighbors = self.get_box_neighbors(box)
        num_neigh     = box_neighbors.shape[0]

        max_ind_count  = 1000
        tot_neigh_inds = np.zeros(max_ind_count,dtype=int); acc = 0
        for neigh in box_neighbors:
            neigh_inds = ind_func(neigh)
            if (acc + neigh_inds.shape[0] > max_ind_count):
                max_ind_count += max_ind_count + neigh_inds.shape[0]
                tmp = np.zeros(max_ind_count,dtype=int)
                tmp[:acc] = tot_neigh_inds[:acc]
                tot_neigh_inds = tmp; del tmp

            tot_neigh_inds[acc : acc + neigh_inds.shape[0]] = neigh_inds.copy()
            acc += neigh_inds.shape[0]
        return tot_neigh_inds[:acc]

    def _color_boxes(self):

        num_colors = 1
        box_colors = np.ones(self.nboxes,dtype=int) * (-1)
        for box in range(self.nboxes):

            neigh_boxes  = self.get_box_neighbors(box)
            neigh_colors = box_colors[neigh_boxes]

            unavailable_colors = np.unique(neigh_colors[neigh_colors >= 0])

            if (unavailable_colors.shape[0] == num_colors):

                # add new color
                box_colors[box] = num_colors
                num_colors += 1
            else:

                available_colors = np.setdiff1d(np.arange(num_colors),unavailable_colors)
                box_colors[box] = available_colors[0]
        self.box_colors = box_colors
        self.num_colors = num_colors

    def get_boxes_level_color(self,lev,color):

        if (not hasattr(self,'box_colors')):
            self._color_boxes()

        assert color < self.num_colors; assert color >= 0
        boxes_lev  = self.get_boxes_level(lev)
        box_colors = self.box_colors[boxes_lev]

        inds_color = np.where(box_colors == color)[0]

        if (inds_color.shape[0] == 0):
            return np.array([],dtype=int)
        else:
            return boxes_lev[ inds_color ]