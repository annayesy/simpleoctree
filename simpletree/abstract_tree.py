import numpy as np
from abc import ABCMeta, abstractmethod, abstractproperty

# Abstract Tree class.
# To instantiate a class of this type, use syntax `class MyTreeClass(AbstractTree)`.
# All instantiated classes of this type MUST implement the abstract methods.
# Otherwise, you will encounter a TypeError.
class AbstractTree(metaclass=ABCMeta):

    # Returns: number [int] of levels.
    @abstractproperty
    def nlevels(self):
        pass

    # Returns: number [int] of boxes.
    @abstractproperty
    def nboxes(self):
        pass

    # Returns: total number [int] of points.
    @abstractproperty
    def N(self):
        pass

    # Returns: level [int] of the box.
    @abstractmethod
    def get_box_level(self, box):
        pass

    ################## Getters for box attributes related to geometry ##################

    # Returns adjacent boxes on the same level
    @abstractmethod
    def get_box_colleague_neigh(self, box):
        pass

    # Returns boxes L, where L is a leaf on a level above and L is adjacent to B
    @abstractmethod
    def get_box_coarse_neigh(self, box):
        pass

    # The near-field of a box consists of its colleagues and coarse neighbors
    def get_box_neighbors(self, box):
        # Get colleague neighbors
        coll = self.get_box_colleague_neigh(box)
        # Get coarse neighbors
        coarse = self.get_box_coarse_neigh(box)

        # Concatenate colleague and coarse neighbors
        neigh_list = np.concatenate((coll, coarse))
        # Remove the box itself from the neighbor list
        tmp = np.setdiff1d(neigh_list, np.array([box]))
        # Add the box itself at the start of the neighbor list
        neigh_list = np.concatenate((np.array([box]), tmp))

        return neigh_list

    ################# Getter and setters for box indices ##################
    # The process of skeletonization of a box separates indices into an
    # active and inactive set.

    # Input: box [int].
    # Returns: inds [1d int numpy array] of points in box B.
    @abstractmethod
    def get_box_inds(self, box):
        pass

    # Returns the parent box of the given box
    @abstractmethod
    def get_box_parent(self, box):
        pass

    # Returns the length of the box
    @abstractmethod
    def get_box_length(self, box):
        pass

    # Returns the center of the box
    @abstractmethod
    def get_box_center(self, box):
        pass

    # Returns a list of all leaf boxes
    @abstractmethod
    def get_leaves(self):
        pass

    # Returns all boxes at a specific level
    @abstractmethod
    def get_boxes_level(self, level):
        pass

    # Returns whether a box is a leaf
    @abstractmethod
    def is_leaf(self, box):
        pass

    # Returns indices of points in neighboring boxes
    def get_neigh_inds(self, box):
        # Function to get indices of points in a box
        ind_func = self.get_box_inds

        # Get neighbors of the box
        box_neighbors = self.get_box_neighbors(box)
        num_neigh = box_neighbors.shape[0]

        # Allocate memory for neighbor indices
        max_ind_count = 1000
        tot_neigh_inds = np.zeros(max_ind_count, dtype=int)
        acc = 0

        # Iterate through each neighbor
        for neigh in box_neighbors:
            neigh_inds = ind_func(neigh)
            # Resize array if needed
            if acc + neigh_inds.shape[0] > max_ind_count:
                max_ind_count += max_ind_count + neigh_inds.shape[0]
                tmp = np.zeros(max_ind_count, dtype=int)
                tmp[:acc] = tot_neigh_inds[:acc]
                tot_neigh_inds = tmp
                del tmp

            # Copy neighbor indices into the total neighbor indices array
            tot_neigh_inds[acc: acc + neigh_inds.shape[0]] = neigh_inds.copy()
            acc += neigh_inds.shape[0]

        # Return the array of neighbor indices
        return tot_neigh_inds[:acc]
