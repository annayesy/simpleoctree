# simpleoctree

[![DOI](https://zenodo.org/badge/811938075.svg)](https://doi.org/10.5281/zenodo.14613403)
[![License](https://img.shields.io/github/license/annayesy/slabLU)](./LICENSE.md)
[![Top language](https://img.shields.io/github/languages/top/annayesy/simplebalancedoctree)](https://www.python.org)
![Code size](https://img.shields.io/github/languages/code-size/annayesy/simplebalancedoctree)
[![Latest commit](https://img.shields.io/github/last-commit/annayesy/simplebalancedoctree)](https://github.com/annayesy/simplebalancedoctree/commits/main)

An adaptive 2:1 balanced tree for point distributions in 2D and 3D, written in Python with minimal dependencies.

## Overview
**simpleoctree** is a Python package that adaptively partitions point distributions in 2D and 3D into a 2:1 balanced tree structure. The package ensures that any two neighboring leaf nodes are either on the same level or differ by one level, resulting in a neighbor list of bounded size.

<p align="center">
    <img src="https://github.com/annayesy/simplebalancedoctree/blob/main/examples/tree_balance.png" width="75%"/>
</p>
<div style="display: flex; justify-content: center;">
    <p style="width: 75%; text-align: center; font-size: 90%;">
        Figure 1: A leaf box that violates the level restriction constraint (highlighted in red) and a refined leaf box added in its place to satisfy the level restriction (highlighted in green).
    </p>
</div>
<p align="center">
    <img src="https://github.com/annayesy/simplebalancedoctree/blob/main/examples/curvy_annulus.png" width="40%"/>
</p>
<div style="display: flex; justify-content: center;">
    <p style="width: 60%; text-align: center; font-size: 90%;">
        Figure 2: A balanced quad-tree for an adaptive point distribution. A box is highlighted in black and its neighbors in green. Notice that the box has coarse neighbors on a level above.
    </p>
</div>

## Installation
To install the package, clone the repository and run the following command:
```
pip install -e .
```

## Features
- Adaptive Partitioning: Efficiently partitions point distributions in both 2D and 3D spaces.
- Level-Restricted Tree: Ensures a 2:1 balance, where neighboring leaves are either on the same level or differ by one level.
- Utility Functions: Includes several utility functions to find neighbors, parents, and other tree properties, all documented in `simpletree/abstract_tree.py`.

## Usage
The tree structure includes many useful utilities for accessing neighbors, parents, and other related nodes. Comprehensive documentation for these functions is available in the `simpletree/abstract_tree.py` file.

This package is particularly useful for the development of fast solvers for adaptive geometries, e.g. the Fast Multipole Method (FMM) and the compression and [invertible] factorization of $\mathcal H$-matrices.
For more detailed examples and usage instructions, please refer to the repository's documentation and example scripts.

