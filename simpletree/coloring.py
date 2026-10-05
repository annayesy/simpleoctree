"""Greedy box coloring using only Python and NumPy."""

import numpy as np


def greedy_colors(colleagues, coarse):
    """Return colors and their count, visiting boxes in increasing index order.

    Both integer matrices have one row per box and use -1 for missing neighbors.
    Self and future neighbors do not constrain this directed greedy ordering.
    Duplicate entries are harmless. Inputs are not modified.
    """
    nboxes = colleagues.shape[0]
    colors = [0] * nboxes
    marked = [-1] * (colleagues.shape[1] + coarse.shape[1] + 1)
    # Bound temporary adjacency storage instead of converting whole tables to
    # Python objects. Each box needs only neighbors that already have a color.
    for start in range(0, nboxes, 4096):
        stop = min(nboxes, start + 4096)
        neighbors = np.concatenate((colleagues[start:stop], coarse[start:stop]), axis=1)
        earlier = (neighbors >= 0) & (neighbors < np.arange(start, stop)[:, None])
        ends = np.cumsum(np.count_nonzero(earlier, axis=1)).tolist()
        adjacent = neighbors[earlier].tolist()
        begin = 0
        for box, end in enumerate(ends, start):
            for neighbor in adjacent[begin:end]:
                marked[colors[neighbor]] = box
            color = 0
            while marked[color] == box:
                color += 1
            colors[box] = color
            begin = end
    colors = np.asarray(colors, dtype=int)
    return colors, int(colors.max(initial=0)) + 1
