#!/usr/bin/env python3
"""
Various visuals for HTR (draft)
"""
import matplotlib.pyplot as plt
import numpy as np
from typing import List


def predictions_over_scores( strings: List[str], scores: List[List[float]]):
    """
    Given a batch of predictions, and the corresponding scores, display both as a heatmap.

    Args:
        strings (List[str]): list of predicted strings.
        scores (List[float]): lists of scores (all lists have the same length as the string of max. length).
    """

    fig, ax = plt.subplots()

    max_width = max( len(l) for l in strings)
    im = ax.imshow( np.array( scores ))
    data = im.get_array()
    ax.set_yticks( np.arange(len(strings)) )
    threshold = im.norm( data.max())/2.
    for l in range(len(strings)):
        for c in range(len(strings[l])):
            text = ax.text(c,l,strings[l][c],ha='center',va='center',color='w' if data[l][c] <threshold else 'b')
    plt.show()

