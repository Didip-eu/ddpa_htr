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
        scores (List[float]): lists of scores 
    """

    fig, ax = plt.subplots()

    max_width = max( len(l) for l in strings)
    im = ax.imshow( np.stack([ np.pad( np.array(sl), (0,max_width-len(sl)) ) for sl in scores] ))
    data = im.get_array()
    ax.set_yticks( np.arange(len(strings)) )
    threshold = im.norm( data.max())/2.
    for l in range(len(strings)):
        for c in range(len(strings[l])):
            text = ax.text(c,l,strings[l][c],ha='center',va='center',color='w' if data[l][c] <threshold else 'b')
    plt.show()


def plot_confusion_matrix( cm: np.ndarray, alph: dict):

    labels = alph.keys()
    
    fig, ax = plt.subplots()
    ax.set_xticks( range(len(alph)), labels=alph.keys())
    ax.set_yticks( range(len(alph)), labels=alph.keys())
    plt.imshow( cm )
    for i in range(len(alph)):
        for j in range(len(alph)):
            if cm[i,j] > .05:
                text = ax.text(j,i, f'{cm[i,j]:.2f}',ha='center',va='center', color='w' if cm[i,j]<.5 else 'b')
    plt.show()
