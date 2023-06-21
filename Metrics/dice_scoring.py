import numpy as np


def dice(pred, true, k=1):
    intersection = np.sum(pred[true == k]) * 2.0
    d = intersection / (np.sum(pred) + np.sum(true))
    return d
