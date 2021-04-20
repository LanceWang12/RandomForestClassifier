import numpy as np
from numba import jit
import math
import random
import cython

class Node:
    def __init__(self, node_class):
        self.node_class = node_class
        self.feature_index = 0
        self.decision_threshold = 0
        self.left = None
        self.right = None

    def set_idx_thrs(self, feature_index, decision_threshold):
        self.feature_index, self.decision_threshold = feature_index, decision_threshold

def select_samples(samples_idx, max_samples):
    samples_idx = np.arange(samples_idx)
    if max_samples % 1 != 0:
        if (max_samples <= 0) or (max_samples > 1):
            raise ValueError('The parameter of max_samples must in (0, 1]!!')
        k = int(len(samples_idx) * max_samples)
    else:
        k = len(samples_idx)
    
    select_idx = np.array(random.choices(samples_idx, k = k))
    
    return select_idx