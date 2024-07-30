import numpy as np
from numpy.linalg import norm
from numpy.typing import ArrayLike
from typing import List

_DISTANCES = {}

def _register_distance(distance_fn):
    _DISTANCES[distance_fn.__name__] = distance_fn
    return distance_fn

@_register_distance
def cosine(e0: ArrayLike, e1: ArrayLike):
    return np.dot(e0, e1) / (norm(e0) * norm(e1))

@_register_distance
def euclidean(e0: ArrayLike, e1: ArrayLike):
    return norm(e0 - e1)

def calc_distance(e0: ArrayLike, e1: ArrayLike, metric: str):
    if metric not in _DISTANCES:
        raise ValueError(f"Distance metric {metric} not found")
    return _DISTANCES[metric](e0, e1)

def calc_distances(e0: ArrayLike, e1s: List[ArrayLike], metric: str):
    # use np.vectorize to apply the distance function to all elements in e1s
    e1s = np.array(e1s)
    return np.vectorize(lambda e1: calc_distance(e0, e1, metric))(e1s)