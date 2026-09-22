"""Source Delaunay connectivity and uniform Laplacian, including object rows."""

from itertools import combinations
import numpy as np
from scipy.spatial import Delaunay


def interaction_graph(source):
    source = np.asarray(source)
    if source.ndim != 2 or source.shape[1] != 3 or not np.isfinite(source).all():
        raise ValueError("Expected finite 3D vertices")
    if len(np.unique(source, axis=0)) != len(source):
        raise ValueError("Duplicate interaction vertices")
    tetrahedra = Delaunay(source).simplices
    edges = sorted({tuple(sorted(pair)) for tet in tetrahedra for pair in combinations(tet, 2)})
    adjacency = np.zeros((len(source), len(source)))
    for a, b in edges:
        adjacency[a, b] = adjacency[b, a] = 1
    degrees = adjacency.sum(axis=1)
    if np.any(degrees == 0):
        raise ValueError("Delaunay omitted a vertex")
    laplacian = np.eye(len(source)) - adjacency / degrees[:, None]
    return laplacian, np.asarray(edges, dtype=np.int32)


def deformation(laplacian, source, target, norm="regrind", epsilon=1e-7):
    residual = laplacian @ (target - source)
    squared = np.sum(residual * residual, axis=1)
    if norm == "regrind":
        return float(np.sum(np.sqrt(squared + epsilon**2) - epsilon)), residual
    if norm == "omni":
        return float(np.sum(squared)), residual
    raise ValueError(f"Unknown Laplacian norm: {norm}")
