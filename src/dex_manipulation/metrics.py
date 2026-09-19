"""Semantic motion comparisons independent of differing human/robot bone lengths."""
import numpy as np

FINGERS = ("thumb", "index", "middle", "ring", "little")


def finger_edges(semantic_names):
    """Fifteen phalange edges; palm lengths must not dominate finger articulation."""
    edges = []
    for finger in FINGERS:
        chain = [semantic_names.index(f"{finger}_{joint}") for joint in ("mcp", "pip", "dip", "tip")]
        edges.extend(zip(chain[:-1], chain[1:]))
    return np.asarray(edges, dtype=int)


def directions(points, edges):
    vectors = np.asarray(points)[edges[:, 1]] - np.asarray(points)[edges[:, 0]]
    lengths = np.linalg.norm(vectors, axis=1)
    if np.any(lengths < 1e-8):
        raise ValueError("Cannot compare a zero-length semantic finger segment")
    return vectors / lengths[:, None]


def direction_error(source, target, edges):
    reference, actual = directions(source, edges), directions(target, edges)
    cosine = np.clip(np.sum(reference * actual, axis=1), -1, 1)
    return dict(mean_deg=float(np.rad2deg(np.arccos(cosine)).mean()),
                by_finger_deg=np.rad2deg(np.arccos(cosine)).reshape(5, 3).mean(1).tolist())
