"""Column-vector SE(3); serialized quaternions always XYZW."""
import numpy as np
from scipy.spatial.transform import Rotation


def transform(rotation=None, translation=None):
    result = np.eye(4)
    if rotation is not None:
        result[:3, :3] = rotation
    if translation is not None:
        result[:3, 3] = translation
    return result


def inverse(matrix):
    rotation = matrix[:3, :3]
    return transform(rotation.T, -rotation.T @ matrix[:3, 3])


def apply(matrix, points):
    return np.asarray(points) @ matrix[:3, :3].T + matrix[:3, 3]


def rotation_about(axis, angle):
    return transform(Rotation.from_rotvec(np.asarray(axis) * angle).as_matrix())


def rigid_fit(source, target):
    """Proper (non-reflecting, non-scaling) least-squares alignment."""
    a, b = np.asarray(source), np.asarray(target)
    u, _, vt = np.linalg.svd((a - a.mean(0)).T @ (b - b.mean(0)))
    correction = np.eye(3)
    correction[2, 2] = np.linalg.det(vt.T @ u.T)
    r = vt.T @ correction @ u.T
    return transform(r, b.mean(0) - r @ a.mean(0))
