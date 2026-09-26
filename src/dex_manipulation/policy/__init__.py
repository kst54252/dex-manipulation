"""Independent residual RL implementation. Isaac is imported only by the backend."""

from dex_manipulation.policy.trajectory import ReferenceMotion
from .task import ResidualTask

__all__ = ["ReferenceMotion", "ResidualTask"]
