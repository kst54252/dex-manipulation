"""Batched SI/XYZW rotation operations, independent of the simulator."""

import torch


def quat_normalize(q):
    return q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def quat_inverse(q):
    return torch.cat((-q[..., :3], q[..., 3:]), -1)


def quat_multiply(a, b):
    av, aw, bv, bw = a[..., :3], a[..., 3:], b[..., :3], b[..., 3:]
    return torch.cat(
        (
            aw * bv + bw * av + torch.cross(av, bv, dim=-1),
            aw * bw - (av * bv).sum(-1, keepdim=True),
        ),
        -1,
    )


def quat_apply(q, v):
    cross = 2 * torch.cross(q[..., :3].expand_as(v), v, dim=-1)
    return v + q[..., 3:] * cross + torch.cross(q[..., :3].expand_as(v), cross, dim=-1)


def from_rotvec(v):
    theta = v.norm(dim=-1, keepdim=True)
    return torch.cat((v * (0.5 * torch.sinc(theta / (2 * torch.pi))), torch.cos(theta / 2)), -1)


def to_rotvec(q):
    q = quat_normalize(q)
    q = torch.where(q[..., 3:] < 0, -q, q)
    n = q[..., :3].norm(dim=-1, keepdim=True)
    scale = 2 * torch.atan2(n, q[..., 3:]) / n.clamp_min(1e-12)
    return q[..., :3] * torch.where(n < 1e-7, torch.full_like(scale, 2), scale)


def rotation_error(target, current):
    return to_rotvec(quat_multiply(target, quat_inverse(current)))


def slerp(a, b, t):
    b = torch.where((a * b).sum(-1, keepdim=True) < 0, -b, b)
    return quat_normalize(quat_multiply(from_rotvec(rotation_error(b, a) * t[..., None]), a))


def rotation6d(q):
    # First two matrix columns, row-major flatten, as in REGRIND's observations.
    x, y, z, w = quat_normalize(q).unbind(-1)
    return torch.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
        ),
        -1,
    )


def uniform(shape, low, high, device, generator):
    return torch.rand(shape, device=device, generator=generator) * (high - low) + low


def numpy_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    if isinstance(value, dict):
        return {k: numpy_tree(v) for k, v in value.items()}
    return value
