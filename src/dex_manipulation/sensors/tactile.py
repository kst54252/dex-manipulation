"""SI contact aggregation and explicitly typed Revo2 tactile register decoding.

Sensor firmware transfer functions and taxel positions are not inferred from a
rigid collision mesh. Simulated contact loads are not calibrated tactile data.
"""

import numpy as np

FINGERS = ("thumb", "index", "middle", "ring", "little")
STANDARD_GRAVITY = 9.80665


def aggregate_contacts(forces, normals, counts, starts):
    """Sum point loads and vectors separately; opposing normals may cancel."""
    forces = np.asarray(forces, float).reshape(-1)
    normals = np.asarray(normals, float)
    counts, starts = np.asarray(counts), np.asarray(starts)
    if normals.shape != (len(forces), 3) or counts.shape != starts.shape:
        raise ValueError("Invalid contact buffer shapes")
    total = np.zeros(counts.shape)
    vectors = np.zeros((*counts.shape, 3))
    for index in np.ndindex(counts.shape):
        n, start = int(counts[index]), int(starts[index])
        if n < 0 or (n and (start < 0 or start + n >= len(forces))):
            raise ValueError("Contact buffer exhausted or invalid; increase capacity")
        if not n:
            continue
        f, normal = forces[start : start + n], normals[start : start + n]
        if not np.isfinite(f).all() or not np.isfinite(normal).all() or (f < -1e-5).any():
            raise ValueError("Invalid contact force data")
        total[index] = np.maximum(f, 0).sum()
        vectors[index] = (f[:, None] * normal).sum(0)
    return total, vectors


def aggregate_separation(separation, counts, starts):
    """Minimum PhysX contact separation; NaN means no reported contact points.

    This measures sampled contact geometry, not full-hand or continuous-path
    clearance. Geometric overlap is max(0, -separation), without subtracting
    a negative rest offset (which would hide the allowed pad overlap).
    """
    separation = np.asarray(separation, float).reshape(-1)
    counts, starts = np.asarray(counts), np.asarray(starts)
    if counts.shape != starts.shape:
        raise ValueError("Invalid contact separation buffer shapes")
    minimum = np.full(counts.shape, np.nan)
    for index in np.ndindex(counts.shape):
        n, start = int(counts[index]), int(starts[index])
        if n < 0 or (n and (start < 0 or start + n >= len(separation))):
            raise ValueError("Contact separation buffer exhausted or invalid")
        if n:
            values = separation[start:start + n]
            if not np.isfinite(values).all():
                raise ValueError("Nonfinite contact separation")
            minimum[index] = values.min()
    return minimum, np.maximum(0., -np.nan_to_num(minimum, nan=0.))


def aggregate_friction(forces, counts, starts):
    forces = np.asarray(forces, float)
    counts, starts = np.asarray(counts), np.asarray(starts)
    if forces.ndim != 2 or forces.shape[-1] != 3 or counts.shape != starts.shape:
        raise ValueError("Invalid friction buffer shapes")
    vectors = np.zeros((*counts.shape, 3))
    for index in np.ndindex(counts.shape):
        n, start = int(counts[index]), int(starts[index])
        if n < 0 or (n and (start < 0 or start + n >= len(forces))):
            raise ValueError("Friction buffer exhausted or invalid; increase capacity")
        if n:
            values = forces[start : start + n]
            if not np.isfinite(values).all():
                raise ValueError("Nonfinite friction data")
            vectors[index] = values.sum(0)
    return vectors


def decode_capacitive(force_registers, status_registers):
    """Official 4200..4214 force triplets and 4225..4229 status, thumb first.

    Normal/tangential 0.01 N per count; direction 0..359 deg, 65535 invalid.
    Status low byte is errors, high byte a sequence counter, not a timestamp.
    """
    raw = np.asarray(force_registers)
    status = np.asarray(status_registers)
    if raw.shape != (5, 3) or status.shape != (5,):
        raise ValueError("Expected five force triplets and five status registers")
    for a in (raw, status):
        if not np.issubdtype(a.dtype, np.integer) or ((a < 0) | (a > 65535)).any():
            raise ValueError("Unsigned 16-bit registers required")
    raw, status = raw.astype(np.uint16), status.astype(np.uint16)
    valid = ((status & 255) == 0) & (raw[:, :2] <= 2500).all(-1)
    direction_valid = valid & (raw[:, 2] < 360)
    return dict(normal_n=raw[:, 0] * .01, shear_n=raw[:, 1] * .01,
                direction_deg=np.where(direction_valid, raw[:, 2], np.nan),
                valid=valid, direction_valid=direction_valid,
                status_error=status & 255, sequence=status >> 8)


def decode_pressure(values, *, calibrated):
    """Nine calibrated gram-force channels per finger; never convert raw ADC."""
    values = np.asarray(values)
    if values.shape != (5, 9) or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Expected five fingers with nine nonnegative pressure values")
    if not calibrated:
        raise ValueError("Raw ADC needs the hardware calibration; cannot convert it to newtons")
    taxel_n = values.astype(float) * STANDARD_GRAVITY / 1000
    return dict(taxel_n=taxel_n, normal_n=taxel_n.sum(-1))


def pad_axes(model):
    """USD-derived approximation, not a manufacturer sensor mounting calibration.

    The thinnest mesh axis defines the pad surface normal, oriented away from
    the distal-link origin toward the pad attachment. The projected vector from
    distal-link origin to semantic fingertip defines
    the zero-degree direction. Clockwise is viewed from outside the pad.
    """
    from ..materials import PAD_BODIES
    from ..transforms import inverse

    transforms = model.link_transforms(np.zeros(len(model.active_names)))
    normals, tips = [], []
    for name in PAD_BODIES:
        vertices = np.array(next(c for c in model.colliders if c['link'] == name)['vertices'])
        axis = int(np.argmin(np.ptp(vertices, axis=0)))
        normal = np.zeros(3)
        joint = next(j for j in model.joints if j['child'] == name)
        origin = (inverse(transforms[name]) @ transforms[joint['parent']])[:3, 3]
        normal[axis] = -np.sign(origin[axis])
        tip = np.array(next(k for k in model.keypoints if k['link'] == name)['xyz']) - origin
        tip -= normal * np.dot(tip, normal)
        if np.linalg.norm(tip) < 1e-8 or not np.isclose(np.linalg.norm(normal), 1):
            raise ValueError('Cannot derive pad axes from this USD model')
        normals.append(normal)
        tips.append(tip / np.linalg.norm(tip))
    return np.array(normals), np.array(tips)


def capacitive_proxy(force_pad_n, outward, toward_tip):
    """Encode a force in approximate pad axes as Revo2 capacitive force triplets.

    0.01 N/count is wire encoding, not asserted sensor accuracy/resolution.
    Preserve unsaturated physical values and an explicit saturation flag.
    Proximity, status errors and sensor noise are not synthesized.
    """
    force = np.asarray(force_pad_n, float)
    outward, toward_tip = np.asarray(outward, float), np.asarray(toward_tip, float)
    if force.shape[-2:] != (5, 3) or outward.shape != (5, 3) or toward_tip.shape != (5, 3):
        raise ValueError('Expected five force vectors and five pad axis pairs')
    if not np.isfinite(force).all() or not np.isfinite(outward).all() or not np.isfinite(toward_tip).all():
        raise ValueError('Nonfinite sensor force or axes')
    if not (np.allclose(np.linalg.norm(outward, axis=-1), 1) and
            np.allclose(np.linalg.norm(toward_tip, axis=-1), 1) and
            np.allclose((outward*toward_tip).sum(-1), 0, atol=1e-6)):
        raise ValueError('Sensor axes must be orthonormal')
    signed = (force * outward).sum(-1)
    normal = np.maximum(-signed, 0)
    tangent = force - signed[..., None] * outward
    shear = np.linalg.norm(tangent, axis=-1)
    clockwise = -np.cross(outward, toward_tip)
    angle = np.mod(np.degrees(np.arctan2((tangent*clockwise).sum(-1), (tangent*toward_tip).sum(-1))), 360)
    # No stable direction below one protocol count; this is our encoding choice.
    direction_valid = shear >= .01
    registers = np.stack((np.rint(np.clip(normal, 0, 25)/.01),
                          np.rint(np.clip(shear, 0, 25)/.01),
                          np.where(direction_valid, np.rint(angle) % 360, 65535)), -1).astype(np.uint16)
    return dict(normal_n=normal, shear_n=shear,
                direction_deg=np.where(direction_valid, angle, np.nan),
                registers=registers, saturated=(normal>25)|(shear>25),
                direction_valid=direction_valid)
