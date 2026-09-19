"""Floating wrist pose PD, shared by physics preview and residual RL."""
import torch
from .policy.math3d import rotation_error


def wrist_pd_wrenches(state, target, masses, root_id, gains):
    """World-frame force at root COM and mass-weighted torques on all links.

    Independently implement the local REGRIND pose-PD method. Targets are held
    for one control interval; feedback is evaluated each physics tick. Damping
    acts on measured velocity, with no gravity or reference-velocity feedforward.
    Gravity is disabled on hand bodies in the environment, not on the object.
    """
    force = (gains['root_kp_position']*(target['position']-state['wrist_position'])
             - gains['root_kd_position']*state['wrist_velocity'])
    torque = (gains['root_kp_rotation']*rotation_error(target['quaternion'],state['wrist_quaternion'])
              - gains['root_kd_rotation']*state['wrist_angular_velocity'])
    forces = torch.zeros((*masses.shape, 3), device=force.device, dtype=force.dtype)
    forces[:, root_id] = force
    weights = masses / masses.sum(dim=1, keepdim=True)
    torques = weights[..., None] * torque[:, None]
    return forces, torques
