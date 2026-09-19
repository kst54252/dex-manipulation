"""Explicit dataset frame changes; SI units, column SE(3), no simulator imports."""
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .transforms import apply, inverse, transform


def cylinder_bottom(pose, shape):
    """World Z support of the actual collision cylinder, including its local offset."""
    if shape['type'] != 'cylinder':
        raise ValueError('A confirmed cylinder collision shape is required')
    local = np.asarray(shape['transform'], dtype=float)
    rotation = Rotation.from_matrix(np.asarray(pose)[:3, :3]).as_matrix()
    center = rotation @ local[:3, 3] + np.asarray(pose)[:3, 3]
    axis = rotation @ local[:3, 2]
    axis /= np.linalg.norm(axis)
    radial_z = np.linalg.norm(axis[:2])
    return float(center[2] - shape['height'] / 2 * abs(axis[2]) - shape['radius'] * radial_z)


def collision_bottom(pose, shapes):
    """Lowest world-Z point of a compound cylinder object, including every part."""
    if not shapes:
        raise ValueError('Collision geometry is required for floor placement')
    return min(cylinder_bottom(pose, shape) for shape in shapes)


def z_aligned_cylinders(geometry):
    """Validate the explicitly supported initial-object +Z-up convention."""
    shapes = geometry['collision_shapes']
    if not shapes or any(s['type'] != 'cylinder' or not np.allclose(np.asarray(s['transform'])[:3, :3], np.eye(3)) for s in shapes):
        raise ValueError('Initial-can-up convention requires object-local Z cylinder parts')
    return shapes


def transform_reference(data, frame):
    """Change all world-space reference fields together; local geometry/q/time stay exact."""
    result = {key: value.copy() for key, value in data.items()}
    for key in ('wrist_transform', 'object_transform'):
        result[key] = frame @ data[key]
    for key in ('human_keypoints', 'robot_keypoints', 'object_keypoints'):
        result[key] = apply(frame, data[key])
    result['wrist_translation_m'] = result['wrist_transform'][:, :3, 3].copy()
    result['wrist_quaternion_xyzw'] = Rotation.from_matrix(result['wrist_transform'][:, :3, :3]).as_quat()
    return result


def ground_dataset(poses_path, reference_path, geometry_path, config_path, output):
    """Save a derived dataset with the first can bottom on Z=0 and its local +Z up."""
    inputs = [Path(p) for p in (poses_path, reference_path, geometry_path, config_path)]
    output = Path(output)
    destinations = [output / name for name in ('poses.npz', 'reference.npz', 'frame.json')]
    if {p.resolve() for p in inputs} & {p.resolve() for p in destinations}:
        raise ValueError('Derived output must not overwrite any source file')
    with np.load(poses_path, allow_pickle=False) as archive:
        poses = {k: archive[k].copy() for k in archive.files}
    with np.load(reference_path, allow_pickle=False) as archive:
        reference = {k: archive[k].copy() for k in archive.files}
    if 'frame_metadata_json' in poses or 'frame_metadata_json' in reference:
        raise ValueError('Already transformed input: use the original camera dataset to avoid double grounding')
    config = json.loads(Path(config_path).read_text())
    from .data import select_frame_arrays
    poses = select_frame_arrays(poses, config.get('frame_range'))
    reference = select_frame_arrays(reference, config.get('frame_range'))
    geometry = json.loads(Path(geometry_path).read_text())
    shapes = z_aligned_cylinders(geometry)
    if geometry.get('fingerprint') and str(reference.get('object_geometry_fingerprint', '')) != geometry['fingerprint']:
        raise ValueError('Reference uses a different object geometry; rerun retargeting')
    if not np.array_equal(poses['frame_ids'], reference['frame_ids']):
        raise ValueError('Pose and reference frame IDs differ')
    if not all(np.isfinite(poses[k]).all() for k in ('pose_y', 'joint_3d')):
        raise ValueError('Source annotations contain nonfinite values')
    source_poses = np.broadcast_to(np.eye(4), (*poses['pose_y'].shape[:2], 4, 4)).copy()
    source_poses[..., :3, :] = poses['pose_y']
    selected = source_poses[:, config['object_index']] @ np.asarray(config['mesh_to_object'])
    if not np.allclose(selected, reference['object_transform'], atol=1e-7, rtol=0):
        raise ValueError('Source and retargeted object trajectories do not match')
    first = reference['object_transform'][0]
    # Dataset float32 rotations have small rounding errors. Only the new frame
    # rotation is projected to SO(3); source pose matrices are never edited locally.
    frame = inverse(transform(Rotation.from_matrix(first[:3, :3]).as_matrix(), first[:3, 3]))
    frame[2, 3] -= collision_bottom(frame @ first, shapes)
    grounded = transform_reference(reference, frame)
    poses['pose_y'] = (frame @ source_poses)[..., :3, :]
    poses['joint_3d'] = apply(frame, poses['joint_3d'])
    relative_before = np.linalg.inv(reference['object_transform']) @ reference['wrist_transform']
    relative_after = np.linalg.inv(grounded['object_transform']) @ grounded['wrist_transform']
    metadata = dict(schema='dex_dataset_frame_v1', coordinate_frame='ground', length_unit='m',
                    quaternion_order='xyzw', ground_z_m=0., initial_frame_id=int(poses['frame_ids'][0]),
                    final_frame_id=int(poses['frame_ids'][-1]), frame_count=len(poses['frame_ids']),
                    frame_ids=poses['frame_ids'].tolist(), frame_range=config.get('frame_range'),
                    duration_s=float(reference['timestamps_s'][-1]-reference['timestamps_s'][0]),
                    source_axes=config['camera_axes'], world_axes='z_up; xy aligned with initial can local xy',
                    world_from_source=frame.tolist(), source_from_world=inverse(frame).tolist(),
                    object_index=config['object_index'], object_name=config['object_name'],
                    initial_object_pose=grounded['object_transform'][0].tolist(),
                    initial_collision_bottom_z_m=collision_bottom(grounded['object_transform'][0], shapes),
                    collision_shapes=shapes,object_geometry_fingerprint=geometry.get('fingerprint'),
                    hand_object_relative_transform_max_error=float(np.abs(relative_before-relative_after).max()),
                    time_basis=config.get('time_basis'), retimed_fps=config.get('retimed_fps'),
                    method='One fixed rigid transform for every hand/object pose and keypoint in all frames; no scaling, per-frame correction, or joint/time changes',
                    assumption='Initial can local +Z is simulation up; no measured table/camera-to-robot calibration',
                    hardware_calibrated=False,
                    sources=[dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in inputs])
    encoded = np.array(json.dumps(metadata))
    poses['frame_metadata_json'] = encoded
    grounded['frame_metadata_json'] = encoded
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destinations[0], **poses)
    np.savez_compressed(destinations[1], **grounded)
    destinations[2].write_text(json.dumps(metadata, indent=2)+'\n')
    return metadata
