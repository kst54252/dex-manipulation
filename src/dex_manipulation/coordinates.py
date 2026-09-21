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


def translate_reference_frames(data, translations):
    """Translate hand/object together per frame, preserving all relative geometry."""
    translations = np.asarray(translations, dtype=float)
    if translations.shape != (len(data['frame_ids']), 3) or not np.isfinite(translations).all():
        raise ValueError('One finite XYZ translation is required per reference frame')
    result = {key: value.copy() for key, value in data.items()}
    for key in ('wrist_transform', 'object_transform'):
        result[key][:, :3, 3] += translations
    for key in ('human_keypoints', 'robot_keypoints', 'object_keypoints'):
        result[key] += translations[:, None, :]
    result['wrist_translation_m'] = result['wrist_transform'][:, :3, 3].copy()
    return result


def reference_in_source_frame(data):
    """Undo recorded floor corrections as well as the fixed ground-frame transform."""
    metadata = json.loads(str(data['frame_metadata_json']))
    if 'floor_stabilization' in metadata:
        correction = metadata['floor_stabilization']
        if correction['frame_ids'] != data['frame_ids'].tolist():
            raise ValueError('Floor correction frame IDs differ from the reference')
        data = translate_reference_frames(data, -np.asarray(correction['post_frame_translation_m']))
    return transform_reference(data, np.asarray(metadata['source_from_world']))


def stabilize_grounded_dataset(poses_path, reference_path, geometry_path, output):
    """Level the initial can base, then remove residual floor penetration.

    A single minimal rotation defines the new simulation up direction. A small
    per-frame shared vertical lift handles recorded pose noise below that plane.
    This is an explicit reference correction, not a runtime pose override.
    Source archives and historical checkpoint inputs remain untouched.
    """
    from .geometry import can_orientation_report
    paths = [Path(p) for p in (poses_path, reference_path, geometry_path)]
    output = Path(output)
    destinations = [output / name for name in ('poses.npz', 'reference.npz', 'frame.json')]
    if any(p.exists() for p in destinations):
        raise ValueError('Stable-ground output already exists; choose a new output directory')
    with np.load(paths[0], allow_pickle=False) as archive:
        poses = {k: archive[k].copy() for k in archive.files}
    with np.load(paths[1], allow_pickle=False) as archive:
        reference = {k: archive[k].copy() for k in archive.files}
    metadata = json.loads(str(reference['frame_metadata_json']))
    if metadata['coordinate_frame'] != 'ground' or 'floor_stabilization' in metadata:
        raise ValueError('Use the original grounded sequence, not a previously stabilized result')
    if not np.array_equal(poses['frame_ids'], reference['frame_ids']):
        raise ValueError('Pose/reference frame IDs differ')
    if json.loads(str(poses['frame_metadata_json'])) != metadata:
        raise ValueError('Pose/reference coordinate frames differ')
    geometry = json.loads(paths[2].read_text())
    shapes = z_aligned_cylinders(geometry)
    if str(reference.get('object_geometry_fingerprint', '')) != geometry['fingerprint']:
        raise ValueError('Reference collision geometry is stale')
    before = can_orientation_report(reference['object_transform'], geometry)
    if not before['initial_base_below_body']:
        raise ValueError('Correct the wider-base orientation before leveling the ground frame')
    first = reference['object_transform'][0]
    axis = Rotation.from_matrix(first[:3, :3]).as_matrix()[:, 2]
    tilt = float(np.arccos(np.clip(axis[2], -1., 1.)))
    cross = np.cross(axis, [0., 0., 1.])
    rotation = (Rotation.from_rotvec(cross / np.linalg.norm(cross) * tilt).as_matrix()
                if np.linalg.norm(cross) > 1e-12 else np.eye(3))
    frame = transform(rotation, -rotation @ first[:3, 3])
    frame[2, 3] -= collision_bottom(frame @ first, shapes)
    grounded = transform_reference(reference, frame)
    bottoms = np.array([collision_bottom(pose, shapes) for pose in grounded['object_transform']])
    shifts = np.zeros((len(bottoms), 3))
    shifts[:, 2] = np.maximum(-bottoms, 0.)
    grounded = translate_reference_frames(grounded, shifts)
    source_poses = np.broadcast_to(np.eye(4), (*poses['pose_y'].shape[:2], 4, 4)).copy()
    source_poses[..., :3, :] = poses['pose_y']
    poses['pose_y'] = (frame @ source_poses)[..., :3, :]
    poses['pose_y'][..., 3] += shifts[:, None, :]
    poses['joint_3d'] = apply(frame, poses['joint_3d']) + shifts[:, None, None, :]
    relative_before = np.linalg.inv(reference['object_transform']) @ reference['wrist_transform']
    relative_after = np.linalg.inv(grounded['object_transform']) @ grounded['wrist_transform']
    total_frame = frame @ np.asarray(metadata['world_from_source'])
    metadata.update(world_from_source=total_frame.tolist(), source_from_world=inverse(total_frame).tolist(),
                    world_axes='initial can axis defines +Z; minimum tilt correction of previous ground XY',
                    assumption='User-requested simulation ground from initial can base; not measured camera/robot calibration',
                    grounding_mode='initial_can_base_flat',
                    initial_object_pose=grounded['object_transform'][0].tolist(),
                    initial_collision_bottom_z_m=collision_bottom(grounded['object_transform'][0], shapes),
                    object_orientation_check=can_orientation_report(grounded['object_transform'], geometry),
                    hand_object_relative_transform_max_error=float(np.abs(relative_after-relative_before).max()),
                    method='Fixed scene rotation/translation, then shared hand/object vertical correction at floor-penetrating frames; joints, timing, local geometry unchanged',
                    sources=[dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths],
                    floor_stabilization=dict(
                        fixed_new_from_previous=frame.tolist(), initial_tilt_before_deg=float(np.degrees(tilt)),
                        frame_ids=reference['frame_ids'].tolist(), post_frame_translation_m=shifts.tolist(),
                        maximum_floor_correction_m=float(shifts[:, 2].max()),
                        corrected_frame_ids=reference['frame_ids'][shifts[:, 2] > 1e-9].tolist(),
                        initial_velocity='Existing simulator resets frame zero with zero linear/angular velocity',
                        scope='Reference geometry only; no kinematic object pinning, policy/physics/gain change or new grasp claim'))
    encoded = np.array(json.dumps(metadata))
    poses['frame_metadata_json'] = encoded
    grounded['frame_metadata_json'] = encoded
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destinations[0], **poses)
    np.savez_compressed(destinations[1], **grounded)
    destinations[2].write_text(json.dumps(metadata, indent=2) + '\n')
    return metadata


def ground_dataset(poses_path, reference_path, geometry_path, config_path, output):
    """Ground the first can using the explicitly selected source-up convention."""
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
    mode=config.get('grounding_mode','initial_object_plus_z')
    if mode=='initial_object_plus_z':
        frame = inverse(transform(Rotation.from_matrix(first[:3, :3]).as_matrix(), first[:3, 3]))
        world_axes='z_up; xy aligned with initial can local xy'
        assumption='Initial can local +Z is simulation up; no measured table/camera-to-robot calibration'
    elif mode=='camera_negative_y_up':
        if config['camera_axes']!='x_right_y_down_z_forward':
            raise ValueError('camera_negative_y_up requires declared optical camera axes')
        rotation=np.array([[1.,0,0],[0,0,1.],[0,-1.,0]])
        frame=transform(rotation,-rotation@first[:3,3])
        world_axes='x=camera +X, y=camera +Z, z=camera -Y'
        assumption='Declared camera -Y is simulation up; no measured table/camera-to-robot calibration'
    else:raise ValueError('Unknown explicit grounding_mode')
    frame[2, 3] -= collision_bottom(frame @ first, shapes)
    grounded = transform_reference(reference, frame)
    orientation = None
    if config.get('object_orientation_requirement') == 'initial_base_below_body':
        from .geometry import can_orientation_report
        orientation = can_orientation_report(grounded['object_transform'], geometry)
        if not orientation['initial_base_below_body']:
            raise ValueError('Can wider base is above the body; fix mesh_to_object and rerun retargeting')
    poses['pose_y'] = (frame @ source_poses)[..., :3, :]
    poses['joint_3d'] = apply(frame, poses['joint_3d'])
    relative_before = np.linalg.inv(reference['object_transform']) @ reference['wrist_transform']
    relative_after = np.linalg.inv(grounded['object_transform']) @ grounded['wrist_transform']
    metadata = dict(schema='dex_dataset_frame_v1', coordinate_frame='ground', length_unit='m',
                    quaternion_order='xyzw', ground_z_m=0., initial_frame_id=int(poses['frame_ids'][0]),
                    final_frame_id=int(poses['frame_ids'][-1]), frame_count=len(poses['frame_ids']),
                    frame_ids=poses['frame_ids'].tolist(), frame_range=config.get('frame_range'),
                    duration_s=float(reference['timestamps_s'][-1]-reference['timestamps_s'][0]),
                    source_axes=config['camera_axes'], world_axes=world_axes,
                    world_from_source=frame.tolist(), source_from_world=inverse(frame).tolist(),
                    object_index=config['object_index'], object_name=config['object_name'],
                    initial_object_pose=grounded['object_transform'][0].tolist(),
                    initial_collision_bottom_z_m=collision_bottom(grounded['object_transform'][0], shapes),
                    collision_shapes=shapes,object_geometry_fingerprint=geometry.get('fingerprint'),
                    hand_object_relative_transform_max_error=float(np.abs(relative_before-relative_after).max()),
                    time_basis=config.get('time_basis'), retimed_fps=config.get('retimed_fps'),
                    method='One fixed rigid transform for every hand/object pose and keypoint in all frames; no scaling, per-frame correction, or joint/time changes',
                    assumption=assumption,
                    hardware_calibrated=False,
                    sources=[dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in inputs])
    if mode!='initial_object_plus_z':metadata['grounding_mode']=mode
    if orientation is not None:
        metadata['object_orientation_requirement']='initial_base_below_body'
        metadata['object_orientation_check']=orientation
        metadata['mesh_to_object']=config['mesh_to_object']
    if config.get('source_playback_fps') is not None:metadata['source_playback_fps']=config['source_playback_fps']
    encoded = np.array(json.dumps(metadata))
    poses['frame_metadata_json'] = encoded
    grounded['frame_metadata_json'] = encoded
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destinations[0], **poses)
    np.savez_compressed(destinations[1], **grounded)
    destinations[2].write_text(json.dumps(metadata, indent=2)+'\n')
    return metadata
