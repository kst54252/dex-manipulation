"""Strict, bounded RB3 inverse kinematics. NumPy/SciPy only; no simulator API."""
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from .fk import ArmModel, HandModel
from .transforms import inverse


def checked_pose(value):
    pose = np.asarray(value, dtype=float)
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise ValueError('Expected a finite 4x4 SE(3) transform')
    if (not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-9) or
            not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-6) or
            not np.isclose(np.linalg.det(pose[:3, :3]), 1., atol=1e-6)):
        raise ValueError('Invalid rigid transform; scale/reflection is not accepted')
    return pose


def pose_error(target, actual):
    return np.r_[target[:3, 3] - actual[:3, 3],
                 Rotation.from_matrix(target[:3, :3] @ actual[:3, :3].T).as_rotvec()]


def model_fingerprint(model):
    fields = {key:model.description[key] for key in ('root_link','joints','keypoints')}
    if isinstance(model, ArmModel):
        fields['flange_to_wrist'] = model.description['flange_to_wrist']
    return hashlib.sha256(json.dumps(fields,sort_keys=True).encode()).hexdigest()


@dataclass
class IKOptions:
    position_tolerance_m: float = 1e-5
    orientation_tolerance_rad: float = 1e-4
    jacobian_length_m: float = .5
    singular_sigma_min: float = .001
    near_sigma_min: float = .03
    max_condition: float = 10000.
    near_condition: float = 200.
    damping_min: float = 1e-6
    damping_max: float = .04
    limit_warning_rad: float = np.deg2rad(5.)
    limit_margin_rad: float = 1e-4
    max_joint_step_rad: float = np.deg2rad(20.)
    iteration_step_rad: float = .2
    max_iterations: int = 180
    initial_seeds: int = 24
    continuation_seeds: int = 5
    transition_samples: int = 20
    seed: int = 7310

    def __post_init__(self):
        positive = ('position_tolerance_m', 'orientation_tolerance_rad', 'jacobian_length_m',
                    'singular_sigma_min', 'near_sigma_min', 'max_condition', 'near_condition',
                    'damping_min', 'damping_max', 'limit_warning_rad', 'limit_margin_rad',
                    'max_joint_step_rad', 'iteration_step_rad', 'max_iterations',
                    'initial_seeds', 'continuation_seeds', 'transition_samples')
        if any(not np.isfinite(getattr(self, k)) or getattr(self, k) <= 0 for k in positive):
            raise ValueError('IK tolerances/counts must be finite and positive')
        if self.singular_sigma_min >= self.near_sigma_min or self.damping_min > self.damping_max:
            raise ValueError('Inconsistent singularity/damping thresholds')


@dataclass
class IKResult:
    q: np.ndarray
    fk_pose: np.ndarray
    position_error_m: float
    orientation_error_rad: float
    sigma_min: float
    condition: float
    near_singular: bool
    joint_limit_distance_rad: float
    near_joint_limit: bool
    success: bool
    reason: str
    iterations: int
    damping_max: float
    candidates: int = 0


class ArmIK:
    def __init__(self, model: ArmModel, options=None):
        self.model, self.options = model, options or IKOptions()
        self.lower = model.lower + self.options.limit_margin_rad
        self.upper = model.upper - self.options.limit_margin_rad
        if np.any(self.lower >= self.upper):
            raise ValueError('Joint margin removes the feasible interval')
        self.scale = np.array([1 / self.options.jacobian_length_m] * 3 + [1.] * 3)

    def singularity(self, q):
        # Normalize translational rows before mixing metres and radians in SVD.
        jac = self.scale[:, None] * self.model.jacobian(q, 'flange')
        values = np.linalg.svd(jac, compute_uv=False)
        condition = float(values[0] / values[-1]) if values[-1] > 1e-15 else float('inf')
        return float(values[-1]), condition

    def _result(self, q, target, iterations, damping, bounds):
        actual = self.model.pose(q)
        error = pose_error(target, actual)
        pos, angle = float(np.linalg.norm(error[:3])), float(np.linalg.norm(error[3:]))
        sigma, condition = self.singularity(q)
        margin = float(np.min(np.minimum(q - self.model.lower, self.model.upper - q)))
        reasons = []
        if pos > self.options.position_tolerance_m or angle > self.options.orientation_tolerance_rad:
            reasons.append('pose_tolerance_not_met')
        if sigma < self.options.singular_sigma_min or condition > self.options.max_condition:
            reasons.append('singular_configuration')
        if margin < self.options.limit_margin_rad - 1e-10:
            reasons.append('joint_limit_margin')
        if np.any(q < bounds[0] - 1e-10) or np.any(q > bounds[1] + 1e-10):
            reasons.append('step_or_velocity_limit')
        return IKResult(q.copy(), actual, pos, angle, sigma, condition,
                        sigma < self.options.near_sigma_min or condition > self.options.near_condition,
                        margin, margin < self.options.limit_warning_rad, not reasons,
                        ';'.join(reasons) if reasons else 'ok', iterations, damping)

    def _solve_seed(self, target, seed, preferred, lower, upper):
        o = self.options
        flange = self.model.flange_target(target)
        q = np.clip(seed, lower, upper)
        peak_damping = 0.
        for iteration in range(o.max_iterations):
            wrist_error = pose_error(target, self.model.pose(q))
            if (np.linalg.norm(wrist_error[:3]) <= o.position_tolerance_m * .1 and
                    np.linalg.norm(wrist_error[3:]) <= o.orientation_tolerance_rad * .1):
                break
            error = self.scale * pose_error(flange, self.model.pose(q, 'flange'))
            jac = self.scale[:, None] * self.model.jacobian(q, 'flange')
            sigma = np.linalg.svd(jac, compute_uv=False)[-1]
            damping = o.damping_min + o.damping_max * max(0., 1 - sigma / o.near_sigma_min)**2
            peak_damping = max(peak_damping, damping)
            distance = np.maximum(np.minimum(q - self.model.lower, self.model.upper - q), 1e-5)
            # Tikhonov regularization becomes stronger for joints close to limits.
            # A weak bounded centering/previous-pose bias is subordinate to strict pose checks.
            weight = np.sqrt(1 + np.minimum((o.limit_warning_rad / distance)**2, 100.))
            center = (self.model.lower + self.model.upper) / 2
            bias = np.clip(.05 * (center - q) / (self.model.upper - self.model.lower)
                           + .1 * (preferred - q), -.1, .1)
            penalty = damping * np.diag(weight)
            step = np.linalg.lstsq(np.vstack([jac, penalty]), np.r_[error, penalty @ bias], rcond=None)[0]
            maximum = np.max(np.abs(step))
            if maximum > o.iteration_step_rad:
                step *= o.iteration_step_rad / maximum
            # Project each trial into joint AND frame-to-frame constraints.
            cost = float(error @ error)
            accepted = False
            for alpha in (1., .5, .25, .125, .0625, .03125, .015625):
                trial = np.clip(q + alpha * step, lower, upper)
                residual = self.scale * pose_error(flange, self.model.pose(trial, 'flange'))
                if residual @ residual < cost - 1e-20:
                    accepted = True
                    break
            if not accepted or np.max(np.abs(trial - q)) < 1e-11:
                break
            q = trial
        return self._result(q, target, iteration + 1, peak_damping, (lower, upper))

    def solve(self, wrist_target, seed, previous=None, dt=None):
        target = checked_pose(wrist_target)
        seed = np.asarray(seed, float)
        if seed.shape != (6,) or not np.isfinite(seed).all():
            raise ValueError('Expected finite seed q[6] in USD joint order, radians')
        lower, upper = self.lower.copy(), self.upper.copy()
        preferred = seed
        if previous is not None:
            previous = np.asarray(previous, float)
            if previous.shape != (6,) or not np.isfinite(previous).all() or dt is None or not np.isfinite(dt) or dt <= 0:
                raise ValueError('Previous configuration and actual positive dt are required')
            if np.any(previous < self.lower - 1e-9) or np.any(previous > self.upper + 1e-9):
                raise ValueError('Previous configuration outside arm limits')
            bound = np.minimum(self.model.velocity * dt, self.options.max_joint_step_rad)
            lower, upper = np.maximum(lower, previous - bound), np.minimum(upper, previous + bound)
            preferred = previous
        rng = np.random.default_rng(self.options.seed)
        count = self.options.initial_seeds if previous is None else self.options.continuation_seeds
        seeds = [np.clip(preferred, lower, upper)]
        for i in range(count - 1):
            # First try local alternatives. Global search is permitted only at startup;
            # every continuation candidate is confined to the previous solution's box.
            if previous is None and i >= 5:
                seeds.append(rng.uniform(np.maximum(lower, -np.pi), np.minimum(upper, np.pi)))
            else:
                radius = .35 if previous is None else .12
                seeds.append(np.clip(preferred + rng.normal(0, radius, 6), lower, upper))
        candidates = []
        for initial in seeds:
            candidate = self._solve_seed(target, initial, preferred, lower, upper)
            candidates.append(candidate)
            if candidate.success and np.linalg.norm(candidate.q - preferred) < 1e-9:
                break  # Exact closest possible valid solution.
        valid = [r for r in candidates if r.success]
        if valid:
            # No modulo-2pi shortcut: actual commanded joint differences matter.
            result = min(valid, key=lambda r: (np.linalg.norm(r.q - preferred), -r.joint_limit_distance_rad))
        else:
            result = min(candidates, key=lambda r: r.position_error_m / self.options.jacobian_length_m + r.orientation_error_rad)
            if previous is not None and ('pose_tolerance' in result.reason):
                result.reason += ';unreachable_or_continuity_bound'
        result.candidates = len(candidates)
        return result

    def transition(self, before, after):
        samples = np.linspace(before, after, self.options.transition_samples + 1)
        values = np.array([self.singularity(q) for q in samples])
        return dict(sigma_min=float(values[:, 0].min()), condition_max=float(values[:, 1].max()),
                    valid=bool(values[:, 0].min() >= self.options.singular_sigma_min and
                               values[:, 1].max() <= self.options.max_condition))


def interpolate_ik_source(source, count, substeps=1):
    """Retain exact source knots and insert timed SE(3)/finger interpolation knots.

    Inserted samples have frame_id=-1 and explicit source-interval IDs. They are
    never presented as camera frames. Validity is inherited from both endpoints
    and the existing retarget transition check; IK still checks each new sample.
    """
    if isinstance(substeps,bool) or not isinstance(substeps,(int,np.integer)) or substeps<1:
        raise ValueError('trajectory_substeps must be a positive integer')
    times=source['timestamps_s'][:count]
    ids=source['frame_ids'][:count]
    sample_times=np.r_[np.concatenate([np.linspace(a,b,substeps,endpoint=False) for a,b in zip(times[:-1],times[1:])]),times[-1]] if count>1 else times.copy()
    n=len(sample_times); originals=np.arange(count)*substeps
    mask=np.zeros(n,bool);mask[originals]=True
    frame_ids=np.full(n,-1,dtype=ids.dtype);frame_ids[originals]=ids
    intervals=np.stack((frame_ids,frame_ids),axis=1)
    valid=np.zeros(n,bool);transitions=np.zeros(n,bool)
    valid[originals]=source['valid'][:count]
    transitions[originals]=source['transition_valid'][:count]
    for i in range(count-1):
        span=slice(i*substeps+1,(i+1)*substeps)
        intervals[span]=[ids[i],ids[i+1]]
        valid[span]=source['valid'][i] and source['valid'][i+1] and source['transition_valid'][i+1]
        transitions[span]=source['transition_valid'][i+1]
    out=dict(source)
    out.update(timestamps_s=sample_times,frame_ids=frame_ids,valid=valid,transition_valid=transitions,
               source_frame_mask=mask,source_interval_frame_ids=intervals)
    for name in ('wrist_transform','object_transform'):
        if name not in source:continue
        poses=source[name][:count]
        result=np.tile(np.eye(4),(n,1,1))
        result[:,:3,3]=np.stack([np.interp(sample_times,times,poses[:,i,3]) for i in range(3)],axis=1)
        if count>1:result[:,:3,:3]=Slerp(times,Rotation.from_matrix(poses[:,:3,:3]))(sample_times).as_matrix()
        result[originals]=poses  # Preserve the original matrices exactly.
        out[name]=result
    q=source['active_q_rad'][:count]
    out['active_q_rad']=np.stack([np.interp(sample_times,times,q[:,i]) for i in range(q.shape[1])],axis=1)
    out['active_q_rad'][originals]=q
    return out


def solve_trajectory(model, hand, input_path, output, base_from_source, seed, options=None,
                     alignment_status='provided_unverified', max_frames=None, trajectory_substeps=1,
                     *, scene_placement=None):
    """Preserve targets/timestamps/fingers; failed candidates stay diagnostic only."""
    alignment = checked_pose(base_from_source)
    with np.load(input_path, allow_pickle=False) as data:
        source = {key: data[key].copy() for key in data.files}
    names = list(source['active_joint_names'])
    if len(names) != 6 or set(names) != set(hand.active_names):
        raise ValueError('Finger joint names do not match the Revo2 independent joints')
    finger = source['active_q_rad'][:, [names.index(name) for name in hand.active_names]]
    times = source['timestamps_s']
    if times.ndim != 1 or len(times) < 1 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError('Strictly increasing real/explicitly retimed timestamps required')
    if (source['wrist_transform'].shape != (len(times),4,4) or
            any(source[k].shape != (len(times),) for k in ('frame_ids','valid','transition_valid'))):
        raise ValueError('Mismatched source frame arrays')
    n = len(times) if max_frames is None else min(max_frames, len(times))
    if n < 1 or finger.shape != (len(times), 6):
        raise ValueError('Invalid frame count/finger shape')
    for pose in source['wrist_transform']:
        checked_pose(pose)
    if not np.isfinite(finger).all() or np.any(finger < hand.lower - 1e-7) or np.any(finger > hand.upper + 1e-7):
        raise ValueError('Finger trajectory violates the extracted hand limits')
    source_count=n
    source = interpolate_ik_source(source, n, trajectory_substeps)
    times = source['timestamps_s']; n=len(times)
    finger = source['active_q_rad'][:, [names.index(name) for name in hand.active_names]]
    wrist_source = source['wrist_transform']
    targets = alignment @ wrist_source
    solver = ArmIK(model, options)
    results, transitions = [], []
    last_q, last_time, last_index = None, None, None
    for i, target in enumerate(targets):
        result = solver.solve(target, seed if last_q is None else last_q,
                              previous=last_q, dt=None if last_time is None else times[i] - last_time)
        transition = dict(valid=i == 0, sigma_min=result.sigma_min, condition_max=result.condition)
        if i > 0:
            transition['valid'] = False
            if last_index == i - 1 and result.success:
                transition = solver.transition(last_q, result.q)
                if not transition['valid']:
                    result.success = False; result.reason += ';singular_transition'
        if not source['valid'][i] or (i > 0 and not source['transition_valid'][i]):
            result.success = False; result.reason += ';invalid_source_reference'
        if i > 0 and np.any(np.abs(finger[i] - finger[i-1]) / (times[i] - times[i-1]) > hand.velocity + 1e-6):
            result.success = False; result.reason += ';finger_velocity_limit'
        # No path through a failed frame is silently promoted into a valid segment.
        if result.success:
            last_q, last_time, last_index = result.q, times[i], i
        transition['valid'] = bool(transition['valid'] and result.success)
        results.append(result); transitions.append(transition)
        label=source['frame_ids'][i] if source['source_frame_mask'][i] else f"{source['source_interval_frame_ids'][i].tolist()} intermediate"
        print(f'[ik] frame={label} valid={result.success} '
              f'pos={result.position_error_m:.3g}m rot={result.orientation_error_rad:.3g}rad '
              f'sigma={result.sigma_min:.4g} {result.reason}', flush=True)
    arm = np.stack([r.q for r in results])
    success = np.array([r.success for r in results])
    step = np.vstack([np.zeros(6), np.abs(np.diff(arm, axis=0))])
    velocity = np.vstack([np.zeros(6), np.diff(arm, axis=0) / np.diff(times)[:, None]])
    metadata = dict(schema='dex_arm_hand_reference_v1', length_unit='m', angle_unit='rad',
                    quaternion_order='xyzw', target_frame='RB3 USD base link',
                    source_wrist_frame=hand.root, base_from_source=alignment.tolist(),
                    alignment_status=alignment_status, hardware_calibrated=False,
                    arm_model_source_sha256=model.description['source_sha256'],
                    arm_fingerprint=model_fingerprint(model), hand_fingerprint=model_fingerprint(hand),
                    input_sha256=hashlib.sha256(Path(input_path).read_bytes()).hexdigest(),
                    object_geometry_fingerprint=str(source.get('object_geometry_fingerprint','')),
                    initial_seed_q_rad=np.asarray(seed).tolist(), initial_robot_state_verified=False,
                    model_joint_order=model.active_names, finger_joint_order=hand.active_names,
                    interpolation='piecewise linear unwrapped joint position',
                    collision_validation='not performed for arm/environment or combined arm-hand',
                    trajectory_substeps=trajectory_substeps, source_frame_count=source_count,
                    source_frame_ids=source['frame_ids'][source['source_frame_mask']].tolist(),
                    input_interpolation='Exact source knots retained; linear translation/finger joints and wrist/object SLERP between knots',
                    control_validation='kinematic reference; not dynamic tracking or hardware execution',
                    solver=asdict(solver.options))
    if scene_placement is not None:
        metadata['scene_placement'] = scene_placement
    if 'frame_metadata_json' in source:
        metadata['dataset_frame'] = json.loads(str(source['frame_metadata_json']))
    arrays = dict(timestamps_s=times, frame_ids=source['frame_ids'][:n], q_arm=arm, q_finger=finger,
                  source_frame_mask=source['source_frame_mask'], source_interval_frame_ids=source['source_interval_frame_ids'],
                  joint_position_rad=np.c_[arm, finger], joint_names=np.array(model.active_names + hand.active_names),
                  arm_joint_names=np.array(model.active_names), finger_joint_names=np.array(hand.active_names),
                  joint_velocity_limits_rad_s=np.r_[model.velocity, hand.velocity],
                  joint_lower_rad=np.r_[model.lower, hand.lower], joint_upper_rad=np.r_[model.upper, hand.upper],
                  wrist_target_source=wrist_source, wrist_target_pose=targets,
                  flange_target_pose=np.stack([model.flange_target(t) for t in targets]),
                  fk_pose=np.stack([r.fk_pose for r in results]),
                  position_error_m=np.array([r.position_error_m for r in results]),
                  orientation_error_rad=np.array([r.orientation_error_rad for r in results]),
                  sigma_min=np.array([r.sigma_min for r in results]), condition=np.array([r.condition for r in results]),
                  near_singular=np.array([r.near_singular for r in results]),
                  joint_limit_distance_rad=np.array([r.joint_limit_distance_rad for r in results]),
                  near_joint_limit=np.array([r.near_joint_limit for r in results]),
                  damping_max=np.array([r.damping_max for r in results]),
                  solver_iterations=np.array([r.iterations for r in results]),
                  candidate_count=np.array([r.candidates for r in results]),
                  joint_step_rad=step, arm_velocity_rad_s=velocity, success=success,
                  transition_valid=np.array([t['valid'] for t in transitions]),
                  transition_sigma_min=np.array([t['sigma_min'] for t in transitions]),
                  transition_condition_max=np.array([t['condition_max'] for t in transitions]),
                  failure_reason=np.array([r.reason for r in results]), metadata_json=np.array(json.dumps(metadata)))
    if 'object_transform' in source:
        arrays['object_transform'] = alignment @ source['object_transform'][:n]
    report = dict(metadata=metadata, frame_count=n, source_frame_count=source_count,
                  source_success_count=int(success[source['source_frame_mask']].sum()),
                  success_count=int(success.sum()), success_rate=float(success.mean()),
                  continuous_reference_valid=bool(success.all() and arrays['transition_valid'].all()),
                  position_error_m=dict(mean=float(arrays['position_error_m'].mean()), maximum=float(arrays['position_error_m'].max())),
                  orientation_error_rad=dict(mean=float(arrays['orientation_error_rad'].mean()), maximum=float(arrays['orientation_error_rad'].max())),
                  max_joint_step_rad=float(step.max()), max_joint_speed_rad_s=float(np.abs(velocity).max()),
                  minimum_sigma=float(arrays['sigma_min'].min()),
                  minimum_transition_sigma=float(arrays['transition_sigma_min'].min()),
                  minimum_joint_limit_distance_rad=float(arrays['joint_limit_distance_rad'].min()),
                  near_singular_transition_frame_ids=source['frame_ids'][:n][
                      (arrays['transition_sigma_min'] < solver.options.near_sigma_min) |
                      (arrays['transition_condition_max'] > solver.options.near_condition)].tolist(),
                  near_singular_frame_ids=source['frame_ids'][:n][arrays['near_singular']].tolist(),
                  near_joint_limit_frame_ids=source['frame_ids'][:n][arrays['near_joint_limit']].tolist(),
                  failed_frame_indices=np.flatnonzero(~success).tolist(), failed_frame_ids=source['frame_ids'][:n][~success].tolist(),
                  failed_source_intervals=source['source_interval_frame_ids'][~success].tolist(),
                  failed_transition_frame_ids=source['frame_ids'][:n][~arrays['transition_valid']].tolist(),
                  diagnostic_candidates_include_failures=True)
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / 'trajectory.npz', **arrays)
    (output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    return report
