"""Training/evaluation orchestration, invoked after Isaac startup."""
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from ..fk import HandModel
from ..data import legacy_demo_paths
from .trajectory import ReferenceMotion, digest
from .env import PhysxResidualEnv
from .ppo import PPO
from .math3d import numpy_tree
from .progress import format_progress,scalar_stats,timing_stats


def evaluate(env, learner, output, label, seed, zero=False, protocol='strict'):
    # Evaluation must not advance curriculum, train the RSI sampler, or consume training RNG.
    saved = env.training_state_dict()
    threshold = env.cfg['failure_object_error_m']
    if protocol not in ('source', 'strict'):
        raise ValueError('Unknown evaluation protocol')
    try:
        env.evaluation_protocol = protocol
        if protocol == 'source':
            env.cfg['failure_object_error_m'] = 1.0
        env.set_training(False)
        env.rng = np.random.default_rng(seed)
        env.generator.manual_seed(seed)
        return _evaluate_episode_batch(env, learner, output, label, seed, zero)
    finally:
        env.cfg['failure_object_error_m'] = threshold
        env.load_training_state_dict(saved)


def _evaluate_episode_batch(env, learner, output, label, seed, zero=False):
    """Full sequence from frame 15; never counts random mid-sequence resets as success."""
    env.reset(randomize=True)
    start_reset_count = env.object_reset_count
    obs = env.observation()
    n = env.num_envs
    alive, success = np.ones(n, bool), np.zeros(n, bool)
    demo_completed, tracking_only = np.zeros(n, bool), np.zeros(n, bool)
    initial_height = numpy_tree(env.state()['object_position'])[:, 2].copy()
    maximum_lift = np.zeros(n)
    rows, metric_rows, returns, lengths = [], [], np.zeros(n), np.zeros(n, int)
    max_error = np.zeros(n)
    constraints_ok = np.ones(n, bool)
    failure_reasons = [None] * n
    first_done = None
    for step in range(int(round(env.reference.duration / env.task.dt)) + 1):
        if zero:
            action = np.zeros((n, 12), np.float32)
        else:
            with torch.no_grad():
                action = learner.act(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action, auto_reset=False)
        if env.object_reset_count != start_reset_count:
            raise AssertionError("Object was reset during evaluation")
        info = {k: numpy_tree(v) for k, v in info.items() if k != "final_observation"}
        terminated, truncated = numpy_tree(terminated), numpy_tree(truncated)
        reward, action = numpy_tree(reward), numpy_tree(action)
        metrics = info["metrics"]
        metric_rows.append({k: v.copy() for k, v in metrics.items()} | {"alive": alive.copy()})
        returns += reward * alive
        lengths += alive
        max_error = np.maximum(max_error, metrics["object_keypoint_error_m"] * alive)
        maximum_lift = np.maximum(maximum_lift, (metrics['object_height_m'] - initial_height) * alive)
        step_constraints = ((metrics["joint_limit_violation_rad"] < 1e-3) &
                            (metrics["joint_velocity_violation_rad_s"] < 1e-2) &
                            (metrics["coupling_error_rad"] < 1e-2))
        if 'table_clearance_m' in metrics: step_constraints &= metrics['table_clearance_m'] >= 0.
        if 'arm_ik_success' in metrics:
            options=env.ik.options if hasattr(env,'ik') else env.bridge.solver.options
            step_constraints &= ((metrics['arm_ik_success']>.5)&(metrics['arm_joint_limit_violation_rad']<1e-3)
                &(metrics['arm_joint_velocity_violation_rad_s']<1e-2)&(metrics['arm_actual_sigma_min']>=options.singular_sigma_min)
                &(metrics['arm_actual_condition']<=options.max_condition))
        constraints_ok &= ~alive | step_constraints
        done = alive & (terminated | truncated)
        for i in np.flatnonzero(done):
            demo_completed[i] = bool(not metrics["early_failure"][i] and metrics["demo_end"][i])
            tracking_only[i] = bool(demo_completed[i] and max_error[i] < env.cfg["success_object_error_m"]
                                   and abs(metrics["object_height_m"][i] - metrics["target_object_height_m"][i]) < env.cfg["success_final_height_error_m"])
            success[i] = tracking_only[i] and constraints_ok[i]
            failure_reasons[i] = "tracking_threshold_or_coupling_or_drop" if metrics["early_failure"][i] else (None if success[i] else "completed_but_tracking_tolerance_failed")
        # Save the entire horizon, including the aftermath of a failure.
        state, ref = info["state"], info["reference"]
        rows.append(dict(time_s=info["reference_time"][0], action=action[0].copy(),
                         physics_time_s=(step+1)*env.task.dt,
                         terminated=bool(terminated[0]), truncated=bool(truncated[0]),
                         **{k: v[0].copy() for k, v in state.items()},
                         **{'raw_target_'+k:v[0].copy() for k,v in info['raw_targets'].items()},
                         **{'applied_target_'+k:v[0].copy() for k,v in info['applied_targets'].items()},
                         **{"reference_" + k: v[0].copy() for k, v in ref.items() if k != "time"}))
        if done[0] and first_done is None:
            first_done = dict(terminated=bool(terminated[0]), truncated=bool(truncated[0]))
        alive[done] = False
    output = Path(output)
    arrays = {key: np.asarray([r[key] for r in rows]) for key in rows[0]}
    arrays["body_names"] = np.array(env.robot.body_names)
    arrays["active_joint_names"] = np.array(env.model.active_names)
    arrays["full_joint_names"] = np.array(env.model.full_names)
    arrays["camera_to_world"] = env.reference.camera_to_world
    for key in info["metrics"]:
        arrays["metric_" + key] = np.array([r[key][0] for r in metric_rows[:len(rows)]])
    arrays["joint_constraints_valid"] = ((arrays["metric_joint_limit_violation_rad"] < 1e-3) &
                                         (arrays["metric_joint_velocity_violation_rad_s"] < 1e-2) &
                                         (arrays["metric_coupling_error_rad"] < 1e-2))
    arrays["episode_alive"] = np.logical_not(np.maximum.accumulate(arrays["metric_early_failure"]))
    arrays["valid"] = arrays["joint_constraints_valid"] & arrays["episode_alive"]
    if 'metric_table_clearance_m' in arrays:
        arrays['table_clearance_valid']=arrays['metric_table_clearance_m'] >= 0.
        arrays['valid'] &= arrays['table_clearance_valid']
    if 'metric_arm_ik_success' in arrays:
        arrays['arm_joint_names']=np.asarray(env.arm.active_names)
        arrays['arm_constraints_valid']=((arrays['metric_arm_ik_success']>.5)
            &(arrays['metric_arm_joint_limit_violation_rad']<1e-3)
            &(arrays['metric_arm_joint_velocity_violation_rad_s']<1e-2)
            &(arrays['metric_arm_actual_sigma_min']>=options.singular_sigma_min)
            &(arrays['metric_arm_actual_condition']<=options.max_condition))
        arrays['valid'] &= arrays['arm_constraints_valid']
    arrays["tracking_within_tolerance"] = arrays["metric_object_keypoint_error_m"] < env.cfg["success_object_error_m"]
    # Exact link-local semantics from measured rigid bodies, without imposing ideal coupling again.
    from scipy.spatial.transform import Rotation
    kp = []
    for point in env.model.keypoints:
        index = env.robot.body_names.index(point["link"])
        poses = arrays["link_transforms"][:, index]
        kp.append(Rotation.from_quat(poses[:, 3:7]).apply(point["xyz"]) + poses[:, :3])
    arrays["robot_keypoints"] = np.stack(kp, axis=1)
    arrays["object_keypoints"] = env.reference.points(arrays["object_position"], arrays["object_quaternion"])
    arrays["semantic_names"] = np.array(env.model.semantic_names)
    np.savez_compressed(output / f"{label}_rollout.npz", **arrays)
    mask = np.stack([r["alive"] for r in metric_rows])
    report = dict(label=label, seed=seed, episodes=n, success_count=int(success.sum()),
                  demo_completed_count=int(demo_completed.sum()),
                  object_tracking_only_count=int(tracking_only.sum()),
                  object_tracking_only_definition="full sequence and object tracking/height tolerances; excludes joint constraints, not a grasp certificate",
                  maximum_lift_m=maximum_lift.tolist(),
                  mean_maximum_lift_m=float(maximum_lift.mean()),
                  gravity_m_s2=env.gravity.value, rsi_enabled=False,
                  protocol=env.evaluation_protocol,
                  motion_control=env.motion_controller.config,
                  table_safety=env.table_safety.config if getattr(env,'table_safety',None) is not None else None,
                  motion_control_differs_from_training=env.motion_controller.config!=env.cfg.get('motion_control'),
                  augmentation_enabled=env.evaluation_protocol=='source' and env.cfg['augmentation']['enabled'],
                  observation_noise_and_delay_enabled=env.evaluation_protocol=='source',
                  failure_object_error_m=env.cfg['failure_object_error_m'],
                  success_definition="full sequence, no early failure, every-step object 50-keypoint mean error <25mm, final height error <25mm, joint/coupling tolerances and nonnegative recorded table clearance when enabled",
                  joint_constraints_passed=constraints_ok.tolist(),
                  returns=returns.tolist(), episode_steps=lengths.tolist(), failure_reasons=failure_reasons,
                  maximum_object_error_m=max_error.tolist(), first_rollout_end=first_done,
                  object_writes_during_rollout=env.object_reset_count - start_reset_count,
                  interpretation="dynamic simulated tracking; not a real-world grasp certification",
                  valid_mask_definition="finite simulated step, no early termination, joint/coupling numerical tolerances and nonnegative table clearance when enabled; not a general nonpenetration or grasp certificate")
    if 'table_clearance_m' in metric_rows[0]:
        report['table_clearance']=dict(minimum_m=float(min(r['table_clearance_m'].min() for r in metric_rows)),
            maximum_target_lift_m=float(max(r['table_target_lift_m'].max() for r in metric_rows)),
            violating_step_count=int(sum((r['table_clearance_m']<0).sum() for r in metric_rows)),
            geometry='conservative collision-mesh enclosing boxes against tabletop half-space')
    if 'arm_ik_success' in metric_rows[0]:
        report['arm']=dict(
            ik_success_fraction=float(np.stack([r['arm_ik_success'] for r in metric_rows]).mean()),
            ik_failure_step_indices_first_environment=np.flatnonzero(~arrays['metric_arm_ik_success'].astype(bool)).tolist(),
            constraints_passed_first_environment=bool(arrays['arm_constraints_valid'].all()),
            minimum_actual_sigma=float(min(r['arm_actual_sigma_min'].min() for r in metric_rows)),
            maximum_actual_condition=float(max(r['arm_actual_condition'].max() for r in metric_rows)))
        for name in ('arm_ik_position_error_m','arm_ik_orientation_error_rad','arm_tracking_position_error_m',
                     'arm_tracking_orientation_error_rad','arm_fk_consistency_position_error_m'):
            values=np.stack([r[name] for r in metric_rows])
            report['arm'][name]=dict(mean=float(values.mean()),maximum=float(values.max()))
        report['success_definition']+='; arm IK, arm velocity/limits and actual Jacobian singularity checks at every step'
        report['valid_mask_definition']+='; also includes arm IK, velocity/limits and singularity checks'
    for key in ("object_keypoint_error_m", "wrist_error_m", "joint_rmse_rad", "coupling_error_rad",
                "joint_limit_violation_rad", "joint_velocity_violation_rad_s"):
        values = np.stack([r[key] for r in metric_rows])[mask]
        report[key] = dict(mean=float(values.mean()), maximum=float(values.max()))
    from .evaluation import full_horizon_metrics
    report['full_horizon']=full_horizon_metrics(
        np.stack([r['object_keypoint_error_m'] for r in metric_rows]),
        np.stack([r['early_failure'] for r in metric_rows]),
        np.stack([r['object_height_m'] for r in metric_rows]),
        np.stack([r['target_object_height_m'] for r in metric_rows]))
    report['legacy_error_metrics']='Metrics outside full_horizon stop at first failure; never use them alone to claim tracking performance.'
    (output / f"{label}_evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
    print("[evaluation]", json.dumps(report), flush=True)
    return report


def plot_comparison(output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for label, color in (("zero_residual", "#dd8050"), ("trained_residual", "#3d92d1")):
        path = Path(output) / f"{label}_rollout.npz"
        if not path.exists():
            continue
        with np.load(path) as d:
            t = d["time_s"]
            axes[0].plot(t, d["object_position"][:, 2], label=label, color=color)
            axes[0].plot(t, d["reference_object_position"][:, 2], "k--", alpha=0.6, label="reference" if label == "zero_residual" else None)
            axes[1].plot(t, np.linalg.norm(d["object_position"] - d["reference_object_position"], axis=-1) * 1000, color=color)
            axes[2].plot(t, np.linalg.norm(d["wrist_position"] - d["reference_wrist_position"], axis=-1) * 1000, color=color)
    axes[0].set_ylabel("Can height (m)")
    axes[1].set_ylabel("Can position error (mm)")
    axes[2].set_ylabel("Wrist error (mm)")
    axes[2].set_xlabel("Retimed sequence time (s)")
    axes[0].legend()
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.suptitle("Dynamic PhysX evaluation: full horizon, including failed steps")
    fig.tight_layout()
    fig.savefig(Path(output) / "evaluation.png", dpi=150)
    plt.close(fig)


def run(args,root,on_ready=None,is_running=None):
    print('[startup] load configuration and reference',flush=True)
    config=json.loads(Path(args.config).read_text())
    config['source_profile']=args.source_profile or config.get('source_profile','leap')
    if config['source_profile']=='wuji':
        config['reward']['object_std_m']=.015
        config['reward']['action_l2_weight']=.5
        config['domain_randomization']['hand_friction']=[.5,1.5]
        config['domain_randomization']['joint_default_offset_rad']=0.
        config['domain_randomization']['object_com_range_m']=[.005,.02,.005]
        config['domain_randomization']['com_basis']='Source Wuji rigid screwdriver XYZ COM ranges'
    if args.device is not None:config['physics_device']=args.device
    if args.policy_device is None:args.policy_device=config['physics_device']
    if args.num_envs is None:args.num_envs=config['training']['num_envs'] if args.mode=='train' else 2
    if args.iterations is None:args.iterations=config['training']['iterations']
    if args.save_every is None:args.save_every=config['training']['save_every']
    if args.rsi_sampling is not None:config['rsi']['sampling']=args.rsi_sampling
    if args.enable_rsi:config['rsi']['enabled']=True
    for name,key in (('disable_rsi','rsi'),('disable_augmentation','augmentation'),('disable_gravity_curriculum','gravity_curriculum')):
        if getattr(args,name):config[key]['enabled']=False
    torch.set_num_threads(1)
    torch.manual_seed(config['seed']);np.random.seed(config['seed'])
    torch.backends.cuda.matmul.allow_tf32=True
    torch.backends.cudnn.allow_tf32=True
    model=HandModel.load(root/config['model'])
    reference=ReferenceMotion(root/config['reference'],model,root/config['object_geometry'],config['world_frame'])
    if args.mode=='train' and on_ready is None and 'base_diameter_m' in reference.object_geometry.get('specification',{}):
        # Keep historical checkpoints inspectable, but prevent new learning
        # from silently repeating an inverted asymmetric-can experiment.
        reference.validate_can_base_down()
    # Preserve the exact metadata contract of pre-v7 checkpoints.
    if 'reference_timing' in config:reference.configure_timing(config)
    else:reference.retime_for_control_horizon(config['episode_length_s'],config['physics_dt']*config['control_decimation'])
    last_frame=reference.metadata['control_reference_frames']-1
    dt=config['physics_dt']*config['control_decimation']
    if config['augmentation'].get('mode','fade')=='fade':
        for key,ratio in (('blend_start_s','phase_start_ratio'),('blend_end_s','phase_end_ratio')):
            config['augmentation'][key]=int(last_frame*config['augmentation'][ratio])*dt
    args.output.mkdir(parents=True,exist_ok=True)
    metadata=dict(config=config,reference=reference.metadata,model_sha256=digest(root/config['model']),
                  observation_schema='regrind_revo2_v4_actor67_critic94' if config['observation'].get('include_object_velocity',False) else 'regrind_revo2_v3_actor67_critic88',
                  implementation='independent task and simulator adapter; general RSL-RL PPO; no REGRIND imports')
    if config.get('reference_phase')=='endpoint_normalized':
        metadata['observation_schema']='regrind_revo2_v6_actor67_critic94'
    robot_mode=getattr(args,'robot','floating')
    native_arm=config.get('arm_training',{}).get('enabled',False)
    arm_config=None
    if native_arm and robot_mode!='arm':raise ValueError('An arm-trained policy requires --robot arm and real arm observations')
    if robot_mode=='arm':
        arm_path=getattr(args,'arm_config',None) or (root/config['arm_training']['arm_config'] if native_arm else root/'config/ik.json')
        arm_config=json.loads(Path(arm_path).read_text())
        if native_arm:
            metadata['observation_schema']='revo2_arm_actor87_critic114_v1'
            metadata['arm_training']=dict(config=arm_config,
                assets={name:digest(root/arm_config[name]) for name in ('usd','arm_model','workcell','alignment')})
        elif args.mode!='play':raise ValueError('Arm training requires config.arm_training.enabled')
    from ..scene import FloatingSurface
    metadata['surface']=FloatingSurface(config['surface'],reference.object[0,:3,3]).metadata()
    if native_arm:
        from ..scene import Workcell
        metadata['surface']=Workcell.load(root/arm_config['workcell']).metadata()
    metadata['contract_hash']=hashlib.sha256(json.dumps(legacy_demo_paths(metadata),sort_keys=True).encode()).hexdigest()
    control_mode=getattr(args,'motion_control',None) or 'checkpoint'
    stable=(json.loads((root/'config/policy.json').read_text())['motion_control']
            if control_mode=='stable' else None)
    if stable is not None:stable=dict(stable,enabled=True)
    if native_arm:
        from .arm_train import ArmTrainingEnv
        env=ArmTrainingEnv(root,model,reference,config,arm_config,args.num_envs,render=not args.headless)
    elif robot_mode=='arm':
        from .arm_env import ArmPolicyEnv
        env=ArmPolicyEnv(root,model,reference,config,arm_config,render=not args.headless)
    else:
        env=PhysxResidualEnv(root,model,reference,config,args.num_envs,render=not args.headless,
                             motion_control_override=stable)
    table_mode=getattr(args,'table_safety',None) or ('protect' if args.mode=='play' and not native_arm else 'checkpoint')
    if table_mode=='protect':
        table_settings=json.loads((root/'config/policy.json').read_text())['table_safety']
        env.configure_table_safety(dict(table_settings,enabled=True,guard_enabled=True))
    metadata['physics']=env.metadata
    # Execution/output choices do not change the policy/checkpoint contract.
    metadata['execution']=dict(headless=args.headless,render=not args.headless,
        viewport_updates=not args.headless,skip_evaluation=args.skip_evaluation,
        export=args.export,automatic_replay_output=not args.skip_evaluation and not native_arm,mode=args.mode)
    metadata['execution']['robot']=robot_mode
    metadata['execution']['table_safety']=dict(mode=table_mode,trained=config.get('table_safety'),
        applied=env.table_safety.config if env.table_safety is not None else None,
        interpretation='collision-geometry command guard; dynamic contacts still require validation')
    (args.output/'run_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    (args.output/'config.resolved.json').write_text(json.dumps(config,indent=2)+'\n')
    try:
        print('[recipe]',json.dumps(dict(rsi=config['rsi']['enabled'],gravity_curriculum=config['gravity_curriculum']['enabled'],augmentation=config['augmentation']['enabled'],device=str(env.device),num_envs=env.num_envs,
            gpu_dynamics=env.metadata.get('gpu_dynamics_enabled'),broadphase=env.metadata.get('broadphase'),
            learning=on_ready is None and args.mode=='train')),flush=True)
        if on_ready is not None:
            return on_ready(env,metadata)
        print('[startup] allocate PPO model and rollout storage',flush=True)
        learner=PPO(env.task,config['ppo'],device=args.policy_device,num_envs=env.num_envs)
        if args.checkpoint:
            loaded=learner.load(args.checkpoint,metadata,resume=args.mode=='train')
            if robot_mode=='arm':env.checkpoint_physics_names=loaded['metadata']['physics']
            if loaded.get('training_state') is not None:env.load_training_state_dict(loaded['training_state'])
            elif robot_mode=='arm':raise ValueError('Arm deployment requires saved named hand/object physical properties')
            elif args.mode=='train':raise ValueError('Resume requires complete curriculum state')
        initialize=getattr(args,'initialize_actor',None)
        if initialize:
            metadata['execution']['actor_initialization']=learner.initialize_arm_actor(initialize,metadata)
            metadata['execution']['actor_initialization'].update(path=str(initialize),sha256=digest(initialize))
        metadata['execution']['motion_control']=dict(mode=control_mode,
            trained=config.get('motion_control'), applied=env.motion_controller.config,
            differs_from_training=config.get('motion_control')!=env.motion_controller.config)
        (args.output/'run_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
        if args.mode!='train':
            print('[motion control]',json.dumps(metadata['execution']['motion_control']),flush=True)
            print('[table safety]',json.dumps(metadata['execution']['table_safety']),flush=True)
        if args.mode=='play':
            from .play import play
            return play(env,learner,args.output,args.episodes,is_running,
                        protocol='source' if args.evaluation_protocol=='source' else 'strict')
        baseline=None
        primary_protocol='source' if args.evaluation_protocol=='source' else 'strict'
        if not args.skip_evaluation:baseline=evaluate(env,learner,args.output,'zero_residual',config['seed']+1,zero=True,protocol=primary_protocol)
        if args.mode=='train':
            writer=None
            remote_log=None
            if args.logger=='tensorboard':
                from torch.utils.tensorboard import SummaryWriter
                writer=SummaryWriter(str(args.output/'tensorboard'))
            elif args.logger=='wandb':
                import wandb
                remote_log=wandb.init(project='dex-manipulation',name=args.output.name,config=config,dir=str(args.output))
            print('[startup] reset training environments',flush=True)
            env.set_training(True);env.reset()
            if config['training']['initial_random_episode_length']:
                env.episode_length_buf[:]=torch.randint(env.max_episode_length,(env.num_envs,),device=env.device,generator=env.generator)
            obs=env.observation();started=time.monotonic()
            start_iteration=learner.iteration
            with (args.output/'training.jsonl').open('a') as log:
                for iteration_in_run in range(args.iterations):
                    if iteration_in_run==0:print('[startup] collect first PPO rollout',flush=True)
                    iteration_started=time.monotonic()
                    batch,obs,stats=learner.collect(env,obs)
                    collection_finished=time.monotonic()
                    if iteration_in_run==0:print('[startup] optimize first PPO rollout',flush=True)
                    stats.update(learner.update(batch),iteration=learner.iteration,steps=learner.steps,
                        control_steps=env.control_steps,gravity_m_s2=env.gravity.value,gravity_stage=env.gravity.stage,
                        rsi_probabilities=env.sampler.probabilities().cpu().tolist(),push_events=env.push.total_events)
                    if native_arm:stats['arm_reset_ik_rejections_total']=env.reset_ik_rejections
                    finished=time.monotonic()
                    stats.update(timing_stats(elapsed=finished-started,completed=iteration_in_run+1,iterations=args.iterations,
                        start_iteration=start_iteration,collection=collection_finished-iteration_started,learning=finished-collection_finished,
                        iteration=finished-iteration_started,transitions=config['ppo']['rollout_steps']*env.num_envs))
                    log.write(json.dumps(stats)+'\n');log.flush()
                    if args.console=='json':print('[train]',json.dumps(stats),flush=True)
                    else:print(format_progress(stats,num_envs=env.num_envs,control_dt=env.task.dt,
                        rollout_steps=config['ppo']['rollout_steps'],decimation=config['control_decimation'],
                        rsi=config['rsi']['enabled'],augmentation=config['augmentation']['enabled']),flush=True)
                    scalars=scalar_stats(stats)
                    if writer:
                        for k,v in scalars.items():writer.add_scalar(k,v,learner.iteration)
                    if remote_log:remote_log.log(scalars,step=learner.iteration)
                    if learner.iteration%args.save_every==0:
                        checkpoint=args.output/f'model_{learner.iteration}.pt'
                        learner.save(checkpoint,metadata,env.training_state_dict())
                        import os
                        latest=args.output/'policy.pt.tmp'
                        latest.unlink(missing_ok=True)
                        os.link(checkpoint,latest)
                        latest.replace(args.output/'policy.pt')
                learner.save(args.output/'policy.pt',metadata,env.training_state_dict())
            if writer:writer.close()
            if remote_log:remote_log.finish()
        if args.export:learner.export(args.output/'exported')
        if args.skip_evaluation:return
        trained=evaluate(env,learner,args.output,'trained_residual',config['seed']+1,protocol=primary_protocol)
        if args.evaluation_protocol=='both':
            evaluate(env,learner,args.output,'zero_residual_source_play',config['seed']+1,zero=True,protocol='source')
            evaluate(env,learner,args.output,'trained_residual_source_play',config['seed']+1,protocol='source')
        comparison=dict(training_steps=learner.steps,zero_residual_success=baseline['success_count'],trained_success=trained['success_count'],episodes=env.num_envs,
            return_difference_mean=float(np.mean(trained['returns'])-np.mean(baseline['returns'])),
            simulated_full_sequence_tracking_success_observed=bool(trained['success_count']>0),policy_status='candidate_only; no automatic replacement of geometric reference',
            limitations=['single sequence','assumed table/world frame',
                'physical arm with sampled IK transition checks; not a general collision certificate' if native_arm else 'idealized floating wrist',
                'original capture fps unknown'],training_rsi=config['rsi'],training_gravity_curriculum=config['gravity_curriculum'],training_augmentation=config['augmentation'])
        (args.output/'comparison.json').write_text(json.dumps(comparison,indent=2)+'\n')
        plot_comparison(args.output)
        if not native_arm:
            from .replay import write_replay
            write_replay(args.output,model,reference)
    finally:env.close()
