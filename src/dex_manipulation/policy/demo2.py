"""Isolated demo2 five-finger training on the existing physics/PPO components."""
import hashlib
import json
import time
from pathlib import Path

import torch

from .grasp import SCHEMA, FINGERS, attach_contact_reward
from .ppo import PPO
from .progress import format_progress, scalar_stats, timing_stats


def contact_summary(stats):
    """Conditional rates: the approach interval must not dilute grasp metrics."""
    m = stats['metrics']
    active = m['grasp_active']['mean']
    return dict(active_sample_fraction=active,
                all_five_fraction=(m['grasp_window_all_five_duty']['mean'] / active if active else None),
                finger_contact_fraction=(m['grasp_window_contact_fraction']['mean'] / active if active else None),
                per_finger_fraction={name: (m[f'grasp_window_{name}_duty']['mean'] / active if active else None)
                                     for name in FINGERS})


def evaluate_contacts(env, learner, output):
    """Full-gravity, frame-15 starts; no RSI, no reset within an episode.

    Failed/unobserved suffixes count as missing contacts. Geometric tracking
    alone is never relabeled five-finger success.
    """
    import numpy as np
    env.evaluation_protocol = 'strict'
    env.set_training(False)
    env.reset(randomize=False)
    obs = env.observation()
    alive = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    completed = torch.zeros_like(alive)
    all_five = torch.zeros(env.num_envs, device=env.device)
    per_finger = torch.zeros(env.num_envs, 5, device=env.device)
    max_error = torch.zeros_like(all_five)
    max_force = torch.zeros_like(all_five)
    table_ok = torch.ones_like(alive)
    joints_ok = torch.ones_like(alive)
    window_steps = 0
    rows = []
    with torch.no_grad():
        for step in range(round(env.reference.duration / env.task.dt) + 1):
            obs, _, term, trunc, info = env.step(learner.act(obs), auto_reset=False)
            m = info['metrics']
            on = m['grasp_active'] > .5
            window_steps += int(on[0])
            mask = (on & alive).float()
            all_five += mask * m['grasp_all_five_duty']
            duty = torch.stack([m[f'grasp_{name}_duty'] for name in FINGERS], dim=-1)
            per_finger += mask[:, None] * duty
            max_error = torch.maximum(max_error, m['object_keypoint_error_m'])
            max_force = torch.maximum(max_force, m['grasp_max_force_n'])
            table_ok &= m.get('table_clearance_m', torch.zeros_like(max_error)) >= 0
            joints_ok &= ((m['joint_limit_violation_rad'] < 1e-3) &
                          (m['coupling_error_rad'] < 1e-2) &
                          (m['joint_velocity_violation_rad_s'] < 1e-2))
            done = term | trunc
            completed |= alive & done & m['demo_end'].bool() & ~m['early_failure'].bool()
            rows.append({'time_s': info['reference_time'].cpu().numpy(), 'alive': alive.cpu().numpy(),
                         **{name: value.cpu().numpy() for name, value in m.items()}})
            alive &= ~done
    if not window_steps:
        raise RuntimeError('Evaluation did not cover the requested grasp interval')
    fraction = all_five / window_steps
    success = completed & table_ok & joints_ok & (max_error <= env.cfg['success_object_error_m']) & (fraction >= .95)
    report = dict(schema=SCHEMA, gravity_m_s2=env.gravity.value, rsi=False,
        start_time_s=env.task.schedule['start_time_s'], window_control_steps=window_steps,
        full_sequence_completed=completed.cpu().tolist(),
        all_five_contact_fraction=fraction.cpu().tolist(),
        per_finger_contact_fraction=(per_finger / window_steps).cpu().tolist(),
        finger_order=list(FINGERS), maximum_object_error_mm=(max_error * 1000).cpu().tolist(),
        maximum_pad_force_n=max_force.cpu().tolist(), table_clearance_passed=table_ok.cpu().tolist(),
        joint_constraints_passed=joints_ok.cpu().tolist(),
        five_finger_success=success.cpu().tolist(), success_count=int(success.sum()),
        definition='Full sequence, object mean-keypoint error <= configured success tolerance at every step, '
                   'joint/coupling constraints, nonnegative tabletop clearance, and >=95% simultaneous five-pad/can contact duty in the whole grasp window. '
                   'Failed suffixes count as noncontact; simulation diagnostic, not physical grasp certification.')
    output = Path(output)
    (output / 'five_finger_evaluation.json').write_text(json.dumps(report, indent=2) + '\n')
    np.savez_compressed(output / 'five_finger_evaluation.npz', **{
        name: np.stack([row[name] for row in rows]) for name in rows[0]})
    print('[five-finger evaluation]', json.dumps(report), flush=True)
    return report


def run(args, root, is_running=None, on_ready=None):
    """Use the base runner's explicit callback, leaving its source unchanged."""
    from .runner import run as base_run

    def ready(env, metadata):
        env.reference.validate_can_base_down()
        attach_contact_reward(env)
        # Distinguish this task from generic playback/training even with the
        # same config. An entry point that omits the reward must fail to resume.
        contract = dict(base_contract=metadata['contract_hash'], task=SCHEMA,
                        schedule=env.task.schedule)
        metadata['contract_hash'] = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
        metadata['task'] = dict(schema=SCHEMA, schedule=env.task.schedule,
                               implementation='dedicated demo2 contact task; shared unchanged PhysX environment and RSL-RL PPO')
        metadata['execution']['automatic_replay_output'] = False
        metadata['execution']['learning'] = args.mode == 'train' and on_ready is None
        (args.output / 'run_metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
        if on_ready is not None:
            return on_ready(env, metadata)
        cfg = env.cfg
        learner = PPO(env.task, cfg['ppo'], device=args.policy_device, num_envs=env.num_envs)
        if args.checkpoint:
            saved = learner.load(args.checkpoint, metadata, resume=args.mode == 'train')
            if saved.get('training_state') is None:
                raise ValueError('Complete physics/curriculum state is required')
            env.load_training_state_dict(saved['training_state'])
        if args.mode == 'play':
            from .play import play
            return play(env, learner, args.output, args.episodes, is_running, protocol='strict')
        if args.mode == 'evaluate':
            return evaluate_contacts(env, learner, args.output)
        writer = None
        if args.logger == 'tensorboard':
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(str(args.output / 'tensorboard'))
        print('[demo2] five-finger reward enabled; RSI clock follows reference frame IDs', flush=True)
        env.set_training(True)
        env.reset()
        if cfg['training']['initial_random_episode_length']:
            env.episode_length_buf[:] = torch.randint(env.max_episode_length, (env.num_envs,),
                                                      device=env.device, generator=env.generator)
        obs = env.observation()
        started, first_iteration = time.monotonic(), learner.iteration
        try:
            with (args.output / 'training.jsonl').open('a') as log:
                for i in range(args.iterations):
                    start = time.monotonic()
                    batch, obs, stats = learner.collect(env, obs)
                    collected = time.monotonic()
                    stats.update(learner.update(batch), iteration=learner.iteration, steps=learner.steps,
                        control_steps=env.control_steps, gravity_m_s2=env.gravity.value,
                        gravity_stage=env.gravity.stage, push_events=env.push.total_events,
                        rsi_probabilities=env.sampler.probabilities().cpu().tolist())
                    finished = time.monotonic()
                    stats.update(timing_stats(elapsed=finished-started, completed=i+1, iterations=args.iterations,
                        start_iteration=first_iteration, collection=collected-start, learning=finished-collected,
                        iteration=finished-start, transitions=cfg['ppo']['rollout_steps']*env.num_envs))
                    stats['five_finger_contact'] = contact_summary(stats)
                    log.write(json.dumps(stats) + '\n'); log.flush()
                    print('[train] ' + json.dumps(stats) if args.console == 'json' else format_progress(stats,
                        num_envs=env.num_envs, control_dt=env.task.dt, rollout_steps=cfg['ppo']['rollout_steps'],
                        decimation=cfg['control_decimation'], rsi=cfg['rsi']['enabled'],
                        augmentation=cfg['augmentation']['enabled']), flush=True)
                    if writer:
                        for name, value in scalar_stats(stats).items():
                            writer.add_scalar(name, value, learner.iteration)
                    if learner.iteration % args.save_every == 0:
                        learner.save(args.output / f'model_{learner.iteration}.pt', metadata, env.training_state_dict())
                        learner.save(args.output / 'policy.pt', metadata, env.training_state_dict())
                learner.save(args.output / 'policy.pt', metadata, env.training_state_dict())
        finally:
            if writer: writer.close()
        if not args.skip_evaluation:
            evaluate_contacts(env, learner, args.output)

    return base_run(args, Path(root), on_ready=ready, is_running=is_running)
