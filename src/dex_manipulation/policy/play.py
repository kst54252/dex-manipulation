"""Visible, deterministic checkpoint inference with dynamic objects and honest resets."""
import json
from pathlib import Path
import time

import numpy as np
import torch

from .math3d import numpy_tree


def play(env, learner, output, episodes=0, is_running=None, protocol='strict'):
    """Repeat single-env episodes; never optimize or overwrite a checkpoint.

    The first episode is recorded for inspection, including its terminal step.
    Subsequent episodes keep only summaries. Failure triggers a visible restart,
    not a hidden object teleport inside the recorded episode.
    """
    if env.num_envs != 1 or protocol not in ('strict', 'source'):
        raise ValueError('Play requires one environment and a strict/source protocol')
    if is_running is None and episodes == 0:
        raise ValueError('Unbounded play requires an application lifetime callback')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    running = is_running or (lambda: True)
    learner.actor.eval()
    learner.critic.eval()
    env.evaluation_protocol = protocol
    if protocol == 'source':
        env.cfg['failure_object_error_m'] = 1.0
    env.set_training(False)
    manifest = dict(checkpoint_iteration=learner.iteration, training=False,
                    num_envs=1, render=env.render, deterministic=True, protocol=protocol,
                    gravity_m_s2=env.gravity.value, hand_gravity=env.cfg['hand_gravity'],
                    self_collision=env.cfg['self_collision'], rsi=False,
                    augmentation=protocol == 'source' and env.cfg['augmentation']['enabled'],
                    robot=env.metadata.get('robot','floating Revo2'),
                    startup_physics=env.metadata.get('restored_physics','checkpoint environment 0'),
                    deployment_physics=env.metadata,
                    motion_control=env.motion_controller.config,
                    table_safety=env.table_safety.config if getattr(env,'table_safety',None) is not None else None,
                    motion_control_differs_from_training=env.motion_controller.config!=env.cfg.get('motion_control'),
                    command_timestamp='physics_time_s is elapsed simulation time; time_s is reference command time',
                    reset_policy='restart from first frame after termination/timeout',
                    interpretation='dynamic policy playback; completion alone is not grasp success')
    (output / 'playback.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print('[play]', json.dumps(manifest), flush=True)
    episode = 0
    stopped = False
    captures = []
    with (output / 'episodes.jsonl').open('w') as log, torch.no_grad():
        while running() and not stopped and (episodes == 0 or episode < episodes):
            if env.render and env.world.is_stopped():
                break
            env.reset(randomize=protocol == 'source')
            reset_count = env.object_reset_count
            obs = env.observation()
            rows, steps, episode_return = [], 0, 0.0
            max_error, constraints_ok, done = 0.0, True, False
            while running():
                # Respect the Kit pause button without stepping the physics world.
                if env.render and not env.world.is_playing():
                    if env.world.is_stopped():
                        stopped = True
                        break
                    env.world.render()
                    time.sleep(0.02)
                    continue
                action = learner.act(obs, deterministic=True)
                obs, reward, term, trunc, info = env.step(action, auto_reset=False)
                if env.object_reset_count != reset_count:
                    raise AssertionError('Object reset during an active playback episode')
                metrics = {k: float(v[0]) for k, v in info['metrics'].items()}
                steps += 1
                if env.render and episode == 0 and steps in (60, 180) and running():
                    from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport
                    captures.append(capture_viewport_to_file(get_active_viewport(),
                                    str((output / f'preview_{steps:03d}.png').resolve())))
                episode_return += float(reward[0])
                max_error = max(max_error, metrics['object_keypoint_error_m'])
                step_constraints_ok = (metrics['joint_limit_violation_rad'] < 1e-3
                                       and metrics['joint_velocity_violation_rad_s'] < 1e-2
                                       and metrics['coupling_error_rad'] < 1e-2)
                if 'arm_ik_success' in metrics:
                    options=env.ik.options if hasattr(env,'ik') else env.bridge.solver.options
                    step_constraints_ok &= (metrics['arm_ik_success'] > .5
                                            and metrics['arm_joint_velocity_violation_rad_s'] < 1e-2
                                            and metrics['arm_joint_limit_violation_rad'] < 1e-3
                                            and metrics['arm_actual_sigma_min'] >= options.singular_sigma_min
                                            and metrics['arm_actual_condition'] <= options.max_condition)
                constraints_ok &= step_constraints_ok
                table_ok=metrics.get('table_clearance_m',float('inf')) >= 0.
                constraints_ok &= table_ok
                if episode == 0:
                    rows.append(dict(time_s=float(info['reference_time'][0]),
                                     physics_time_s=steps*env.task.dt,
                                     joint_constraints_valid=step_constraints_ok,
                                     table_clearance_valid=table_ok,
                                     valid=step_constraints_ok and table_ok and not metrics['early_failure'],
                                     tracking_within_tolerance=metrics['object_keypoint_error_m'] < env.cfg['success_object_error_m'],
                                     action=numpy_tree(action[0]),
                                     **{k: numpy_tree(v[0]) for k, v in info['state'].items()},
                                     **{'reference_' + k: numpy_tree(v[0]) for k, v in info['reference'].items()},
                                     **{'raw_target_' + k: numpy_tree(v[0]) for k,v in info['raw_targets'].items()},
                                     **{'applied_target_' + k: numpy_tree(v[0]) for k,v in info['applied_targets'].items()},
                                     **{'metric_' + k: v for k, v in metrics.items()}))
                done = bool(term[0] or trunc[0])
                if done:
                    break
            if not steps:
                break
            success = (done and not metrics['early_failure'] and bool(metrics['demo_end'])
                       and constraints_ok and max_error < env.cfg['success_object_error_m']
                       and abs(metrics['object_height_m'] - metrics['target_object_height_m'])
                       < env.cfg['success_final_height_error_m'])
            reason = ('stopped' if stopped else 'window_closed' if not done else 'arm_ik_failure' if metrics.get('arm_ik_success',1.) < .5 else 'early_failure' if metrics['early_failure']
                      else 'demo_end' if metrics['demo_end'] else 'timeout')
            report = dict(episode=episode + 1, steps=steps, return_value=episode_return,
                          end=reason, tracking_success=bool(success), constraints_passed=bool(constraints_ok),
                          maximum_object_error_m=max_error, terminal_metrics=metrics,
                          object_writes_during_episode=env.object_reset_count - reset_count)
            log.write(json.dumps(report) + '\n')
            log.flush()
            print('[play episode]', json.dumps(report), flush=True)
            if rows:
                np.savez_compressed(output / 'first_episode.npz',
                                    **{k: np.asarray([row[k] for row in rows]) for k in rows[0]})
            episode += 1
