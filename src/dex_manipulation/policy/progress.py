"""Rollout statistics and terminal reporting, independent of the simulator API."""
from collections import deque
from datetime import timedelta
import math

import torch


class RolloutMetrics:
    """Accumulate on device; transfer summaries once per rollout, not per step."""

    def __init__(self, num_envs, device, episode_window=100):
        self.returns = torch.zeros(num_envs, device=device)
        self.lengths = torch.zeros_like(self.returns)
        self.episodes = deque(maxlen=episode_window)
        self.begin()

    def begin(self):
        self.names = None
        self.sums = self.maxima = None
        self.reward_sum = torch.zeros_like(self.returns[0])
        self.counts = torch.zeros(4, device=self.returns.device)
        self.samples = 0
        self.finished = []
        self.done_masks = []

    def add(self, reward, terminated, truncated, metrics):
        names = tuple(metrics)
        if self.names is None:
            self.names = names
        elif names != self.names:
            raise ValueError('Rollout metric schema changed within an iteration')
        values = torch.stack([metrics[k].detach().to(self.returns.device, torch.float32) for k in names])
        sums, maxima = values.sum(dim=1), values.amax(dim=1)
        self.sums = sums if self.sums is None else self.sums + sums
        self.maxima = maxima if self.maxima is None else torch.maximum(self.maxima, maxima)
        reward = reward.detach().to(self.returns.device)
        terminated, truncated = terminated.to(reward.device), truncated.to(reward.device)
        failure = metrics['early_failure'].to(reward.device)
        complete = metrics['demo_end'].to(reward.device) & ~failure
        done = terminated | truncated
        self.counts += torch.stack([failure.sum(), complete.sum(), (truncated & ~terminated).sum(), done.sum()])
        self.reward_sum += reward.sum()
        self.samples += reward.numel()
        self.returns += reward
        self.lengths += 1
        self.finished.append(torch.stack((self.returns, self.lengths), dim=-1))
        self.done_masks.append(done)
        self.returns.masked_fill_(done, 0)
        self.lengths.masked_fill_(done, 0)

    def finish(self):
        if not self.samples:
            raise ValueError('Cannot report an empty rollout')
        rows = torch.stack((self.sums / self.samples, self.maxima), dim=-1).cpu().tolist()
        metrics = {name: dict(mean=row[0], max=row[1]) for name, row in zip(self.names, rows)}
        failures, completed, timeouts, episodes, reward = torch.cat((self.counts, self.reward_sum[None] / self.samples)).cpu().tolist()
        # Select on device before readback. Existing unfinished episodes carry
        # across rollout boundaries; only completed episodes enter the window.
        ended = torch.stack(self.finished)[torch.stack(self.done_masks)][-self.episodes.maxlen:].cpu().tolist()
        self.episodes.extend(ended)
        window = len(self.episodes)
        return dict(mean_object_error_m=metrics['object_keypoint_error_m']['mean'],
                    mean_step_reward=reward, failures=int(failures), completed=int(completed), timeouts=int(timeouts),
                    metrics=metrics, episode=dict(count=int(episodes), window_count=window,
                        mean_return=sum(e[0] for e in self.episodes)/window if window else None,
                        mean_length_steps=sum(e[1] for e in self.episodes)/window if window else None))


def timing_stats(*, elapsed, completed, iterations, start_iteration, collection, learning, iteration, transitions):
    """ETA uses this invocation's completed iterations, including on resume."""
    if not 1 <= completed <= iterations:
        raise ValueError('Invalid iteration progress')
    remaining = elapsed / completed * (iterations - completed)
    return dict(iteration_in_run=completed, iterations_in_run=iterations,
                target_iteration=start_iteration + iterations,
                elapsed_s=elapsed, eta_s=remaining, estimated_total_s=elapsed + remaining,
                collection_s=collection, learning_s=learning, iteration_s=iteration,
                transitions_per_s=transitions / max(collection + learning, 1e-9))


def scalar_stats(stats, prefix=''):
    """Flatten all numerical diagnostics for TensorBoard and W&B."""
    result = {}
    for key, value in stats.items():
        name = f'{prefix}/{key}' if prefix else key
        if isinstance(value, dict):
            result.update(scalar_stats(value, name))
        elif isinstance(value, (int, float)):
            result[name] = value
        elif isinstance(value, list):
            result.update(scalar_stats(dict(enumerate(value)), name))
    return result


def _number(value):
    if value is None:
        return 'pending'
    return f'{value:.6g}'


def _duration(seconds):
    return str(timedelta(seconds=round(seconds))) if math.isfinite(seconds) else str(seconds)


def format_progress(stats, *, num_envs, control_dt, rollout_steps, decimation, rsi, augmentation):
    width = 92
    line = '=' * width
    lines = [line, f"Learning iteration {stats['iteration']}/{stats['target_iteration']}"
             f"  |  run {stats['iteration_in_run']}/{stats['iterations_in_run']}  |  {num_envs:,} envs",
             f"Transitions {stats['steps']:,}  |  {stats['transitions_per_s']:,.0f} steps/s"
             f"  |  rollout {rollout_steps} control / {rollout_steps * decimation} physics steps",
             f"Collection {stats['collection_s']:.3f}s  |  Learning {stats['learning_s']:.3f}s"
             f"  |  Iteration {stats['iteration_s']:.3f}s",
             f"Elapsed {_duration(stats['elapsed_s'])}  |  ETA {_duration(stats['eta_s'])}"
             f"  |  Estimated total {_duration(stats['estimated_total_s'])}", '-' * width]
    lines.append(f"Step reward {_number(stats['mean_step_reward'])}  |  Failures {stats['failures']}"
                 f"  |  Demo ends {stats['completed']}  |  Timeouts {stats['timeouts']}")
    episode = stats['episode']
    lines.append(f"Episode return {_number(episode['mean_return'])}  |  Length {_number(episode['mean_length_steps'])} steps"
                 f"  |  window {episode['window_count']}/100; ended this rollout {episode['count']}")
    loss_keys = ('value', 'surrogate', 'entropy', 'learning_rate', 'action_std')
    lines.append('  |  '.join(f'{k} {_number(stats[k])}' for k in loss_keys[:3] if k in stats))
    lines.append('  |  '.join(f'{k} {_number(stats[k])}' for k in loss_keys[3:] if k in stats))
    lines.append(f"Gravity {stats['gravity_m_s2']:.3f} m/s^2 (stage {stats['gravity_stage']})"
                 f"  |  Control steps {stats['control_steps']:,}  |  Pushes {stats['push_events']}")
    lines.append(f"RSI {'ON' if rsi else 'OFF'}  |  Augmentation {'ON' if augmentation else 'OFF'}")
    lines.append(f"RSI bins: {', '.join(f'{p:.3f}' for p in stats['rsi_probabilities'])}")
    lines.extend(['-' * width, f"{'Rollout metrics (all environments and steps)':<55} {'mean':>16} {'max':>16}"])
    for key, values in stats['metrics'].items():
        if key.startswith('reward_'):
            continue
        lines.append(f"{key:<55} {_number(values['mean']):>16} {_number(values['max']):>16}")
    lines.extend(['-' * width, f"{'Weighted reward contributions (including control dt)':<55} {'mean/step':>16} {'max/step':>16}"])
    for key, values in stats['metrics'].items():
        if key.startswith('reward_'):
            lines.append(f"{key:<55} {_number(values['mean'] * control_dt):>16} {_number(values['max'] * control_dt):>16}")
    # Future scalar PPO losses remain visible without having to edit this table.
    known = set(loss_keys) | {'metrics', 'episode', 'mean_object_error_m', 'mean_step_reward', 'failures', 'completed',
        'timeouts', 'iteration', 'steps', 'control_steps', 'gravity_m_s2', 'gravity_stage', 'push_events', 'rsi_probabilities',
        'iteration_in_run', 'iterations_in_run', 'target_iteration', 'elapsed_s', 'eta_s', 'estimated_total_s',
        'collection_s', 'learning_s', 'iteration_s', 'transitions_per_s'}
    lines.extend(f'{k}: {v}' for k, v in stats.items() if k not in known)
    lines.append(line)
    return '\n'.join(lines)
