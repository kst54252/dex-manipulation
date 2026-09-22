"""Integer reference restart distribution and source gravity schedule."""

import numpy as np
import torch


class ReferenceStateSampler:
    def __init__(self, duration, control_dt, config, device="cpu"):
        self.duration, self.dt, self.cfg, self.device = duration, control_dt, config, device
        self.frame_count = int(round(duration / control_dt)) + 1
        self.bin_count = int(self.frame_count // (1 / control_dt)) + 1
        self.failures = torch.zeros(self.bin_count, device=device)
        self.episodes = 0
        if config["sampling"] not in ("uniform", "uniform_frames", "adaptive"):
            raise ValueError("Invalid RSI sampler")

    def probabilities(self):
        uniform = torch.ones_like(self.failures) / self.bin_count
        if self.cfg["sampling"] == "uniform_frames":
            # Report the frame distribution aggregated into the legacy bins;
            # the final bin can have one fewer eligible (nonterminal) frame.
            frames = torch.arange(max(self.frame_count - 1, 1), device=self.device)
            bins = (frames * self.bin_count // self.frame_count).clamp(max=self.bin_count - 1)
            counts = torch.bincount(bins, minlength=self.bin_count).float()
            return counts / counts.sum()
        if self.cfg["sampling"] == "uniform":
            return uniform
        p = (1 - self.cfg["uniform_mix"]) * self.failures / self.failures.sum().clamp_min(
            1e-8
        ) + self.cfg["uniform_mix"] * uniform
        p = torch.where(self.failures.sum() > 1e-8, p, uniform)
        if self.cfg["kernel_size"] > 1:
            kernel = self.cfg["kernel_decay"] ** torch.arange(
                self.cfg["kernel_size"], device=self.device
            )
            padded = torch.nn.functional.pad(p[None, None], (0, len(kernel) - 1), mode="replicate")
            p = torch.nn.functional.conv1d(padded, (kernel / kernel.sum())[None, None]).flatten()
        return p / p.sum()

    def sample(self, count, generator):
        if self.cfg["sampling"] == "uniform_frames":
            # The floating Revo2 source samples each nonterminal frame directly.
            # Keep the legacy bin sampler for reproducing older checkpoints.
            indices = torch.randint(
                max(self.frame_count - 1, 1), (count,), device=self.device, generator=generator
            )
            if not self.cfg["enabled"]:
                indices.zero_()
            return indices * self.dt
        bins = torch.multinomial(self.probabilities(), count, replacement=True, generator=generator)
        u = torch.rand(count, device=self.device, generator=generator)
        indices = ((bins + u) / self.bin_count * (self.frame_count - 1)).long()
        if not self.cfg["enabled"]:
            indices.zero_()
        return indices * self.dt

    def record(self, times, terminated, demo_end, num_envs=None, max_episode_steps=None):
        bins = ((times / self.dt).round().long() * self.bin_count // self.frame_count).clamp(
            0, self.bin_count - 1
        )
        self.episodes += int((terminated | demo_end).sum())
        if self.cfg["sampling"] != "adaptive":
            return
        counts = torch.bincount(bins[terminated & ~demo_end], minlength=self.bin_count).float()
        effective_alpha = min(
            self.cfg["ema_alpha"]
            * (num_envs or len(times))
            / (max_episode_steps or (self.frame_count - 1)),
            1.0,
        )
        self.failures.mul_(1 - effective_alpha).add_(counts, alpha=effective_alpha)

    def state_dict(self):
        return dict(failures=self.failures.cpu().tolist(), episodes=self.episodes)

    def load_state_dict(self, state):
        values = torch.tensor(state["failures"], device=self.device)
        if (
            values.shape != self.failures.shape
            or not torch.isfinite(values).all()
            or (values < 0).any()
        ):
            raise ValueError("Invalid RSI histogram")
        self.failures.copy_(values)
        self.episodes = int(state["episodes"])


class GravityCurriculum:
    def __init__(self, config, evaluation_gravity):
        if config["step_unit"] != "vector_environment_control_step":
            raise ValueError("Gravity thresholds must use vector environment control steps")
        self.cfg, self.evaluation_gravity = config, evaluation_gravity
        self.stages = np.asarray(config["stages"], dtype=float)
        if (
            self.stages.ndim != 2
            or self.stages.shape[1] != 3
            or self.stages[0, 0] != 0
            or not np.isfinite(self.stages).all()
            or np.any(np.diff(self.stages[:, 0]) <= 0)
            or np.any(self.stages[:, 1] < 0)
            or np.any(self.stages[:, 1] > self.stages[:, 2])
            or np.any(np.diff(self.stages[:, 1:], axis=0) < 0)
        ):
            raise ValueError(
                "Gravity stages must be increasing [control_step, min, max] starting at zero"
            )
        self.value = float(evaluation_gravity)
        self.stage = 0

    def bounds(self, control_steps):
        if control_steps < 0:
            raise ValueError("Control step count cannot be negative")
        index = int(np.searchsorted(self.stages[:, 0], control_steps, side="right") - 1)
        return index, self.stages[index, 1], self.stages[index, 2]

    def sample(self, control_steps, rng, training):
        self.stage, low, high = self.bounds(control_steps)
        self.value = (
            float(rng.uniform(low, high))
            if training and self.cfg["enabled"]
            else float(self.evaluation_gravity)
        )
        return self.value

    def state_dict(self):
        return dict(value=self.value, stage=self.stage)

    def load_state_dict(self, state):
        value, stage = float(state["value"]), int(state["stage"])
        if not np.isfinite(value) or value < 0 or not 0 <= stage < len(self.stages):
            raise ValueError("Invalid gravity checkpoint")
        self.value, self.stage = value, stage
