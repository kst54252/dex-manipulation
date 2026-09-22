"""One PPO training loop for floating, arm and optional contact objectives."""

import json
import os
import time

import torch

from .progress import format_progress, scalar_stats, timing_stats


def train_policy(env, learner, args, metadata):
    config = env.cfg
    writer = remote_log = None
    try:
        if args.logger == "tensorboard":
            from torch.utils.tensorboard import SummaryWriter

            writer = SummaryWriter(str(args.output / "tensorboard"))
        elif args.logger == "wandb":
            import wandb

            remote_log = wandb.init(
                project="dex-manipulation",
                name=args.output.name,
                config=config,
                dir=str(args.output),
            )
        print("[startup] reset training environments", flush=True)
        env.set_training(True)
        env.reset()
        if config["training"]["initial_random_episode_length"]:
            env.episode_length_buf[:] = torch.randint(
                env.max_episode_length, (env.num_envs,), device=env.device, generator=env.generator
            )
        observation = env.observation()
        started, start_iteration = time.monotonic(), learner.iteration
        with (args.output / "training.jsonl").open("a") as log:
            for iteration_in_run in range(args.iterations):
                if iteration_in_run == 0:
                    print("[startup] collect first PPO rollout", flush=True)
                iteration_started = time.monotonic()
                batch, observation, stats = learner.collect(env, observation)
                collected = time.monotonic()
                if iteration_in_run == 0:
                    print("[startup] optimize first PPO rollout", flush=True)
                stats.update(
                    learner.update(batch),
                    iteration=learner.iteration,
                    steps=learner.steps,
                    control_steps=env.control_steps,
                    gravity_m_s2=env.gravity.value,
                    gravity_stage=env.gravity.stage,
                    push_events=env.push.total_events,
                    rsi_probabilities=env.sampler.probabilities().cpu().tolist(),
                )
                if config.get("arm_training", {}).get("enabled"):
                    stats["arm_reset_ik_rejections_total"] = env.reset_ik_rejections
                if config.get("five_finger_grasp"):
                    from .contact_reward import contact_summary

                    stats["five_finger_contact"] = contact_summary(stats)
                finished = time.monotonic()
                stats.update(
                    timing_stats(
                        elapsed=finished - started,
                        completed=iteration_in_run + 1,
                        iterations=args.iterations,
                        start_iteration=start_iteration,
                        collection=collected - iteration_started,
                        learning=finished - collected,
                        iteration=finished - iteration_started,
                        transitions=config["ppo"]["rollout_steps"] * env.num_envs,
                    )
                )
                log.write(json.dumps(stats) + "\n")
                log.flush()
                if args.console == "json":
                    print("[train]", json.dumps(stats), flush=True)
                else:
                    print(
                        format_progress(
                            stats,
                            num_envs=env.num_envs,
                            control_dt=env.task.dt,
                            rollout_steps=config["ppo"]["rollout_steps"],
                            decimation=config["control_decimation"],
                            rsi=config["rsi"]["enabled"],
                            augmentation=config["augmentation"]["enabled"],
                        ),
                        flush=True,
                    )
                scalars = scalar_stats(stats)
                if writer:
                    for key, value in scalars.items():
                        writer.add_scalar(key, value, learner.iteration)
                if remote_log:
                    remote_log.log(scalars, step=learner.iteration)
                if learner.iteration % args.save_every == 0:
                    checkpoint = args.output / f"model_{learner.iteration}.pt"
                    learner.save(checkpoint, metadata, env.training_state_dict())
                    latest = args.output / "policy.pt.tmp"
                    latest.unlink(missing_ok=True)
                    os.link(checkpoint, latest)
                    latest.replace(args.output / "policy.pt")
            learner.save(args.output / "policy.pt", metadata, env.training_state_dict())
    finally:
        if writer:
            writer.close()
        if remote_log:
            remote_log.finish()
