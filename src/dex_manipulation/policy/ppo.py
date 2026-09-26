"""General RSL-RL PPO integration; no dependency on REGRIND source.

Keep the tested final-observation timeout correction and complete curriculum/RNG
checkpoint contract around the same general learner used by the reference repo.
"""

from pathlib import Path
import copy
import torch
from tensordict import TensorDict
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from rsl_rl.algorithms import PPO as LibraryPPO
from .progress import RolloutMetrics


class PPO:
    def __init__(self, task, config, device="cpu", num_envs=1):
        self.cfg, self.device, self.num_envs = copy.deepcopy(config), torch.device(device), num_envs
        dummy = TensorDict(
            {
                "policy": torch.zeros(num_envs, task.observation_size, device=device),
                "critic": torch.zeros(num_envs, task.critic_observation_size, device=device),
            },
            batch_size=[num_envs],
        )
        groups = {"actor": ["policy"], "critic": ["critic"]}
        common = dict(
            obs=dummy,
            obs_groups=groups,
            hidden_dims=config["hidden_sizes"],
            activation=config["activation"],
            obs_normalization=config["observation_normalization"],
        )
        self.actor = MLPModel(
            **common,
            obs_set="actor",
            output_dim=task.action_size,
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": config["initial_std"],
                "std_type": "scalar",
            },
        ).to(device)
        self.critic = MLPModel(**common, obs_set="critic", output_dim=1).to(device)
        # Adopt the zero-residual initialization idea through ordinary torch APIs.
        head = [m for m in self.actor.modules() if isinstance(m, torch.nn.Linear)][-1]
        with torch.no_grad():
            head.weight.zero_()
            head.bias.zero_()
        storage = RolloutStorage(
            "rl", num_envs, config["rollout_steps"], dummy, [task.action_size], device=str(device)
        )
        self.algorithm = LibraryPPO(
            self.actor,
            self.critic,
            storage,
            num_learning_epochs=config["epochs"],
            num_mini_batches=config["num_minibatches"],
            clip_param=config["clip_ratio"],
            gamma=config["gamma"],
            lam=config["gae_lambda"],
            value_loss_coef=config["value_coefficient"],
            entropy_coef=config["entropy_coefficient"],
            learning_rate=config["learning_rate"],
            max_grad_norm=config["max_grad_norm"],
            schedule=config["schedule"],
            desired_kl=config["target_kl"],
            use_clipped_value_loss=True,
            normalize_advantage_per_mini_batch=False,
            device=str(device),
        )
        self.optimizer = self.algorithm.optimizer
        self.iteration, self.steps = 0, 0
        self.rollout_metrics = RolloutMetrics(num_envs, self.device)

    def act(self, obs, deterministic=True):
        return self.actor(obs.to(self.device), stochastic_output=not deterministic)

    def collect(self, env, observation):
        self.rollout_metrics.begin()
        with torch.no_grad():
            for _ in range(self.cfg["rollout_steps"]):
                action = self.algorithm.act(observation.to(self.device))
                observation, reward, term, trunc, info = env.step(action.to(env.device))
                observation = observation.to(self.device)
                adjusted = reward.to(self.device).clone()
                timeout = (trunc & ~term).to(self.device)
                if timeout.any():
                    adjusted += (
                        self.cfg["gamma"]
                        * self.critic(info["final_observation"].to(self.device)).squeeze(-1)
                        * timeout
                    )
                # Already corrected with V(final state), so do not pass RSL's V(s_t) timeout branch.
                self.algorithm.process_env_step(
                    observation, adjusted, (term | trunc).to(self.device), {}
                )
                self.rollout_metrics.add(reward, term, trunc, info["metrics"])
            self.algorithm.compute_returns(observation)
        count = self.cfg["rollout_steps"] * env.num_envs
        self.steps += count
        return None, observation, self.rollout_metrics.finish()

    def update(self, _=None):
        stats = self.algorithm.update()
        self.iteration += 1
        return {k: float(v) for k, v in stats.items()} | {
            "learning_rate": self.algorithm.learning_rate,
            "action_std": float(self.actor.output_std.detach().mean()),
        }

    def save(self, path, metadata, training_state=None):
        path = Path(path)
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(
            dict(
                schema=3,
                learner=self.algorithm.save(),
                learning_rate=self.algorithm.learning_rate,
                config=self.cfg,
                iteration=self.iteration,
                steps=self.steps,
                torch_rng_state=torch.get_rng_state(),
                cuda_rng_states=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                metadata=metadata,
                training_state=training_state,
            ),
            temporary,
        )
        temporary.replace(path)

    def load(self, path, expected_metadata, resume=False):
        data = torch.load(path, map_location=self.device, weights_only=True)
        if (
            data.get("schema") != 3
            or data["metadata"]["contract_hash"] != expected_metadata["contract_hash"]
        ):
            raise ValueError(
                "Checkpoint reference/model/control/observation contract mismatch; v2 policies are incompatible"
            )
        if data["config"] != self.cfg:
            raise ValueError("PPO config changed")
        self.algorithm.load(
            data["learner"], {"actor": True, "critic": True, "optimizer": resume}, strict=True
        )
        self.iteration, self.steps = data["iteration"], data["steps"]
        if resume:
            self.algorithm.learning_rate = data["learning_rate"]
            for group in self.optimizer.param_groups:
                group["lr"] = data["learning_rate"]
            torch.set_rng_state(data["torch_rng_state"].cpu())
            if torch.cuda.is_available() and data["cuda_rng_states"]:
                torch.cuda.set_rng_state_all([x.cpu() for x in data["cuda_rng_states"]])
        return data

    def initialize_actor(self, path, metadata):
        """Explicit transfer, not resume: preserve a compatible floating mean actor.

        New arm inputs have zero first-layer weights. Critic, optimizer and
        exploration std are freshly initialized for the new physical task.
        """
        data = torch.load(path, map_location=self.device, weights_only=True)
        old = data["metadata"]
        new = metadata
        if (
            data.get("schema") != 3
            or old.get("observation_schema") != "regrind_revo2_v6_actor67_critic94"
        ):
            raise ValueError("Actor transfer requires the compatible 67-input floating policy")
        for key in ("model_sha256", "reference"):
            if old[key] != new[key]:
                raise ValueError(f"Actor transfer {key} differs; no silent demo/model substitution")
        for key in (
            "observation",
            "reference_phase",
            "reference_velocity_mode",
            "physics_dt",
            "control_decimation",
            "world_frame",
            "residual_translation_m",
            "residual_rotation_rad",
            "residual_joint_rad",
            "table_safety",
        ):
            if old["config"].get(key) != new["config"].get(key):
                raise ValueError(f"Actor transfer contract differs: {key}")
        arm_extension = new.get("observation_schema") == "revo2_arm_actor87_critic114_v1"
        if not arm_extension and new.get("observation_schema") != old["observation_schema"]:
            raise ValueError("Actor transfer observation schema differs")
        source = data["learner"]["actor_state_dict"]
        target = self.actor.state_dict()
        if set(source) != set(target):
            raise ValueError("Actor architecture keys differ")
        for key, value in source.items():
            if key == "distribution.std_param":
                continue
            if key == "mlp.0.weight" and arm_extension:
                if value.shape[1] != 67 or target[key].shape != (value.shape[0], 87):
                    raise ValueError("Expected 67 -> 87 arm actor input extension")
                target[key].zero_()
                target[key][:, :67].copy_(value)
            elif arm_extension and key in ("obs_normalizer._mean", "obs_normalizer._var", "obs_normalizer._std"):
                if value.shape != (1, 67) or target[key].shape != (1, 87):
                    raise ValueError("Unexpected actor normalization dimensions")
                target[key][:, :67].copy_(value)
            elif key == "obs_normalizer.count":
                # Retain the source prior: resetting count abruptly reinterprets
                # the pretrained 67 inputs during the first rollout. The added
                # 20 features already have physical, bounded normalization.
                target[key].copy_(value)
            else:
                if target[key].shape != value.shape:
                    raise ValueError(f"Actor transfer shape mismatch: {key}")
                target[key].copy_(value)
        self.actor.load_state_dict(target, strict=True)
        return dict(
            source_iteration=data["iteration"],
            source_contract_hash=old["contract_hash"],
            transferred="floating actor mean and observation statistics",
            new_inputs="20 zero-weight arm features" if arm_extension else "none",
            critic="fresh",
            optimizer="fresh",
            exploration_std="new training config",
            normalizer_prior_count=float(target["obs_normalizer.count"].item()),
        )

    def initialize_arm_actor(self, path, metadata):
        return self.initialize_actor(path, metadata)

    def export(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        module = self.actor.as_jit().cpu().eval()
        torch.jit.script(module).save(str(directory / "policy_jit.pt"))
        onnx = self.actor.as_onnx(verbose=False).cpu().eval()
        torch.onnx.export(
            onnx,
            onnx.get_dummy_inputs(),
            str(directory / "policy.onnx"),
            opset_version=18,
            input_names=onnx.input_names,
            output_names=onnx.output_names,
            dynamic_axes={"obs": {0: "batch"}, "actions": {0: "batch"}},
            dynamo=False,
        )
