"""Can-picking playback, training configuration and optional grasp rewards."""

import hashlib
import json
import math
from copy import deepcopy
from datetime import datetime
from pathlib import Path
import sys

from ..data import resolve_demo_path
from ..policies import demo_id, latest_policy
from .registry import DEFAULT_TASK, config_task_id, load_task


def attach_reward(env, metadata, training_reference):
    from ..policy.contact_reward import SCHEMA, contact_schedule

    dense = bool(env.cfg.get("grasp_task"))
    contact_only = bool(env.cfg.get("five_finger_grasp"))
    if dense and contact_only:
        raise ValueError("Choose one grasp objective; contact rewards must not be counted twice")
    if dense:
        from ..policy.grasp_reward import attach_grasp_task

        attach_grasp_task(env)
    elif contact_only:
        from ..policy.contact_reward import attach_contact_reward

        attach_contact_reward(env)
        # Legacy dedicated demo2 runs added this task signature after computing
        # the base contract. Keep it byte-for-byte, even when playback is retimed.
        schedule = contact_schedule(training_reference, env.cfg["five_finger_grasp"])
        contract = dict(base_contract=metadata["contract_hash"], task=SCHEMA, schedule=schedule)
        metadata["contract_hash"] = hashlib.sha256(
            json.dumps(contract, sort_keys=True).encode()
        ).hexdigest()
        metadata["task"] = dict(
            schema=SCHEMA,
            schedule=schedule,
            implementation="dedicated demo2 contact task; shared unchanged PhysX environment and RSL-RL PPO",
        )


def plan_playback(root, args, catalog):
    """Pure input/command resolution: never import Isaac or write any files."""
    from ..launch import read, arm_reference_error

    def required(path):
        path = resolve_demo_path(Path(path).expanduser(), root)
        if not path.is_file():
            raise ValueError(f"필수 파일이 없습니다: {path}")
        return path

    commands = []
    random_can = getattr(args, "random_can", False)
    placement_seed = getattr(args, "placement_seed", None)
    if random_can and (args.robot != "arm" or args.mode != "policy"):
        raise ValueError("--random-can은 데모2 arm policy에서만 지원합니다.")
    if placement_seed is not None and (not random_can or placement_seed < 0):
        raise ValueError("--placement-seed는 --random-can과 함께 0 이상의 정수로 지정하세요.")
    requested = args.policy or args.selection or "1"
    selected = catalog.get("aliases", {}).get(requested, requested)
    entry = catalog["policies"].get(selected) if args.mode == "policy" else None
    speed = getattr(args, "speed", None)
    if speed is None:
        speed = catalog.get("playback_speed", {}).get(args.robot, 1.0)
        if isinstance(speed, dict):
            speed = speed.get(args.mode, 1.0)
        if entry:
            speed = entry.get("playback_speed", speed)
    if not math.isfinite(speed) or speed <= 0:
        raise ValueError("--speed는 양의 유한한 수여야 합니다.")
    if args.robot == "arm" and args.mode == "retarget" and speed != 1.0:
        raise ValueError(
            "팔 retarget은 검증된 IK 궤적의 1배속을 사용합니다. 팔 정책은 --speed를 지원합니다."
        )
    label = Path(selected).stem
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    task_id = getattr(args, "task", DEFAULT_TASK)
    output = root / "local/results/play" / task_id / f"{stamp}_{args.robot}_{args.mode}_{label}"
    details = dict(
        task=task_id,
        robot=args.robot,
        mode=args.mode,
        selected=selected,
        output=str(output),
        training=False,
        repeat=args.repeat,
        unlimited=args.repeat == 0,
        speed=speed,
    )
    flags = ["--headless"] if args.headless else []
    flags += ["--speed", str(speed)]
    if args.mode == "policy":
        automatic = (
            latest_policy(root, entry["latest_demo"], args.robot, task=task_id)
            if entry and "latest_demo" in entry
            else None
        )
        checkpoint = required(
            automatic["checkpoint"] if automatic else (entry["checkpoint"] if entry else selected)
        )
        if automatic:
            details["policy_selection"] = automatic
        config_path = required(args.config or checkpoint.parent / "config.resolved.json")
        cfg = read(config_path)
        if config_task_id(cfg) != task_id:
            raise ValueError(f"정책 task는 {config_task_id(cfg)}입니다. --task를 맞춰 지정하세요.")
        native_arm = cfg.get("arm_training", {}).get("enabled", False)
        if native_arm and args.robot != "arm":
            raise ValueError(
                "이 정책은 실제 팔 상태를 관측하며 학습했습니다. arm 환경으로 실행하세요."
            )
        reference_path = required(cfg["reference"])
        for key in ("model", "hand_asset", "object_asset", "object_geometry"):
            required(cfg[key])
        cmd = [
            sys.executable,
            str(root / "scripts/policy.py"),
            "--mode",
            "play",
            "--robot",
            args.robot,
            "--checkpoint",
            str(checkpoint),
            "--config",
            str(config_path),
            "--episodes",
            str(args.repeat),
            "--evaluation-protocol",
            "strict",
            "--motion-control",
            "checkpoint",
            "--output",
            str(output),
            *flags,
        ]
        cmd.extend(
            ["--table-safety", args.table_safety or ("checkpoint" if native_arm else "protect")]
        )
        if getattr(args, "contact_materials", None):
            cmd.extend(["--contact-materials", args.contact_materials])
        if args.robot == "arm":
            arm_path = args.arm_config or (
                cfg["arm_training"]["arm_config"]
                if native_arm
                else (entry["arm_config"] if entry else None)
            )
            if arm_path is None and demo_id(root, cfg) in catalog["demos"]:
                arm_path = catalog["demos"][demo_id(root, cfg)]["arm_config"]
            if arm_path is None:
                # Fresh training uses a demo's current derived input. Historical
                # checkpoints still resolve through their registered policy below.
                ref = resolve_demo_path(cfg["reference"], root).resolve()
                for demo in catalog["demos"].values():
                    candidates = [demo["config"]]
                    if demo.get("training_config"):
                        candidates.append(demo["training_config"])
                    if any(
                        resolve_demo_path(read(root / path)["reference"], root).resolve() == ref
                        for path in candidates
                    ):
                        arm_path = demo["arm_config"]
                        break
            if arm_path is None:
                # Recognize custom checkpoints that still use a registered demo's
                # exact input; never guess a coordinate transform for new inputs.
                ref = resolve_demo_path(cfg["reference"], root).resolve()
                for candidate in catalog["policies"].values():
                    if "checkpoint" not in candidate:
                        continue
                    saved = root / candidate["checkpoint"]
                    config = saved.parent / "config.resolved.json"
                    if (
                        config.is_file()
                        and resolve_demo_path(read(config)["reference"], root).resolve() == ref
                    ):
                        arm_path = candidate["arm_config"]
                        break
            if arm_path is None:
                raise ValueError(
                    "이 사용자 정책의 팔 배치를 알 수 없습니다. --arm-config <팔 설정.json>을 지정하세요."
                )
            arm_path = required(arm_path)
            for key in ("alignment", "workcell", "arm_model", "hand_model", "usd"):
                required(read(arm_path)[key])
            cmd.extend(["--arm-config", str(arm_path)])
            details["arm_config"] = str(arm_path)
            if random_can:
                from ..policy.placement import load_random_placement

                placement = load_random_placement(root, cfg, read(arm_path), speed, placement_seed)
                # Choose entropy once in the launcher and forward it, including
                # dry-run commands, so the logged plan exactly reproduces play.
                cmd.extend(["--random-can", "--placement-seed", str(placement.seed)])
                details["random_can"] = placement.metadata
        details.update(
            checkpoint=str(checkpoint),
            config=str(config_path),
            reference=str(reference_path.resolve()),
            label=entry["label"] if entry else checkpoint.name,
        )
        if automatic:
            details["label"] += f" · {automatic['iteration']}iter"
        if entry and entry.get("note"):
            details["note"] = entry["note"]
        commands.append(cmd)
    else:
        if selected not in catalog["demos"]:
            raise ValueError("리타게팅 데모는 1 또는 2를 선택하세요.")
        entry = catalog["demos"][selected]
        config = required(entry["config"])
        arm_config = required(args.arm_config or entry["arm_config"])
        reference = required(read(arm_config)["input"])
        cmd = [
            sys.executable,
            str(root / "scripts/physics.py"),
            args.robot,
            "--config",
            str(config),
            "--arm-config",
            str(arm_config),
            "--loops",
            str(args.repeat),
            "--output",
            str(output),
            *flags,
        ]
        if args.robot == "arm":
            cached = root / entry["arm_reference"]
            reason = arm_reference_error(root, cached, arm_config)
            if reason:
                # A separate cache avoids overwriting previously validated runs.
                cache = root / "local/results/play/ik" / task_id / selected
                cached = cache / "trajectory.npz"
                if arm_reference_error(root, cached, arm_config):
                    commands.append(
                        [
                            sys.executable,
                            str(root / "scripts/ik.py"),
                            "solve",
                            "--config",
                            str(arm_config),
                            "--output",
                            str(cache),
                        ]
                    )
                    details["prepare_ik"] = reason
            cmd.extend(["--arm-reference", str(cached)])
            details["arm_reference"] = str(cached)
        commands.append(cmd)
        details.update(
            config=str(config),
            arm_config=str(arm_config),
            reference=str(reference),
            label=entry["label"],
        )
    return dict(**details, commands=commands)


def plan_training(root, args):
    """Read inputs only: no Isaac import, checkpoint mutation, or file creation."""
    from ..launch import read
    from ..training import scale_gravity

    root = Path(root).resolve()
    task_id = getattr(args, "task", None) or DEFAULT_TASK
    task = load_task(root, task_id)

    def required(path):
        path = resolve_demo_path(Path(path).expanduser(), root)
        if not path.is_file():
            raise ValueError(f"필수 파일이 없습니다: {path}")
        return path

    for name in ("iterations", "num_envs", "save_every"):
        value = getattr(args, name)
        if value is not None and value < 1:
            raise ValueError(f"--{name.replace('_', '-')}는 양수여야 합니다.")
    checkpoint = required(args.resume) if args.resume else None
    actor = required(args.initialize_actor) if args.initialize_actor else None
    if checkpoint and actor:
        raise ValueError("--resume와 --initialize-actor를 함께 사용할 수 없습니다.")
    custom_config = getattr(args, "config", None)
    if checkpoint and custom_config:
        raise ValueError("--resume는 저장된 설정을 사용합니다. --config로 덮어쓸 수 없습니다.")
    if checkpoint:
        source = required(checkpoint.parent / "config.resolved.json")
        config = read(source)
        robot = "arm" if config.get("arm_training", {}).get("enabled", False) else "floating"
        reference = required(config["reference"]).resolve()
        demo = next(
            (n for n in ("1", "2") if reference.is_relative_to(root / f"data/can_grasping/demo{n}")), None
        )
        # Derived contact references are deliberately stored under local/;
        # their new configs explicitly identify the original demo.
        if demo is None and config.get("demo_id") in ("1", "2"):
            demo = config["demo_id"]
        if demo is None:
            raise ValueError(
                "checkpoint 입력이 demo1/demo2에 속하지 않습니다. scripts/policy.py로 직접 설정하세요."
            )
        if (args.robot and args.robot != robot) or (args.demo and args.demo != demo):
            raise ValueError(
                f"checkpoint는 {robot}, {demo}번 데모입니다. 다른 환경/데모로 이어 학습할 수 없습니다."
            )
        saved_metadata = checkpoint.parent / "run_metadata.json"
        saved_envs = (
            read(saved_metadata).get("physics", {}).get("num_envs")
            if saved_metadata.is_file()
            else None
        ) or config["training"]["num_envs"]
        if args.num_envs is not None and args.num_envs != saved_envs:
            raise ValueError(
                f"이어 학습은 저장된 환경 수 {saved_envs}개를 유지해야 합니다 (환경별 물리 상태 복원)."
            )
        num_envs = saved_envs
    else:
        if not args.robot or not args.demo:
            raise ValueError(
                "환경과 데모를 지정하세요. 예: ./train.sh floating 2 --iterations 2000"
            )
        robot, demo = args.robot, args.demo
        demos = task.catalog()["demos"]
        if demo not in demos:
            raise ValueError(f"{task.id}: 등록된 데모는 {', '.join(demos)}입니다.")
        catalog = demos[demo]
        source = required(
            custom_config
            or (
                catalog.get("training_config", catalog["config"])
                if robot == "floating"
                else task.definition["arm_training_config"]
            )
        )
        config = deepcopy(read(source))
        if custom_config:
            configured_robot = (
                "arm" if config.get("arm_training", {}).get("enabled") else "floating"
            )
            if (
                configured_robot != robot
                or demo_id(root, config) != demo
                or config_task_id(config) != task.id
            ):
                raise ValueError("선택한 --config의 로봇 환경/데모가 명령과 다릅니다.")
        if robot == "arm" and not custom_config:
            # The arm template owns physical/controller/PPO settings; only the
            # selected demo's reference, coordinate description and placement vary.
            demo_config = read(required(catalog["config"]))
            config["reference"] = demo_config["reference"]
            config["scene_assumption"] = demo_config["scene_assumption"]
            config["arm_training"]["arm_config"] = catalog["arm_config"]
            if demo_config["reference"] != read(source)["reference"]:
                config["recipe"] = f"revo2_rb3_online_ik_demo{demo}_v1"
        num_envs = args.num_envs or config["training"]["num_envs"]
    if actor and robot != "arm":
        raise ValueError("--initialize-actor는 새 arm 학습에서만 사용하세요.")
    iterations = args.iterations or config["training"]["iterations"]
    save_every = args.save_every or config["training"]["save_every"]
    if not checkpoint:
        config["task_id"] = task.id
        config["demo_id"] = demo
        scale_gravity(config, iterations)
        config["training"].update(iterations=iterations, num_envs=num_envs, save_every=save_every)
    # Resumes pass execution counts as CLI args only. Editing the snapshot would
    # invalidate the strict checkpoint contract and restart curriculum timing.
    reference = required(config["reference"])
    for key in ("model", "hand_asset", "object_asset", "object_geometry"):
        required(config[key])
    arm_path = None
    if robot == "arm":
        arm_path = required(config["arm_training"]["arm_config"])
        for key in ("alignment", "workcell", "arm_model", "hand_model", "usd"):
            required(read(arm_path)[key])
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output = args.output or Path(
        f"local/results/policy/{task.id}/demo{demo}_{robot}_{iterations}_{stamp}"
    )
    output = Path(output).expanduser()
    output = (root / output).resolve()
    if not output.is_relative_to(root / "local"):
        raise ValueError("학습 결과는 Git 제외 영역인 local/ 아래에 저장하세요.")
    if output.exists():
        raise ValueError(f"결과를 덮어쓸 수 없습니다. 새 --output 폴더를 지정하세요: {output}")
    snapshot = output / "config.input.json"
    command = [
        sys.executable,
        "-u",
        str(root / "scripts/policy.py"),
        "--mode",
        "train",
        "--robot",
        robot,
        "--config",
        str(snapshot),
        "--output",
        str(output),
        "--num-envs",
        str(num_envs),
        "--iterations",
        str(iterations),
        "--save-every",
        str(save_every),
        "--headless",
        "--skip-evaluation",
        "--console",
        "pretty",
        "--logger",
        args.logger,
    ]
    if arm_path:
        command += ["--arm-config", str(arm_path)]
    if checkpoint:
        command += ["--checkpoint", str(checkpoint)]
    if actor:
        command += ["--initialize-actor", str(actor)]
    gravity = config["gravity_curriculum"]
    full_steps = next(
        (s for s, low, high in gravity["stages"] if low == high == config["gravity"]), None
    )
    full_iteration = full_steps / config["ppo"]["rollout_steps"] if full_steps is not None else None
    return dict(
        task=task.id,
        robot=robot,
        demo=demo,
        source_config=str(source),
        config=config,
        reference=str(reference.resolve()),
        output=str(output),
        command=command,
        iterations=iterations,
        num_envs=num_envs,
        save_every=save_every,
        resume=str(checkpoint) if checkpoint else None,
        initialize_actor=str(actor) if actor else None,
        iteration_mode="additional" if checkpoint else "new",
        gravity_full_iteration=full_iteration if gravity["enabled"] else 0,
        gravity_schedule="checkpoint_unchanged" if checkpoint else "scaled_to_new_run",
        headless=True,
        skip_evaluation=True,
    )
