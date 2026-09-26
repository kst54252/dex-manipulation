"""Prepare isolated, headless training runs without changing source configs."""

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

from .data import resolve_demo_path
from .launch import choose, read
from .tasks import config_task_id, load_task


def parser():
    p = argparse.ArgumentParser(
        prog="./train.sh",
        description="학습: ./train.sh [floating|arm] [데모 번호] 또는 ./run.sh train ... (can_pick: 2)",
    )
    p.add_argument("robot", nargs="?", choices=("floating", "arm"))
    p.add_argument("demo", nargs="?", help="선택한 task의 데모 번호")
    p.add_argument("--task", help="작업 이름; 기본 can_pick, 이어 학습은 checkpoint에서 읽음")
    p.add_argument(
        "-i", "--iterations", type=int, help="실행할 iteration 수; 이어 학습은 추가 횟수"
    )
    p.add_argument("-n", "--num-envs", type=int, help="병렬 환경 수 (새 학습 기본 4096)")
    p.add_argument(
        "--save-every", type=int, help="checkpoint 저장 주기 (기본 100iter, 마지막은 항상 저장)"
    )
    p.add_argument("--config", type=Path, help="새 학습의 설정 선택; 이어 학습은 저장된 설정 사용")
    start = p.add_mutually_exclusive_group()
    start.add_argument(
        "--resume", type=Path, help="checkpoint와 옆의 config.resolved.json으로 이어 학습"
    )
    start.add_argument(
        "--initialize-actor",
        type=Path,
        help="새 floating/arm 학습에 호환되는 floating actor 이관 (resume 아님)",
    )
    p.add_argument("--output", type=Path, help="새 결과 폴더 (local/ 내부, 기본 자동 생성)")
    p.add_argument("--logger", choices=("tensorboard", "none"), default="tensorboard")
    p.add_argument(
        "--dry-run", action="store_true", help="Isaac 시작·파일 생성 없이 설정과 명령 확인"
    )
    return p


def scale_gravity(config, iterations):
    """Keep the existing stage values/relative timing for a new training horizon."""
    original_iterations = config["training"]["iterations"]
    curriculum = config["gravity_curriculum"]
    if not curriculum["enabled"] or iterations == original_iterations:
        return
    ratio = iterations / original_iterations
    stages = []
    for step, low, high in curriculum["stages"]:
        scaled = [int(round(step * ratio)), low, high]
        # For very short smoke runs, retain the latest bounds at a shared tick.
        if stages and stages[-1][0] == scaled[0]:
            stages[-1] = scaled
        else:
            stages.append(scaled)
    curriculum["stages"] = stages
    curriculum["basis"] = (
        f"Launcher scaled control-step thresholds by {ratio:.12g} "
        f"for {iterations} new iterations; original gravity bounds retained."
    )


def resolve_plan(root, args):
    task_id = getattr(args, "task", None)
    if args.resume:
        saved = (
            resolve_demo_path(Path(args.resume).expanduser(), root).parent / "config.resolved.json"
        )
        checkpoint_task = config_task_id(read(saved))
        if task_id is not None and task_id != checkpoint_task:
            raise ValueError(f"checkpoint task는 {checkpoint_task}입니다.")
        task_id = checkpoint_task
    task = load_task(root, task_id)
    resolved = argparse.Namespace(**vars(args))
    resolved.task = task.id
    return task.entrypoint("training_plan")(root, resolved)


def main(root=None, argv=None):
    root = Path(root or Path(__file__).resolve().parents[2]).resolve()
    p = parser()
    args = p.parse_args(argv)
    try:
        if not args.resume:
            load_task(root, args.task).entrypoint("training_plan")
        if not args.resume and (not args.robot or not args.demo):
            if not sys.stdin.isatty():
                p.error("예: ./train.sh floating 2 --iterations 2000 (메뉴는 터미널에서 실행)")
            action = (
                "new"
                if args.initialize_actor
                else choose("학습 방식", [("new", "새 학습"), ("resume", "이어 학습")])
            )
            if action == "resume":
                value = input("checkpoint .pt 경로: ").strip()
                if not value:
                    raise ValueError("checkpoint 경로가 비어 있습니다.")
                args.resume = Path(value)
            else:
                args.robot = args.robot or choose(
                    "환경", [("floating", "플로팅 핸드"), ("arm", "RB3 + Revo2 · 온라인 IK")]
                )
                demos = load_task(root, args.task).catalog()["demos"]
                args.demo = args.demo or choose(
                    "데모", [(n, item["label"]) for n, item in demos.items()]
                )
            if args.iterations is None:
                default = resolve_plan(root, args)["iterations"]
                args.iterations = int(
                    input(f"학습 횟수 (이어 학습은 추가 횟수) [{default}]: ").strip() or default
                )
        plan = resolve_plan(root, args)
        if args.dry_run:
            print(json.dumps(plan, indent=2, ensure_ascii=False))
            return 0
        qualifier = "추가 " if plan["resume"] else ""
        print(
            f"\n데모 {plan['demo']} | {plan['robot']} | {plan['num_envs']}환경 | {qualifier}{plan['iterations']}iter",
            flush=True,
        )
        print(f"입력: {plan['reference']}\n결과: {plan['output']}", flush=True)
        if plan["resume"]:
            print("이어 학습: checkpoint의 optimizer·중력 진행 상태·설정을 복원합니다.", flush=True)
        else:
            print(
                f"전체 중력 도달: {plan['gravity_full_iteration']:g}iter 경계"
                if plan["gravity_full_iteration"] is not None
                else "전체 중력 도달 단계가 없는 설정입니다.",
                flush=True,
            )
        print("창 없이 학습만 실행합니다. 종료 후 평가·재생은 실행하지 않습니다.", flush=True)
        output = Path(plan["output"])
        output.mkdir(parents=True, exist_ok=False)
        (output / "config.input.json").write_text(json.dumps(plan["config"], indent=2) + "\n")
        (output / "launch.json").write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n")
        print("[train] " + shlex.join(plan["command"]), flush=True)
        code = subprocess.call(plan["command"], cwd=root)
        (output / "exit.json").write_text(json.dumps(dict(exit_code=code)) + "\n")
        if code:
            print(f"학습 실행 실패 (종료 코드 {code}). 위 오류를 확인하세요.", file=sys.stderr)
        else:
            print(f"학습 완료: {output / 'policy.pt'}", flush=True)
        return code
    except (OSError, ValueError, KeyError) as error:
        print(f"학습 준비 실패: {error}", file=sys.stderr)
        return 2
    except (KeyboardInterrupt, EOFError):
        print(
            "\n학습 실행을 중단했습니다. 이어 학습에는 마지막으로 저장된 checkpoint를 사용하세요."
        )
        return 130
