"""Resolve playback inputs without changing training contracts or source data."""

from .configuration import read_config
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from .data import resolve_demo_path
from .policies import latest_policy
from .tasks import DEFAULT_TASK, load_task


def read(path):
    return read_config(Path(path))


def choose(title, options):
    print("\n" + title)
    for index, (_, label) in enumerate(options, 1):
        print(f"  {index}. {label}")
    while True:
        value = input("번호 [1]: ").strip() or "1"
        if value.isdigit() and 1 <= int(value) <= len(options):
            return options[int(value) - 1][0]
        print("표시된 번호를 입력하세요. 종료: Ctrl+C")


def parser():
    p = argparse.ArgumentParser(
        description="Isaac 재생: ./run.sh [floating|arm] [retarget|policy] [1|2|checkpoint.pt]",
        epilog="학습: ./train.sh 또는 ./run.sh train --help",
    )
    p.add_argument("robot", nargs="?", choices=("floating", "arm"))
    p.add_argument("--task", default=DEFAULT_TASK, help="작업 이름; 목록: ./run.sh tasks")
    p.add_argument("mode", nargs="?", choices=("retarget", "policy"))
    p.add_argument(
        "selection", nargs="?", help="데모 번호 1/2, 등록 정책 이름 또는 checkpoint 경로"
    )
    p.add_argument("--policy", dest="policy", help="정책 이름 또는 checkpoint 경로 (policy 모드)")
    p.add_argument(
        "--config",
        type=Path,
        help="직접 선택한 정책의 학습 설정; 기본은 checkpoint 옆 config.resolved.json",
    )
    p.add_argument("--arm-config", type=Path, help="사용자 정책의 팔 배치/IK 설정")
    p.add_argument(
        "--repeat", type=int, default=0, help="반복 횟수 (기본 0: 무한 반복, 양수: 지정 횟수)"
    )
    p.add_argument(
        "--speed",
        type=float,
        help="학습/reference 시간 기준 배속 (데모2 정책 기본 1; 나머지는 config/tasks/can_pick/play.json)",
    )
    p.add_argument(
        "--random-can",
        action="store_true",
        help="데모2 arm policy: 매 반복 검증된 IK 격자점에 캔·손 궤적 랜덤 배치",
    )
    p.add_argument(
        "--placement-seed", type=int, help="랜덤 캔 배치 순서를 재현할 seed (--random-can 전용)"
    )
    p.add_argument("--headless", action="store_true", help="창 없이 동일 물리 재생")
    p.add_argument("--record-tactile", action="store_true", help="정책 재생 중 손끝 정상력·접선력 120Hz 기록")
    p.add_argument(
        "--table-safety",
        choices=("protect", "checkpoint"),
        help="정책 상판 보호: 기존 정책은 protect, 팔 학습 정책은 checkpoint가 기본",
    )
    p.add_argument(
        "--contact-materials",
        choices=("rubber", "checkpoint"),
        help="정책 재생: 기본 rubber 패드, checkpoint는 학습 당시 접촉 물성 재현",
    )
    p.add_argument(
        "--dry-run", action="store_true", help="실제 실행·파일 생성 없이 연결할 입력과 명령 확인"
    )
    p.add_argument("--list", action="store_true", help="데모와 등록 정책 목록")
    return p


def arm_reference_error(root, path, config_path):
    """Check full IK validity and the actual input/model/placement, before Kit."""
    import numpy as np
    from .joint_trajectory import JointReference
    from .fk import ArmModel, HandModel
    from .ik import IKOptions, model_fingerprint
    from dataclasses import asdict
    from .scene import Workcell

    try:
        cfg = read(config_path)
        ref = JointReference(path)
        arm = ArmModel.load(root / cfg["arm_model"])
        hand = HandModel.load(root / cfg["hand_model"])
        scene = Workcell.load(root / cfg["workcell"])
        alignment = scene.resolve_alignment(read(root / cfg["alignment"]))
        scene.validate_reference(ref.metadata, alignment)
        expected = dict(
            input_sha256=hashlib.sha256(
                resolve_demo_path(cfg["input"], root).read_bytes()
            ).hexdigest(),
            arm_fingerprint=model_fingerprint(arm),
            hand_fingerprint=model_fingerprint(hand),
            trajectory_substeps=cfg.get("trajectory_substeps", 1),
            solver=asdict(IKOptions(**cfg["solver"])),
        )
        for key, value in expected.items():
            if ref.metadata.get(key) != value:
                return f"{key} changed"
        if not np.allclose(
            ref.metadata["base_from_source"], alignment["base_from_source"], atol=1e-10, rtol=0
        ):
            return "placement changed"
        if not np.array_equal(ref.metadata["initial_seed_q_rad"], cfg["seed_q_rad"]):
            return "initial IK branch seed changed"
        with np.load(resolve_demo_path(cfg["input"], root), allow_pickle=False) as source:
            if ref.metadata["source_frame_ids"] != source["frame_ids"].tolist():
                return "partial IK result"
        return None
    except (OSError, ValueError, KeyError) as error:
        return str(error)


def resolve_plan(root, args, catalog):
    task = load_task(root, getattr(args, "task", DEFAULT_TASK))
    return task.entrypoint("playback_plan")(root, args, catalog)


def main(root=None, argv=None):
    root = Path(root or Path(__file__).resolve().parents[2]).resolve()
    p = parser()
    args = p.parse_args(argv)
    try:
        task = load_task(root, args.task)
        catalog = task.catalog()
    except (OSError, ValueError) as error:
        p.error(str(error))
    if args.list:
        for group in ("demos", "policies"):
            print("데모" if group == "demos" else "\n정책")
            for name, entry in catalog[group].items():
                print(f"  {name:10} {entry['label']}")
                if "checkpoint" in entry:
                    print("             " + entry["checkpoint"])
                if "latest_demo" in entry:
                    try:
                        latest = latest_policy(
                            root, entry["latest_demo"], args.robot or "floating", task=task.id
                        )
                        print(f"             {latest['checkpoint']} ({latest['iteration']}iter)")
                    except ValueError as error:
                        print("             " + str(error))
                if entry.get("note"):
                    print("             " + entry["note"])
        return 0
    if args.repeat < 0:
        p.error("--repeat는 0(무한 반복) 또는 양수여야 합니다.")
    try:
        task.entrypoint("playback_plan")
        if not args.robot or not args.mode:
            if not sys.stdin.isatty():
                p.error("예: ./run.sh floating retarget 1 (전체 도움말: --help)")
            args.robot = args.robot or choose(
                "환경", [("floating", "플로팅 핸드"), ("arm", "RB3 로봇팔 + Revo2")]
            )
            args.mode = args.mode or choose(
                "동작",
                [("retarget", "리타게팅만 — 중력·접촉 물리 재생"), ("policy", "학습된 정책 적용")],
            )
            if not args.selection and not args.policy:
                group = "policies" if args.mode == "policy" else "demos"
                options = [(name, entry["label"]) for name, entry in catalog[group].items()]
                if args.mode == "policy":
                    options.append(("custom", "다른 체크포인트 경로 입력"))
                args.selection = choose("정책" if args.mode == "policy" else "데모", options)
                if args.selection == "custom":
                    args.selection = input("checkpoint .pt 경로: ").strip()
                    if not args.selection:
                        raise ValueError("체크포인트 경로가 비어 있습니다.")
        aliases = catalog.get("aliases", {})
        if (
            args.policy
            and args.selection
            and aliases.get(args.policy, args.policy) != aliases.get(args.selection, args.selection)
        ):
            p.error("정책 이름을 서로 다르게 두 번 지정했습니다.")
        if args.mode != "policy" and (args.policy or args.config):
            p.error("--policy/--config는 policy 모드에서 사용하세요.")
        if args.mode != "policy" and args.contact_materials:
            p.error("--contact-materials는 policy 모드의 학습 물성 재현 옵션입니다.")
        plan = resolve_plan(root, args, catalog)
        if args.dry_run:
            print(json.dumps(plan, indent=2, ensure_ascii=False))
            return 0
        repetition = "무한 반복" if args.repeat == 0 else f"{args.repeat}회"
        print(
            f"\n{plan['label']} | {args.robot} | {args.mode} | {plan['speed']:g}배속 | {repetition}",
            flush=True,
        )
        print(
            f"입력: {plan['reference']}\n결과: {plan['output']}\n종료: 창 닫기 또는 Ctrl+C",
            flush=True,
        )
        if plan.get("checkpoint"):
            print(f"정책: {plan['checkpoint']}", flush=True)
        if plan.get("note"):
            print("[policy] " + plan["note"], flush=True)
        output = Path(plan["output"])
        output.mkdir(parents=True, exist_ok=False)
        (output / "launch.json").write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n")
        for command in plan["commands"]:
            stage = "팔 IK 준비" if Path(command[1]).name == "ik.py" else "Isaac Sim 재생 시작"
            print("[run] " + stage, flush=True)
            code = subprocess.call(command, cwd=root)
            if code:
                print(f"실행 실패 (종료 코드 {code}). 위 오류를 확인하세요.", file=sys.stderr)
                return code
        return 0
    except (OSError, ValueError) as error:
        print(f"실행 준비 실패: {error}", file=sys.stderr)
        return 2
    except (KeyboardInterrupt, EOFError):
        print("\n재생을 종료합니다.")
        return 130
