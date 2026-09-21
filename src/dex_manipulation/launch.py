"""Resolve playback inputs without changing training contracts or source data."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from .data import resolve_demo_path


def read(path):
    return json.loads(Path(path).read_text())


def choose(title, options):
    print('\n' + title)
    for index, (_, label) in enumerate(options, 1):
        print(f'  {index}. {label}')
    while True:
        value = input('번호 [1]: ').strip() or '1'
        if value.isdigit() and 1 <= int(value) <= len(options):
            return options[int(value) - 1][0]
        print('표시된 번호를 입력하세요. 종료: Ctrl+C')


def parser():
    p = argparse.ArgumentParser(description='Isaac 재생: ./run.sh [floating|arm] [retarget|policy] [1|2|checkpoint.pt]',
                                epilog='학습: ./train.sh 또는 ./run.sh train --help')
    p.add_argument('robot', nargs='?', choices=('floating', 'arm'))
    p.add_argument('mode', nargs='?', choices=('retarget', 'policy'))
    p.add_argument('selection', nargs='?', help='데모 번호 1/2, 등록 정책 이름 또는 checkpoint 경로')
    p.add_argument('--policy', dest='policy', help='정책 이름 또는 checkpoint 경로 (policy 모드)')
    p.add_argument('--config', type=Path, help='직접 선택한 정책의 학습 설정; 기본은 checkpoint 옆 config.resolved.json')
    p.add_argument('--arm-config', type=Path, help='사용자 정책의 팔 배치/IK 설정')
    p.add_argument('--repeat', type=int, default=0, help='반복 횟수 (기본 0: 무한 반복, 양수: 지정 횟수)')
    p.add_argument('--headless', action='store_true', help='창 없이 동일 물리 재생')
    p.add_argument('--table-safety', choices=('protect','checkpoint'),
                   help='정책 상판 보호: 기존 정책은 protect, 팔 학습 정책은 checkpoint가 기본')
    p.add_argument('--contact-materials', choices=('rubber','checkpoint'),
                   help='정책 재생: 기본 rubber 패드, checkpoint는 학습 당시 접촉 물성 재현')
    p.add_argument('--dry-run', action='store_true', help='실제 실행·파일 생성 없이 연결할 입력과 명령 확인')
    p.add_argument('--list', action='store_true', help='데모와 등록 정책 목록')
    return p


def arm_reference_error(root, path, config_path):
    """Check full IK validity and the actual input/model/placement, before Kit."""
    import numpy as np
    from .reference import JointReference
    from .fk import ArmModel, HandModel
    from .ik import IKOptions, model_fingerprint
    from dataclasses import asdict
    from .scene import Workcell
    try:
        cfg = read(config_path)
        ref = JointReference(path)
        arm = ArmModel.load(root / cfg['arm_model'])
        hand = HandModel.load(root / cfg['hand_model'])
        scene = Workcell.load(root / cfg['workcell'])
        alignment = scene.resolve_alignment(read(root / cfg['alignment']))
        scene.validate_reference(ref.metadata, alignment)
        expected = dict(
            input_sha256=hashlib.sha256(resolve_demo_path(cfg['input'],root).read_bytes()).hexdigest(),
            arm_fingerprint=model_fingerprint(arm), hand_fingerprint=model_fingerprint(hand),
            trajectory_substeps=cfg.get('trajectory_substeps', 1),
            solver=asdict(IKOptions(**cfg['solver'])))
        for key, value in expected.items():
            if ref.metadata.get(key) != value:
                return f'{key} changed'
        if not np.allclose(ref.metadata['base_from_source'], alignment['base_from_source'], atol=1e-10, rtol=0):
            return 'placement changed'
        if not np.array_equal(ref.metadata['initial_seed_q_rad'], cfg['seed_q_rad']):
            return 'initial IK branch seed changed'
        with np.load(resolve_demo_path(cfg['input'],root), allow_pickle=False) as source:
            if ref.metadata['source_frame_ids'] != source['frame_ids'].tolist():
                return 'partial IK result'
        return None
    except (OSError, ValueError, KeyError) as error:
        return str(error)


def resolve_plan(root, args, catalog):
    """Pure input/command resolution: never import Isaac or write any files."""
    def required(path):
        path = resolve_demo_path(Path(path).expanduser(), root)
        if not path.is_file():
            raise ValueError(f'필수 파일이 없습니다: {path}')
        return path

    commands = []
    requested = args.policy or args.selection or '1'
    selected = catalog.get('aliases', {}).get(requested, requested)
    label = Path(selected).stem
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output = root / 'local/results/play' / f'{stamp}_{args.robot}_{args.mode}_{label}'
    details = dict(robot=args.robot, mode=args.mode, selected=selected, output=str(output), training=False,
                   repeat=args.repeat, unlimited=args.repeat == 0)
    flags = ['--headless'] if args.headless else []
    if args.mode == 'policy':
        entry = catalog['policies'].get(selected)
        checkpoint = required(entry['checkpoint'] if entry else selected)
        config_path = required(args.config or checkpoint.parent / 'config.resolved.json')
        cfg = read(config_path)
        native_arm=cfg.get('arm_training',{}).get('enabled',False)
        if native_arm and args.robot!='arm':
            raise ValueError('이 정책은 실제 팔 상태를 관측하며 학습했습니다. arm 환경으로 실행하세요.')
        reference_path = required(cfg['reference'])
        for key in ('model', 'hand_asset', 'object_asset', 'object_geometry'):
            required(cfg[key])
        cmd = [sys.executable, str(root/'scripts/policy.py'), '--mode', 'play', '--robot', args.robot,
               '--checkpoint', str(checkpoint), '--config', str(config_path), '--episodes', str(args.repeat),
               '--evaluation-protocol', 'strict', '--motion-control', 'checkpoint', '--output', str(output), *flags]
        cmd.extend(['--table-safety',args.table_safety or ('checkpoint' if native_arm else 'protect')])
        if getattr(args,'contact_materials',None):
            cmd.extend(['--contact-materials',args.contact_materials])
        if args.robot == 'arm':
            arm_path = args.arm_config or (cfg['arm_training']['arm_config'] if native_arm else (entry['arm_config'] if entry else None))
            if arm_path is None:
                # Fresh training uses a demo's current derived input. Historical
                # checkpoints still resolve through their registered policy below.
                ref = resolve_demo_path(cfg['reference'],root).resolve()
                for demo in catalog['demos'].values():
                    candidates=[demo['config']]
                    if demo.get('training_config'):candidates.append(demo['training_config'])
                    if any(resolve_demo_path(read(root/path)['reference'],root).resolve()==ref for path in candidates):
                        arm_path = demo['arm_config']
                        break
            if arm_path is None:
                # Recognize custom checkpoints that still use a registered demo's
                # exact input; never guess a coordinate transform for new inputs.
                ref = resolve_demo_path(cfg['reference'],root).resolve()
                for candidate in catalog['policies'].values():
                    saved = root / candidate['checkpoint']
                    config = saved.parent / 'config.resolved.json'
                    if config.is_file() and resolve_demo_path(read(config)['reference'],root).resolve() == ref:
                        arm_path = candidate['arm_config']
                        break
            if arm_path is None:
                raise ValueError('이 사용자 정책의 팔 배치를 알 수 없습니다. --arm-config <팔 설정.json>을 지정하세요.')
            arm_path = required(arm_path)
            for key in ('alignment', 'workcell', 'arm_model', 'hand_model', 'usd'):
                required(read(arm_path)[key])
            cmd.extend(['--arm-config', str(arm_path)])
            details['arm_config'] = str(arm_path)
        details.update(checkpoint=str(checkpoint), config=str(config_path), reference=str(reference_path.resolve()),
                       label=entry['label'] if entry else checkpoint.name)
        if entry and entry.get('note'):
            details['note'] = entry['note']
        commands.append(cmd)
    else:
        if selected not in catalog['demos']:
            raise ValueError('리타게팅 데모는 1 또는 2를 선택하세요.')
        entry = catalog['demos'][selected]
        config = required(entry['config'])
        arm_config = required(args.arm_config or entry['arm_config'])
        reference = required(read(arm_config)['input'])
        cmd = [sys.executable, str(root/'scripts/physics.py'), args.robot,
               '--config', str(config), '--arm-config', str(arm_config),
               '--loops', str(args.repeat), '--output', str(output), *flags]
        if args.robot == 'arm':
            cached = root / entry['arm_reference']
            reason = arm_reference_error(root, cached, arm_config)
            if reason:
                # A separate cache avoids overwriting previously validated runs.
                cache = root / 'local/results/play/ik' / selected
                cached = cache / 'trajectory.npz'
                if arm_reference_error(root, cached, arm_config):
                    commands.append([sys.executable, str(root/'scripts/ik.py'), 'solve',
                                     '--config', str(arm_config), '--output', str(cache)])
                    details['prepare_ik'] = reason
            cmd.extend(['--arm-reference', str(cached)])
            details['arm_reference'] = str(cached)
        commands.append(cmd)
        details.update(config=str(config), arm_config=str(arm_config), reference=str(reference), label=entry['label'])
    return dict(**details, commands=commands)


def main(root=None, argv=None):
    root = Path(root or Path(__file__).resolve().parents[2]).resolve()
    p = parser()
    args = p.parse_args(argv)
    catalog = read(root/'config/play.json')
    if args.list:
        for group in ('demos', 'policies'):
            print('데모' if group == 'demos' else '\n정책')
            for name, entry in catalog[group].items():
                print(f"  {name:10} {entry['label']}")
                if 'checkpoint' in entry:
                    print('             ' + entry['checkpoint'])
                if entry.get('note'):
                    print('             ' + entry['note'])
        return 0
    if args.repeat < 0:
        p.error('--repeat는 0(무한 반복) 또는 양수여야 합니다.')
    try:
        if not args.robot or not args.mode:
            if not sys.stdin.isatty():
                p.error('예: ./run.sh floating retarget 1 (전체 도움말: --help)')
            args.robot = args.robot or choose('환경', [('floating', '플로팅 핸드'), ('arm', 'RB3 로봇팔 + Revo2')])
            args.mode = args.mode or choose('동작', [('retarget', '리타게팅만 — 중력·접촉 물리 재생'), ('policy', '학습된 정책 적용')])
            if not args.selection and not args.policy:
                group = 'policies' if args.mode == 'policy' else 'demos'
                options = [(name, entry['label']) for name, entry in catalog[group].items()]
                if args.mode == 'policy':
                    options.append(('custom', '다른 체크포인트 경로 입력'))
                args.selection = choose('정책' if args.mode == 'policy' else '데모', options)
                if args.selection == 'custom':
                    args.selection = input('checkpoint .pt 경로: ').strip()
                    if not args.selection:
                        raise ValueError('체크포인트 경로가 비어 있습니다.')
        aliases = catalog.get('aliases', {})
        if args.policy and args.selection and aliases.get(args.policy, args.policy) != aliases.get(args.selection, args.selection):
            p.error('정책 이름을 서로 다르게 두 번 지정했습니다.')
        if args.mode != 'policy' and (args.policy or args.config):
            p.error('--policy/--config는 policy 모드에서 사용하세요.')
        if args.mode != 'policy' and args.contact_materials:
            p.error('--contact-materials는 policy 모드의 학습 물성 재현 옵션입니다.')
        plan = resolve_plan(root, args, catalog)
        if args.dry_run:
            print(json.dumps(plan, indent=2, ensure_ascii=False))
            return 0
        repetition = '무한 반복' if args.repeat == 0 else f'{args.repeat}회'
        print(f"\n{plan['label']} | {args.robot} | {args.mode} | {repetition}", flush=True)
        print(f"입력: {plan['reference']}\n결과: {plan['output']}\n종료: 창 닫기 또는 Ctrl+C", flush=True)
        if plan.get('note'):
            print('[policy] ' + plan['note'], flush=True)
        output = Path(plan['output'])
        output.mkdir(parents=True, exist_ok=False)
        (output/'launch.json').write_text(json.dumps(plan, indent=2, ensure_ascii=False)+'\n')
        for command in plan['commands']:
            stage = '팔 IK 준비' if Path(command[1]).name == 'ik.py' else 'Isaac Sim 재생 시작'
            print('[run] ' + stage, flush=True)
            code = subprocess.call(command, cwd=root)
            if code:
                print(f'실행 실패 (종료 코드 {code}). 위 오류를 확인하세요.', file=sys.stderr)
                return code
        return 0
    except (OSError, ValueError) as error:
        print(f'실행 준비 실패: {error}', file=sys.stderr)
        return 2
    except (KeyboardInterrupt, EOFError):
        print('\n재생을 종료합니다.')
        return 130
