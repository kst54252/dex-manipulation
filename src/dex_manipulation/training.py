"""Prepare isolated, headless training runs without changing source configs."""
import argparse
from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path
import shlex
import subprocess
import sys

from .data import resolve_demo_path
from .launch import choose, read


def parser():
    p = argparse.ArgumentParser(prog='./train.sh', description='학습: ./train.sh [floating|arm] [1|2] 또는 ./run.sh train ...')
    p.add_argument('robot', nargs='?', choices=('floating', 'arm'))
    p.add_argument('demo', nargs='?', choices=('1', '2'))
    p.add_argument('-i', '--iterations', type=int, help='실행할 iteration 수; 이어 학습은 추가 횟수')
    p.add_argument('-n', '--num-envs', type=int, help='병렬 환경 수 (새 학습 기본 4096)')
    p.add_argument('--save-every', type=int, help='checkpoint 저장 주기 (기본 100iter, 마지막은 항상 저장)')
    start = p.add_mutually_exclusive_group()
    start.add_argument('--resume', type=Path, help='checkpoint와 옆의 config.resolved.json으로 이어 학습')
    start.add_argument('--initialize-actor', type=Path, help='새 팔 학습에 호환되는 floating actor만 이관 (resume 아님)')
    p.add_argument('--output', type=Path, help='새 결과 폴더 (local/ 내부, 기본 자동 생성)')
    p.add_argument('--logger', choices=('tensorboard', 'none'), default='tensorboard')
    p.add_argument('--dry-run', action='store_true', help='Isaac 시작·파일 생성 없이 설정과 명령 확인')
    return p


def scale_gravity(config, iterations):
    """Keep the existing stage values/relative timing for a new training horizon."""
    original_iterations = config['training']['iterations']
    curriculum = config['gravity_curriculum']
    if not curriculum['enabled'] or iterations == original_iterations:
        return
    ratio = iterations / original_iterations
    stages = []
    for step, low, high in curriculum['stages']:
        scaled = [int(round(step * ratio)), low, high]
        # For very short smoke runs, retain the latest bounds at a shared tick.
        if stages and stages[-1][0] == scaled[0]:
            stages[-1] = scaled
        else:
            stages.append(scaled)
    curriculum['stages'] = stages
    curriculum['basis'] = (f'Launcher scaled control-step thresholds by {ratio:.12g} '
                           f'for {iterations} new iterations; original gravity bounds retained.')


def resolve_plan(root, args):
    """Read inputs only: no Isaac import, checkpoint mutation, or file creation."""
    root = Path(root).resolve()

    def required(path):
        path = resolve_demo_path(Path(path).expanduser(), root)
        if not path.is_file():
            raise ValueError(f'필수 파일이 없습니다: {path}')
        return path

    for name in ('iterations', 'num_envs', 'save_every'):
        value = getattr(args, name)
        if value is not None and value < 1:
            raise ValueError(f'--{name.replace("_", "-")}는 양수여야 합니다.')
    checkpoint = required(args.resume) if args.resume else None
    actor = required(args.initialize_actor) if args.initialize_actor else None
    if checkpoint and actor:
        raise ValueError('--resume와 --initialize-actor를 함께 사용할 수 없습니다.')
    if checkpoint:
        source = required(checkpoint.parent / 'config.resolved.json')
        config = read(source)
        robot = 'arm' if config.get('arm_training', {}).get('enabled', False) else 'floating'
        reference = required(config['reference']).resolve()
        demo = next((n for n in ('1', '2') if reference.is_relative_to(root / f'data/demo{n}')), None)
        if demo is None:
            raise ValueError('checkpoint 입력이 demo1/demo2에 속하지 않습니다. scripts/policy.py로 직접 설정하세요.')
        if (args.robot and args.robot != robot) or (args.demo and args.demo != demo):
            raise ValueError(f'checkpoint는 {robot}, {demo}번 데모입니다. 다른 환경/데모로 이어 학습할 수 없습니다.')
        saved_metadata = checkpoint.parent / 'run_metadata.json'
        saved_envs = (read(saved_metadata).get('physics', {}).get('num_envs')
                      if saved_metadata.is_file() else None) or config['training']['num_envs']
        if args.num_envs is not None and args.num_envs != saved_envs:
            raise ValueError(f'이어 학습은 저장된 환경 수 {saved_envs}개를 유지해야 합니다 (환경별 물리 상태 복원).')
        num_envs = saved_envs
    else:
        if not args.robot or not args.demo:
            raise ValueError('환경과 데모를 지정하세요. 예: ./train.sh floating 2 --iterations 2000')
        robot, demo = args.robot, args.demo
        catalog = read(root / 'config/play.json')['demos'][demo]
        source = required(catalog['config'] if robot == 'floating' else 'config/policy_arm.json')
        config = deepcopy(read(source))
        if robot == 'arm':
            # The arm template owns physical/controller/PPO settings; only the
            # selected demo's reference, coordinate description and placement vary.
            demo_config = read(required(catalog['config']))
            config['reference'] = demo_config['reference']
            config['scene_assumption'] = demo_config['scene_assumption']
            config['arm_training']['arm_config'] = catalog['arm_config']
            if demo_config['reference'] != read(source)['reference']:
                config['recipe'] = f'revo2_rb3_online_ik_demo{demo}_v1'
        num_envs = args.num_envs or config['training']['num_envs']
    if actor and robot != 'arm':
        raise ValueError('--initialize-actor는 새 arm 학습에서만 사용하세요.')
    iterations = args.iterations or config['training']['iterations']
    save_every = args.save_every or config['training']['save_every']
    if not checkpoint:
        scale_gravity(config, iterations)
        config['training'].update(iterations=iterations, num_envs=num_envs, save_every=save_every)
    # Resumes pass execution counts as CLI args only. Editing the snapshot would
    # invalidate the strict checkpoint contract and restart curriculum timing.
    reference = required(config['reference'])
    for key in ('model', 'hand_asset', 'object_asset', 'object_geometry'):
        required(config[key])
    arm_path = None
    if robot == 'arm':
        arm_path = required(config['arm_training']['arm_config'])
        for key in ('alignment', 'workcell', 'arm_model', 'hand_model', 'usd'):
            required(read(arm_path)[key])
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output = args.output or Path(f'local/results/policy/demo{demo}_{robot}_{iterations}_{stamp}')
    output = Path(output).expanduser()
    output = (root / output).resolve()
    if not output.is_relative_to(root / 'local'):
        raise ValueError('학습 결과는 Git 제외 영역인 local/ 아래에 저장하세요.')
    if output.exists():
        raise ValueError(f'결과를 덮어쓸 수 없습니다. 새 --output 폴더를 지정하세요: {output}')
    snapshot = output / 'config.input.json'
    command = [sys.executable, '-u', str(root / 'scripts/policy.py'), '--mode', 'train',
               '--robot', robot, '--config', str(snapshot), '--output', str(output),
               '--num-envs', str(num_envs), '--iterations', str(iterations), '--save-every', str(save_every),
               '--headless', '--skip-evaluation', '--console', 'pretty', '--logger', args.logger]
    if arm_path:
        command += ['--arm-config', str(arm_path)]
    if checkpoint:
        command += ['--checkpoint', str(checkpoint)]
    if actor:
        command += ['--initialize-actor', str(actor)]
    gravity = config['gravity_curriculum']
    full_steps = next((s for s, low, high in gravity['stages'] if low == high == config['gravity']), None)
    full_iteration = full_steps / config['ppo']['rollout_steps'] if full_steps is not None else None
    return dict(robot=robot, demo=demo, source_config=str(source), config=config,
                reference=str(reference.resolve()), output=str(output), command=command,
                iterations=iterations, num_envs=num_envs, save_every=save_every,
                resume=str(checkpoint) if checkpoint else None, initialize_actor=str(actor) if actor else None,
                iteration_mode='additional' if checkpoint else 'new',
                gravity_full_iteration=full_iteration if gravity['enabled'] else 0,
                gravity_schedule='checkpoint_unchanged' if checkpoint else 'scaled_to_new_run',
                headless=True, skip_evaluation=True)


def main(root=None, argv=None):
    root = Path(root or Path(__file__).resolve().parents[2]).resolve()
    p = parser()
    args = p.parse_args(argv)
    try:
        if not args.resume and (not args.robot or not args.demo):
            if not sys.stdin.isatty():
                p.error('예: ./train.sh floating 2 --iterations 2000 (메뉴는 터미널에서 실행)')
            action = 'new' if args.initialize_actor else choose('학습 방식', [('new', '새 학습'), ('resume', '이어 학습')])
            if action == 'resume':
                value = input('checkpoint .pt 경로: ').strip()
                if not value:
                    raise ValueError('checkpoint 경로가 비어 있습니다.')
                args.resume = Path(value)
            else:
                args.robot = args.robot or choose('환경', [('floating', '플로팅 핸드'), ('arm', 'RB3 + Revo2 · 온라인 IK')])
                demos = read(root / 'config/play.json')['demos']
                args.demo = args.demo or choose('데모', [(n, item['label']) for n, item in demos.items()])
            if args.iterations is None:
                default = resolve_plan(root, args)['iterations']
                args.iterations = int(input(f'학습 횟수 (이어 학습은 추가 횟수) [{default}]: ').strip() or default)
        plan = resolve_plan(root, args)
        if args.dry_run:
            print(json.dumps(plan, indent=2, ensure_ascii=False))
            return 0
        qualifier = '추가 ' if plan['resume'] else ''
        print(f"\n데모 {plan['demo']} | {plan['robot']} | {plan['num_envs']}환경 | {qualifier}{plan['iterations']}iter", flush=True)
        print(f"입력: {plan['reference']}\n결과: {plan['output']}", flush=True)
        if plan['resume']:
            print('이어 학습: checkpoint의 optimizer·중력 진행 상태·설정을 복원합니다.', flush=True)
        else:
            print(f"전체 중력 도달: {plan['gravity_full_iteration']:g}iter 경계" if plan['gravity_full_iteration'] is not None
                  else '전체 중력 도달 단계가 없는 설정입니다.', flush=True)
        print('창 없이 학습만 실행합니다. 종료 후 평가·재생은 실행하지 않습니다.', flush=True)
        output = Path(plan['output'])
        output.mkdir(parents=True, exist_ok=False)
        (output / 'config.input.json').write_text(json.dumps(plan['config'], indent=2) + '\n')
        (output / 'launch.json').write_text(json.dumps(plan, indent=2, ensure_ascii=False) + '\n')
        print('[train] ' + shlex.join(plan['command']), flush=True)
        code = subprocess.call(plan['command'], cwd=root)
        (output / 'exit.json').write_text(json.dumps(dict(exit_code=code)) + '\n')
        if code:
            print(f'학습 실행 실패 (종료 코드 {code}). 위 오류를 확인하세요.', file=sys.stderr)
        else:
            print(f"학습 완료: {output / 'policy.pt'}", flush=True)
        return code
    except (OSError, ValueError, KeyError) as error:
        print(f'학습 준비 실패: {error}', file=sys.stderr)
        return 2
    except (KeyboardInterrupt, EOFError):
        print('\n학습 실행을 중단했습니다. 이어 학습에는 마지막으로 저장된 checkpoint를 사용하세요.')
        return 130
