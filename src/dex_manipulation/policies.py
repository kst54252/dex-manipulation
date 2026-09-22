"""Find completed local training runs without starting Isaac or loading CUDA."""
import json
from pathlib import Path

from .data import resolve_demo_path


def demo_id(root, config):
    explicit = config.get('demo_id')
    if explicit in ('1', '2'):
        return explicit
    reference = resolve_demo_path(config['reference'], root).resolve()
    for demo in ('1', '2'):
        if reference.is_relative_to((Path(root)/f'data/demo{demo}').resolve()):
            return demo
    return None


def _last_stats(path):
    # Only the final JSONL row is needed, even for long training runs.
    with path.open('rb') as stream:
        stream.seek(0, 2)
        stream.seek(max(0, stream.tell()-131072))
        return json.loads(stream.read().splitlines()[-1])


def latest_policy(root, demo, robot, search_root='local/results/policy'):
    """Select newest completed checkpoint compatible with the requested demo.

    Completion requires all iterations in this invocation AND matching saved
    weights. This also handles resumes whose total iteration exceeds the old
    config's training horizon. Periodic checkpoints from partial runs do not win.
    """
    root = Path(root).resolve()
    candidates = []
    for checkpoint in (root/search_root).rglob('policy.pt'):
        try:
            cfg = json.loads((checkpoint.parent/'config.resolved.json').read_text())
            if demo_id(root, cfg) != demo:
                continue
            native_arm = cfg.get('arm_training', {}).get('enabled', False)
            if native_arm and robot != 'arm':
                continue
            metadata = json.loads((checkpoint.parent/'run_metadata.json').read_text())
            if metadata.get('execution', {}).get('mode') != 'train':
                continue
            stats = _last_stats(checkpoint.parent/'training.jsonl')
            if (stats['iterations_in_run'] <= 0 or
                    stats['iteration_in_run'] != stats['iterations_in_run'] or
                    stats['iteration'] != stats['target_iteration']):
                continue
            if not resolve_demo_path(cfg['reference'], root).is_file():
                continue
            candidates.append((checkpoint.stat().st_mtime_ns, str(checkpoint), cfg, metadata, stats))
        except (OSError, ValueError, KeyError, TypeError, IndexError):
            continue
    # Load only plausible completed runs, newest first. weights_only rejects
    # arbitrary pickled code; map_location keeps discovery on the CPU.
    if candidates:
        import torch
        for _, path, cfg, metadata, stats in sorted(candidates, key=lambda row: row[:2], reverse=True):
            try:
                saved = torch.load(path, map_location='cpu', weights_only=True)
                if (saved['schema'] != 3 or saved['iteration'] != stats['target_iteration'] or
                        saved['metadata']['config'] != cfg or
                        saved['metadata']['contract_hash'] != metadata['contract_hash']):
                    continue
            except Exception:
                # An incomplete/corrupt checkpoint must not displace a usable run.
                continue
            return dict(checkpoint=path, iteration=saved['iteration'], demo=demo,
                        trained_robot='arm' if cfg.get('arm_training', {}).get('enabled') else 'floating',
                        selection='latest_completed', search_root=str(root/search_root))
    raise ValueError(f'데모 {demo}: {robot}에서 사용할 학습 완료 정책이 없습니다 ({root/search_root}).')
