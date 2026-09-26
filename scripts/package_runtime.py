#!/usr/bin/env python3
"""Publish selected execution inputs; omit raw datasets and intermediate policies."""

import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from dex_manipulation.configuration import read_config
from dex_manipulation.data import resolve_demo_path
from dex_manipulation.execution import RecordedCommands
from dex_manipulation.policies import latest_policy
from dex_manipulation.portable import sha256


def build(root):
    root = Path(root).resolve()
    output = root / 'runtime'
    output.mkdir(exist_ok=True)
    groups, selected = {}, {}
    for demo in ('1', '2'):
        choice = latest_policy(root, demo, 'floating')
        checkpoint = Path(choice['checkpoint'])
        selected[demo] = choice
        groups[f'policy_demo{demo}.zip'] = {checkpoint.parent / name for name in
            ('policy.pt', 'config.resolved.json', 'run_metadata.json', 'training.jsonl')}
        if (checkpoint.parent / 'exit.json').is_file():
            groups[f'policy_demo{demo}.zip'].add(checkpoint.parent / 'exit.json')
        cfg = read_config(checkpoint.parent / 'config.resolved.json')
        reference = resolve_demo_path(cfg['reference'], root)
        if reference.is_relative_to(root / 'local'):
            groups[f'policy_demo{demo}.zip'].add(reference)
    execution = read_config(root / 'config/execution.json')
    commands = root / execution['recording']
    frozen = RecordedCommands(commands)
    recorded_checkpoint = resolve_demo_path(frozen.metadata['checkpoint'], root)
    if recorded_checkpoint not in {Path(p['checkpoint']) for p in selected.values()}:
        groups['policy_recorded.zip'] = {recorded_checkpoint.parent / name for name in
            ('policy.pt', 'config.resolved.json', 'run_metadata.json', 'training.jsonl')}
        reference = resolve_demo_path(frozen.metadata['config']['reference'], root)
        if reference.is_relative_to(root / 'local'):
            groups['policy_recorded.zip'].add(reference)
    hardware = read_config(root / 'config/hardware.example.json')
    groups['execution.zip'] = {commands, root / hardware['simulation_validation']}
    # Optional baseline tactile needed for reproducing the published comparison.
    baseline = root / 'local/reports/hardware_measurement_20260926/sim_frozen/tactile'
    groups['execution.zip'].update(baseline / name for name in ('episode_0001.npz', 'metadata.json'))
    refs = groups['references.zip'] = set()
    catalog = read_config(root / 'config/tasks/can_pick/play.json')
    for demo in catalog['demos'].values():
        path = root / demo['arm_reference']
        if path.is_file():
            refs.add(path)
    for config in (root / 'config/tasks/can_pick').glob('policy*.json'):
        ref = resolve_demo_path(read_config(config)['reference'], root)
        if ref.is_relative_to(root / 'local'):
            refs.add(ref)
    placements = groups['placements.zip'] = set()
    region = root / next(iter(catalog['demos']['2']['random_can_policy_regions'].values()))['1']
    for name in ('manifest.json', 'grid.json', 'collision_model.json'):
        placements.add(region / name)
    scan = read_config(region / 'manifest.json')
    for source, expected in scan['input_hashes'].items():
        path = resolve_demo_path(source, root)
        if sha256(path) != expected:
            raise ValueError(f'Stale placement dependency: {source}')
        if path.is_relative_to(root / 'local'):
            placements.add(path)
    for row in read_config(region / 'grid.json'):
        if (row['initial_can_supported'] and row['ik_success'] and row.get('collision_free')
            and not row['near_singular_samples'] and not row['near_singular_transition_samples']
            and not row['near_limit_samples']):
            placements.add(region / 'points' / f"{row['ix']:02}_{row['iy']:02}.npz")
    data = dict(schema=1, source_roots=[str(root)], policies={
        demo: dict(checkpoint=str(Path(p['checkpoint']).relative_to(root)), iteration=p['iteration'])
        for demo, p in selected.items()}, archives={})
    assigned = set()
    for archive, files in groups.items():
        items = {}
        with zipfile.ZipFile(output / archive, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as packed:
            for path in sorted(files - assigned):
                if not path.is_file() or path.is_symlink() or not path.is_relative_to(root / 'local'):
                    raise ValueError(f'Invalid runtime input: {path}')
                name = str(path.relative_to(root))
                packed.write(path, name)
                items[name] = dict(size=path.stat().st_size, mtime_ns=path.stat().st_mtime_ns, sha256=sha256(path))
        assigned.update(files)
        data['archives'][archive] = dict(size=(output / archive).stat().st_size, sha256=sha256(output / archive), files=items)
    (output / 'manifest.json').write_text(json.dumps(data, indent=2) + '\n')
    print(json.dumps(dict(archives=len(groups), files=len(assigned), compressed_mb=sum(v['size'] for v in data['archives'].values())/1e6, policies=data['policies']), indent=2))


if __name__ == '__main__':
    build(ROOT)
