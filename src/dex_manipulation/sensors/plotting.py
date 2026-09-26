"""Offline, unsmoothed real/sim tactile plots with an explicit shared time origin."""

import csv
import json
from pathlib import Path

import numpy as np

from .tactile import FINGERS

COLORS = ['#386cb0', '#e68a2e', '#3f995b', '#a36cba', '#dd5058']


def simulation(path):
    with np.load(path, allow_pickle=False) as data:
        if list(data['finger_names']) != list(FINGERS):
            raise ValueError('Simulation finger order mismatch')
        times = data['physics_time_s'].copy()
        if not len(times) or not np.isfinite(times).all() or not (np.diff(times) > 0).all():
            raise ValueError('Invalid simulation timestamps')
        valid = data['sensor_pair_coverage_valid'] & ~data['sensor_saturated']
        series = {}
        for i, finger in enumerate(FINGERS):
            series[finger] = (times, np.stack([
                np.where(valid[:, i], data[key][:, i], np.nan)
                for key in ('sensor_normal_n', 'sensor_tangential_n', 'sensor_direction_deg')
            ], axis=1))
        digest = str(data['source_commands_sha256']) if 'source_commands_sha256' in data else None
    return series, digest


def real_data(path, offset_s=0.):
    with Path(path).open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    series, counts = {}, {}
    for finger in FINGERS:
        selected = [r for r in rows if r['finger'] == finger]
        if not selected:
            raise ValueError(f'Missing tactile finger: {finger}')
        t = np.array([float(r['time_s']) - offset_s for r in selected])
        if not np.isfinite(t).all() or not (np.diff(t) > 0).all():
            raise ValueError(f'{finger}: timestamps must increase')
        valid = np.array([r['valid'].lower() in ('true', '1') and r['fresh'].lower() in ('true', '1') for r in selected])
        values = np.array([[float(r[key]) for key in ('normal_n', 'tangential_n', 'direction_deg')] for r in selected])
        values[~valid] = np.nan
        # 65535 is not an angle; do not connect 359-to-0 wrap jumps.
        values[(values[:, 2] < 0) | (values[:, 2] >= 360), 2] = np.nan
        series[finger] = (t, values)
        counts[finger] = dict(total=len(t), fresh_valid=int(valid.sum()))
    return series, counts


def aligned_offset(sim_path, real_path, offset_s=None):
    """Only infer t=0 when both records identify the exact same command file."""
    _, digest = simulation(sim_path)
    metadata_path = Path(real_path).parent / 'measurement.json'
    metadata = json.loads(metadata_path.read_text()) if metadata_path.is_file() else {}
    matched = bool(digest and digest == metadata.get('source_sha256'))
    if digest and metadata.get('source_sha256') and not matched:
        raise ValueError('Real and simulated command trajectory hashes differ')
    if offset_s is None:
        if not matched or metadata.get('start_monotonic_s') is None:
            raise ValueError('Supply --offset-s from a measured start event, or use matching execute recordings')
        offset_s = 0.
    if not np.isfinite(offset_s):
        raise ValueError('Time offset must be finite')
    return float(offset_s), matched


def plot(output, *, sim_path=None, real_path=None, offset_s=None, grasp_start_s=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    if sim_path is None and real_path is None:
        raise ValueError('Provide --real and/or --sim')
    matched = None
    if real_path and sim_path:
        offset_s, matched = aligned_offset(sim_path, real_path, offset_s)
    else:
        offset_s = 0. if offset_s is None else offset_s
    if not np.isfinite(offset_s) or (grasp_start_s is not None and not np.isfinite(grasp_start_s)):
        raise ValueError('Finite time alignment and grasp start required')
    panels, info = [], dict(sim=str(sim_path) if sim_path else None,
                           real=str(real_path) if real_path else None,
                           offset_s=offset_s, matched_commands=matched, smoothing=False)
    if sim_path:
        panels.append(('Simulation (force proxy)', simulation(sim_path)[0]))
    if real_path:
        data, counts = real_data(real_path, offset_s)
        metadata_path = Path(real_path).parent / 'measurement.json'
        metadata = json.loads(metadata_path.read_text()) if metadata_path.is_file() else {}
        label = 'Real robot' if metadata.get('hardware', True) else 'Mock transport (NOT real forces)'
        panels.append((label, data))
        info['real_samples'] = counts
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    fig, axes = plt.subplots(len(panels), 3, figsize=(15, 3.4 * len(panels) + 1.2), squeeze=False, sharex=True)
    titles = ('Normal force', 'Tangential force', 'Tangential direction')
    upper = [0., 0.]
    for row, (label, data) in enumerate(panels):
        for col, title in enumerate(titles):
            ax = axes[row, col]
            for finger, color in zip(FINGERS, COLORS):
                t, values = data[finger]
                if col == 2:
                    ax.scatter(t, values[:, col], s=7, c=color, alpha=.75, linewidths=0)
                else:
                    ax.plot(t, values[:, col], color=color, lw=1.3, marker='.', markersize=2)
                    finite = values[:, col][np.isfinite(values[:, col])]
                    if len(finite):
                        upper[col] = max(upper[col], float(finite.max()))
            if grasp_start_s is not None:
                ax.axvline(grasp_start_s, color='#626a73', ls='--', lw=1)
            ax.set_title(f'{label} | {title}', loc='left', fontsize=11)
            ax.set_ylabel('Angle [deg]' if col == 2 else 'Force [N]')
            ax.grid(alpha=.2)
            if col == 2:
                ax.set_ylim(-8, 368)
                ax.set_yticks([0, 90, 180, 270, 360])
            if row == len(panels) - 1:
                ax.set_xlabel('Time from aligned start [s]')
    for col in (0, 1):
        for ax in axes[:, col]:
            ax.set_ylim(0, max(.1, upper[col] * 1.08))
    fig.suptitle('Revo2 | Fingertip tactile time series', fontsize=16, y=.98)
    fig.legend(handles=[Line2D([0], [0], color=c, lw=2, label=f.title()) for f, c in zip(FINGERS, COLORS)],
               ncol=5, loc='upper center', bbox_to_anchor=(.5, .935), frameon=False)
    note = 'No smoothing or resampling. Invalid/stale readings are gaps; direction dots avoid angle-wrap lines.'
    if sim_path and real_path:
        note += '\n' + ('Same saved commands; aligned scheduled start.' if matched else 'Command identity unverified; explicit time offset applied.')
        note += ' Simulated sensor axes are not hardware calibrated.'
    fig.text(.5, .025, note, ha='center', va='bottom', fontsize=9, color='#505861')
    fig.subplots_adjust(left=.06, right=.985, bottom=.15 if len(panels)==1 else .12, top=.80 if len(panels)==1 else .85, hspace=.35, wspace=.27)
    for suffix in ('png', 'pdf'):
        fig.savefig(output / f'tactile.{suffix}', dpi=180, facecolor='white')
    plt.close(fig)
    (output / 'plot.json').write_text(json.dumps(info, indent=2) + '\n')
    return output / 'tactile.png'
