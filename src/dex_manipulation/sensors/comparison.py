"""Compare measured real tactile with simulated proxies at an explicit time offset."""

import csv
import json
from pathlib import Path

import numpy as np

from .tactile import FINGERS


def compare(sim_path, real_path, offset_s, output):
    if not np.isfinite(offset_s):
        raise ValueError('A finite measured start-time offset is required')
    with np.load(sim_path, allow_pickle=False) as data:
        sim = {key: data[key].copy() for key in data.files}
    if list(sim['finger_names']) != list(FINGERS):
        raise ValueError('Simulation finger order mismatch')
    with Path(real_path).open(newline='') as f:
        rows = list(csv.DictReader(f))
    times = sim['physics_time_s']
    if len(times) < 2 or not np.isfinite(times).all() or not (np.diff(times) > 0).all():
        raise ValueError('Invalid simulation timestamps')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    result = dict(schema='revo2_tactile_comparison_v1', sim=str(sim_path), real=str(real_path),
                  offset_s=offset_s, alignment='simulation_time = real_host_time - offset_s',
                  interpretation='Real sensor feedback vs uncalibrated USD-axis force proxy; no inferred alignment',
                  fingers={})
    for i, name in enumerate(FINGERS):
        selected = [r for r in rows if r['finger'] == name and r['valid'] == 'True' and r['fresh'] == 'True']
        t = np.array([float(r['time_s'])-offset_s for r in selected])
        keep = np.isfinite(t) & (t >= times[0]) & (t <= times[-1])
        selected = [r for r, good in zip(selected, keep) if good]
        t = t[keep]
        if len(t) < 2 or not (np.diff(t) > 0).all():
            raise ValueError(f'{name}: need at least two fresh valid overlapping samples')
        coverage = np.interp(t, times, sim['sensor_pair_coverage_valid'][:, i].astype(float)) == 1.
        saturation = np.interp(t, times, sim['sensor_saturated'][:, i].astype(float)) > 0.
        good = coverage & ~saturation
        metrics = dict(samples=int(good.sum()), rejected_sim_samples=int((~good).sum()))
        if metrics['samples'] < 2:
            raise ValueError(f'{name}: insufficient unsaturated complete simulation samples')
        for real_key, sim_key in [('normal_n', 'sensor_normal_n'), ('tangential_n', 'sensor_tangential_n')]:
            measured = np.array([float(r[real_key]) for r in selected])[good]
            predicted = np.interp(t, times, sim[sim_key][:, i])[good]
            if not np.isfinite(measured).all():
                raise ValueError('Nonfinite real force value')
            error = predicted - measured
            metrics[real_key] = dict(real_mean=float(measured.mean()), sim_mean=float(predicted.mean()),
                                    bias=float(error.mean()), mae=float(np.abs(error).mean()),
                                    rmse=float(np.sqrt(np.square(error).mean())))
        result['fingers'][name] = metrics
    (output/'comparison.json').write_text(json.dumps(result, indent=2)+'\n')
    return output/'comparison.json'
