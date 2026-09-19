"""Transport-neutral named joint-position trajectory (SI units).

Consumers may send the same targets through an Isaac or future hardware adapter.
This module neither imports a simulator nor opens a robot/network connection.
"""
from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np


@dataclass(frozen=True)
class JointTarget:
    timestamp_s: float
    joint_names: tuple[str, ...]
    position_rad: np.ndarray

    def ordered(self, names):
        """Explicit name mapping; never assume simulator/controller DOF ordering."""
        names = tuple(names)
        if len(names) != len(set(names)) or any(name not in self.joint_names for name in names):
            raise ValueError('Unknown or duplicated command joint name')
        return self.position_rad[[self.joint_names.index(name) for name in names]].copy()


class JointReference:
    def __init__(self, path):
        with np.load(path, allow_pickle=False) as data:
            self.data = {key:data[key].copy() for key in data.files}
        d = self.data
        self.metadata = json.loads(str(d['metadata_json']))
        if self.metadata['schema'] != 'dex_arm_hand_reference_v1':
            raise ValueError('Unsupported joint reference schema')
        if self.metadata['angle_unit'] != 'rad' or self.metadata['length_unit'] != 'm':
            raise ValueError('Reference must use SI units')
        self.names = tuple(str(name) for name in d['joint_names'])
        self.times = d['timestamps_s']
        self.positions = d['joint_position_rad']
        if self.times.ndim != 1 or not len(self.times):
            raise ValueError('Empty or malformed timestamps')
        if len(self.names) != 12 or len(set(self.names)) != 12 or self.positions.shape != (len(self.times),12):
            raise ValueError('Expected six arm plus six independent finger joints')
        if any(d[key].shape != (len(self.times),) for key in ('success','transition_valid')):
            raise ValueError('Missing per-frame validity flags')
        if not np.array_equal(self.positions,np.c_[d['q_arm'],d['q_finger']]):
            raise ValueError('Inconsistent combined joint positions')
        if not d['success'].all() or not d['transition_valid'].all():
            raise ValueError('Refusing to command a failed/discontinuous IK reference')
        if not np.isfinite(self.times).all() or not np.isfinite(self.positions).all() or np.any(np.diff(self.times) <= 0):
            raise ValueError('Nonfinite positions or non-increasing timestamps')
        for key in ('joint_lower_rad','joint_upper_rad','joint_velocity_limits_rad_s'):
            if d[key].shape != (12,) or not np.isfinite(d[key]).all():
                raise ValueError('Invalid joint bounds')
        if np.any(d['joint_velocity_limits_rad_s']<=0) or np.any(d['joint_lower_rad']>=d['joint_upper_rad']):
            raise ValueError('Invalid joint limit intervals')
        if (np.any(self.positions < d['joint_lower_rad'] - 1e-7) or
                np.any(self.positions > d['joint_upper_rad'] + 1e-7)):
            raise ValueError('Joint position limit violation')
        if len(self.times) > 1:
            velocity = np.abs(np.diff(self.positions, axis=0) / np.diff(self.times)[:,None])
            if np.any(velocity > d['joint_velocity_limits_rad_s'] + 1e-6):
                raise ValueError('Joint velocity limit violation')

    def sample(self, timestamp_s):
        t = float(timestamp_s)
        if not np.isfinite(t) or t < self.times[0] - 1e-9 or t > self.times[-1] + 1e-9:
            raise ValueError('Timestamp outside reference; no implicit extrapolation')
        q = np.array([np.interp(t,self.times,self.positions[:,i]) for i in range(12)])
        return JointTarget(t,self.names,q)

    def iter_targets(self, rate_hz=None):
        if rate_hz is None:
            times = self.times
        else:
            if not np.isfinite(rate_hz) or rate_hz <= 0:
                raise ValueError('Positive finite command rate required')
            times = self.times[0] + np.arange(int(np.ceil((self.times[-1]-self.times[0])*rate_hz))) / rate_hz
            times = np.r_[times[times < self.times[-1]], self.times[-1]]
        for t in times:
            yield self.sample(t)

    def export_csv(self, path, rate_hz=None):
        targets = list(self.iter_targets(rate_hz))
        rows = np.array([np.r_[target.timestamp_s,target.position_rad] for target in targets])
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(path,rows,delimiter=',',header=','.join(['timestamp_s', *[n+'_rad' for n in self.names]]),comments='')
