"""Motion-synchronized tactile and encoder logging on the controller's I/O owner."""

import csv
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from .tactile import FINGERS


class TactileCSV:
    """Preserve raw packets; repeated data are recorded and explicitly marked."""

    columns = (
        'time_s', 'read_started_s', 'read_completed_s', 'finger', 'normal_n',
        'tangential_n', 'direction_deg', 'valid', 'fresh', 'sequence', 'status_error',
        'proximity_raw', 'normal_raw', 'tangential_raw', 'direction_raw',
    )

    def __init__(self, path):
        self.file = Path(path).open('x', newline='')
        self.writer = csv.DictWriter(self.file, fieldnames=self.columns)
        self.writer.writeheader()
        self.previous = None
        self.count = 0
        self.fresh_count = np.zeros(5, int)
        self.valid_count = np.zeros(5, int)
        self.times = []
        self.read_windows = []

    def append(self, sample, started, completed):
        if not np.isfinite([started, completed]).all() or completed < started:
            raise ValueError('Invalid tactile read interval')
        fresh = np.ones(5, bool) if self.previous is None else sample['sequence'] != self.previous
        self.previous = sample['sequence'].copy()
        timestamp = (started + completed) / 2
        for i, finger in enumerate(FINGERS):
            self.writer.writerow(dict(
                time_s=timestamp, read_started_s=started, read_completed_s=completed,
                finger=finger, normal_n=sample['normal_n'][i], tangential_n=sample['shear_n'][i],
                direction_deg=sample['direction_deg'][i], valid=bool(sample['valid'][i]),
                fresh=bool(fresh[i]), sequence=int(sample['sequence'][i]),
                status_error=int(sample['status_error'][i]), proximity_raw=int(sample['proximity_raw'][i]),
                normal_raw=int(sample['raw_registers'][i, 0]),
                tangential_raw=int(sample['raw_registers'][i, 1]),
                direction_raw=int(sample['raw_registers'][i, 2]),
            ))
        self.count += 1
        self.fresh_count += fresh
        self.valid_count += fresh & sample['valid']
        self.times.append(timestamp)
        self.read_windows.append(completed - started)
        self.file.flush()

    def summary(self):
        elapsed = self.times[-1] - self.times[0] if len(self.times) > 1 else 0.
        return dict(
            packets=self.count,
            observed_poll_hz=(self.count - 1) / elapsed if elapsed else None,
            maximum_read_window_s=max(self.read_windows, default=None),
            maximum_sample_gap_s=float(np.diff(self.times).max()) if len(self.times) > 1 else None,
            fresh_samples=dict(zip(FINGERS, self.fresh_count.tolist())),
            fresh_valid_samples=dict(zip(FINGERS, self.valid_count.tolist())),
        )

    def close(self):
        self.file.close()


class MotionTelemetry:
    """Sample only in spare command time; never open another serial connection.

    Requested rate is an upper bound. Reserve a full configured I/O timeout plus
    scheduling margin before the next command; log misses rather than catch up.
    """

    def __init__(self, trajectory, output, requested_hz=100., margin_s=.002):
        if not np.isfinite([requested_hz, margin_s]).all() or requested_hz <= 0 or margin_s <= 0:
            raise ValueError('Positive tactile request rate and scheduling margin required')
        self.trajectory, self.output = trajectory, Path(output)
        self.period, self.margin = 1 / requested_hz, margin_s
        self.writer = self.joints = None
        self.start_time = None
        self.next_poll = 0.
        self.skipped = self.feedback_count = 0
        self.metadata = dict(
            schema='dex_motion_measurement_v1', requested_tactile_hz=requested_hz,
            time_zero='first scheduled command; host monotonic clock shared by commands/feedback/tactile',
            timestamp='tactile request/response midpoint; device sampling latency is unknown',
            scheduling='single I/O owner; tactile only when full timeout fits before next command',
            reserve_margin_s=margin_s, command_hz=1 / trajectory.dt,
            joint_names=list(trajectory.names), finger_order=list(FINGERS),
            units=dict(force='N', angle='degree', joint='rad', time='s'),
            direction='sensor local; 0 deg toward fingertip, clockwise; uncalibrated to USD axes',
            proximity='raw sensor units, not metric distance',
        )

    async def prepare(self, backend):
        from ..execution import sha256

        self.metadata.update(
            hardware=bool(getattr(backend, 'hardware', False)),
            source_sha256=sha256(self.trajectory.path),
            source_recording=str(self.trajectory.path),
            checkpoint_sha256=self.trajectory.metadata.get('checkpoint_sha256'),
            prepared_utc=datetime.now(timezone.utc).isoformat(),
        )
        self.metadata['sensor'] = await backend.prepare_tactile()
        self.timeout = float(self.metadata['sensor']['io_timeout_s'])
        if not 0 < self.timeout + self.margin < self.trajectory.dt:
            raise ValueError('Tactile I/O timeout and margin must fit within one command interval')
        # Validate all five channels before any motion command.
        initial = await backend.read_tactile()
        if not np.asarray(initial['sample']['valid']).all():
            raise ValueError('Tactile preflight reports a sensor fault; no motion sent')
        self.writer = TactileCSV(self.output / 'tactile.csv')
        self.joints = (self.output / 'joint_feedback.csv').open('x', newline='')
        self.joint_writer = csv.writer(self.joints)
        self.joint_writer.writerow([
            'time_s', 'read_completed_s', 'command_index', 'rb3_device_time_s', 'ready',
            *[f'{n}_measured_rad' for n in self.trajectory.names],
            *[f'{n}_target_rad' for n in self.trajectory.names],
            *[f'{n}_error_rad' for n in self.trajectory.names],
        ])

    def start(self, timestamp):
        self.start_time = float(timestamp)
        self.metadata['start_monotonic_s'] = self.start_time

    def feedback(self, state, expected, command_index):
        sample = float(state['sample_time_s']) - self.start_time
        q = np.asarray(state['q_rad'])
        self.joint_writer.writerow([
            sample, sample + float(state.get('read_window_s', 0.)), command_index,
            state.get('rb3_device_time_s', ''), bool(state['ready']), *q, *expected, *(q - expected),
        ])
        self.joints.flush()
        self.feedback_count += 1

    async def poll_until(self, backend, deadline, clock, sleep):
        while True:
            now = clock()
            due = self.start_time + self.next_poll
            if due < now - self.period:
                missed = int((now - due) / self.period)
                self.skipped += missed
                self.next_poll += missed * self.period
                due = self.start_time + self.next_poll
            wake = max(now, due)
            if wake + self.timeout + self.margin > deadline:
                return
            await sleep(max(0., wake - clock()))
            # Oversleep must not consume the controller's next interval.
            if clock() + self.timeout + self.margin > deadline:
                return
            packet = await backend.read_tactile()
            self.writer.append(packet['sample'], packet['started_s'] - self.start_time,
                               packet['completed_s'] - self.start_time)
            self.next_poll += self.period
            if clock() > deadline:
                raise TimeoutError('Tactile read exceeded the available command interval')

    def finish(self, report):
        self.metadata.update(
            completed=bool(report.get('completed')), error=report.get('error'),
            feedback_samples=self.feedback_count, skipped_poll_slots=self.skipped,
            motion_report='report.json', command_log='commands.jsonl',
        )
        try:
            if self.writer is not None:
                self.metadata['tactile'] = self.writer.summary()
                self.metadata['tactile_usable'] = bool(np.all(self.writer.valid_count >= 2))
                self.writer.close()
        finally:
            if self.joints is not None:
                self.joints.close()
            (self.output / 'measurement.json').write_text(json.dumps(self.metadata, indent=2) + '\n')
