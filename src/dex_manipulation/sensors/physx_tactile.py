"""Opt-in 120 Hz pad force recording, independent of reward and control."""

import csv
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from ..materials import PAD_BODIES
from dex_manipulation.sensors.tactile import FINGERS, aggregate_contacts, aggregate_friction, aggregate_separation, capacitive_proxy, pad_axes


def cpu(value):
    return value.detach().cpu().numpy().copy()


class TactileRecorder:
    def __init__(self, env, output):
        if env.num_envs != 1:
            raise ValueError("Tactile recording requires single-environment playback")
        from pxr import PhysxSchema

        self.env, self.output = env, Path(output) / 'tactile'
        self.output.mkdir(parents=True, exist_ok=True)
        self.dt = float(env.cfg['physics_dt'])
        self.ids = [env.robot.body_names.index(n) for n in PAD_BODIES]
        paths = [env.robot._physics_view.link_paths[0][i] for i in self.ids]
        for path in paths:
            PhysxSchema.PhysxContactReportAPI.Apply(env.world.stage.GetPrimAtPath(path)).CreateThresholdAttr(0.)
        filters = [[env.can.prim_paths[0], env.table.prim_paths[0]] for _ in paths]
        self.view = env.world.physics_sim_view.create_rigid_contact_view(paths, filters, 8192)
        if self.view.sensor_count != 5 or self.view.filter_count != 2:
            raise RuntimeError('Expected five pads, each filtered against can and table')
        reported = list(self.view.sensor_paths)
        if len(set(reported)) != 5 or set(reported) != set(paths):
            raise RuntimeError('Tactile pad mapping differs from named bodies')
        self.order = np.array([reported.index(p) for p in paths])
        if list(self.view.filter_paths) != filters:
            raise RuntimeError('Tactile can/table filter order differs')
        self.outward, self.toward_tip = pad_axes(env.model)
        self.rows, self.episode, self.active = [], 0, False
        self.reference_time_override = None
        self.metadata = dict(
            schema='revo2_contact_load_v1', source='PhysX contact impulses / physics_dt',
            finger_order=list(FINGERS), pad_bodies=list(PAD_BODIES),
            filters=['can', 'table'], filter_paths=filters, physics_dt_s=self.dt,
            sample_hz=1/self.dt, max_contact_data_count=8192,
            units=dict(force='N', time='s', position='m'),
            force_frame='world; separate vectors in authored pad-link coordinates, not calibrated sensor axes',
            normal_sum='Sum of compressive point loads; distinct from norm of their vector sum',
            shear='Vector sum of PhysX friction forces for the same pad/filter pair',
            tactile_equivalence='Uncalibrated contact proxy; no taxel layout, transfer function, bandwidth, noise, proximity or hardware axis calibration assumed',
            hardware_sensor_type='capacitive normal/tangential/direction, inferred from user-described output',
            official_protocol='https://staging.brainco.tech/docs/revolimb-hand/en/revo2/modbus_touch.html',
            capacitive_encoding=dict(force_range_n=[0,25], force_count_n=.01, direction_range_deg=[0,359], invalid_direction=65535),
            sensor_axes_basis='USD pad thickness axis and projected distal-origin-to-tip vector; NOT hardware calibrated',
            outward_axes_pad_link=self.outward.tolist(), toward_tip_axes_pad_link=self.toward_tip.tolist(),
            proxy_pair_coverage='can + tabletop; other contacts flagged as incomplete',
            contact_threshold_n=.05,
            contact_separation='minimum reported PhysX contact-point separation; NaN for no points; not full-hand collision clearance',
            penetration_depth='max(0, -separation), geometric overlap without depth clipping',
            pad_rest_offset_m=env.cfg['rest_offset_m'],
            penetration_reporting_threshold_m=.002,
        )
        (self.output/'metadata.json').write_text(json.dumps(self.metadata, indent=2)+'\n')

    def begin(self, episode):
        self.rows, self.episode, self.active = [], episode, True

    def capture(self):
        if not self.active:
            return
        # Each PhysX call may reuse its buffers, so copy before the next read.
        force, points, normal, separation, counts, starts = [cpu(v) for v in self.view.get_contact_data(self.dt)]
        total, vectors = aggregate_contacts(force, normal, counts, starts)
        minimum_separation, penetration = aggregate_separation(separation, counts, starts)
        minimum_separation, penetration = minimum_separation[self.order], penetration[self.order]
        friction, _, fcounts, fstarts = [cpu(v) for v in self.view.get_friction_data(self.dt)]
        shear = aggregate_friction(friction, fcounts, fstarts)
        matrix = cpu(self.view.get_contact_force_matrix(self.dt))
        if not np.allclose(vectors, matrix, atol=2e-4, rtol=2e-4):
            raise RuntimeError('Contact point/vector sum disagrees with PhysX pair force; incomplete data')
        net = cpu(self.view.get_net_contact_forces(self.dt))[self.order]
        total, vectors, shear, counts = [v[self.order] for v in (total, vectors, shear, counts)]
        poses = cpu(self.env.robot._physics_view.get_link_transforms())[0, self.ids]
        rotation = Rotation.from_quat(poses[:, 3:]).as_matrix()
        def local(v):
            return np.einsum('fji,fkj->fki', rotation, v)
        proxy = capacitive_proxy(local(vectors + shear).sum(axis=1), self.outward, self.toward_tip)
        other_normal = net - vectors.sum(axis=1)
        coverage = np.linalg.norm(other_normal, axis=-1) < 1e-4
        self.rows.append(dict(
            physics_time_s=(len(self.rows)+1)*self.dt,
            reference_time_s=(float(self.env.time[0]) if self.reference_time_override is None else self.reference_time_override),
            normal_sum_n=total, normal_world_n=vectors, friction_world_n=shear,
            normal_pad_link_n=local(vectors), friction_pad_link_n=local(shear),
            all_contact_normal_world_n=net, normal_contact_count=counts,
            minimum_contact_separation_m=minimum_separation, penetration_depth_m=penetration,
            penetration_over_2mm=penetration > .002,
            sensor_normal_n=proxy['normal_n'], sensor_tangential_n=proxy['shear_n'],
            sensor_direction_deg=proxy['direction_deg'], sensor_registers=proxy['registers'],
            sensor_saturated=proxy['saturated'], sensor_pair_coverage_valid=coverage,
        ))

    def finish(self, report):
        self.active = False
        if not self.rows:
            return
        data = {k: np.asarray([row[k] for row in self.rows]) for k in self.rows[0]}
        if 'source_commands_sha256' in self.metadata:
            data['source_commands_sha256'] = np.asarray(self.metadata['source_commands_sha256'])
        data['finger_names'] = np.asarray(FINGERS)
        data['filter_names'] = np.asarray(['can', 'table'])
        stem = f'episode_{self.episode:04d}'
        np.savez_compressed(self.output/(stem+'.npz'), **data)
        with (self.output/(stem+'.csv')).open('w', newline='') as f:
            columns = ['physics_time_s', 'reference_time_s']
            for finger in FINGERS:
                for pair in ('can', 'table'):
                    columns += [f'{finger}_{pair}_normal_n', f'{finger}_{pair}_shear_n',
                                f'{finger}_{pair}_separation_m', f'{finger}_{pair}_penetration_m']
            for finger in FINGERS:
                columns += [f'{finger}_sensor_normal_n', f'{finger}_sensor_tangential_n',
                            f'{finger}_sensor_direction_deg', f'{finger}_sensor_pair_coverage_valid']
            writer = csv.writer(f)
            writer.writerow(columns)
            for row in self.rows:
                values = [row['physics_time_s'], row['reference_time_s']]
                for i in range(5):
                    for j in range(2):
                        values += [row['normal_sum_n'][i,j], np.linalg.norm(row['friction_world_n'][i,j]),
                                   row['minimum_contact_separation_m'][i,j], row['penetration_depth_m'][i,j]]
                for i in range(5):
                    values += [row['sensor_normal_n'][i], row['sensor_tangential_n'][i],
                               row['sensor_direction_deg'][i], row['sensor_pair_coverage_valid'][i]]
                writer.writerow(values)
        schedule = self.env.metadata.get('grasp_task', self.env.metadata.get('five_finger_grasp', {})).get('schedule')
        window = data['reference_time_s'] >= schedule['start_time_s']-1e-6 if schedule else np.ones(len(self.rows), bool)
        normal = data['normal_sum_n'][window, :, 0]
        shear = np.linalg.norm(data['friction_world_n'][window, :, 0], axis=-1)
        summary = dict(episode=self.episode, samples=len(self.rows), sample_hz=1/self.dt,
                       window='configured grasp interval' if schedule else 'whole episode',
                       window_start_reference_s=schedule['start_time_s'] if schedule else 0.,
                       window_samples=int(window.sum()), end=report['end'],
                       placement=report.get('placement'), fingers={})
        if len(normal):
            for i, name in enumerate(FINGERS):
                summary['fingers'][name] = dict(
                    normal_mean_n=float(normal[:,i].mean()), normal_peak_n=float(normal[:,i].max()),
                    normal_p95_n=float(np.percentile(normal[:,i],95)),
                    shear_mean_n=float(shear[:,i].mean()), shear_peak_n=float(shear[:,i].max()),
                    contact_fraction=float((normal[:,i]>=.05).mean()),
                    penetration_mean_mm=float(data['penetration_depth_m'][window,i,0].mean()*1000),
                    penetration_peak_mm=float(data['penetration_depth_m'][window,i,0].max()*1000),
                    penetration_over_2mm_fraction=float(data['penetration_over_2mm'][window,i,0].mean()),
                    sensor_normal_mean_n=float(data['sensor_normal_n'][window,i].mean()),
                    sensor_normal_peak_n=float(data['sensor_normal_n'][window,i].max()),
                    sensor_tangential_mean_n=float(data['sensor_tangential_n'][window,i].mean()),
                    sensor_tangential_peak_n=float(data['sensor_tangential_n'][window,i].max()),
                )
            summary.update(normal_sum_mean_n=float(normal.sum(-1).mean()),
                           normal_sum_peak_n=float(normal.sum(-1).max()),
                           all_five_contact_fraction=float((normal>=.05).all(-1).mean()))
        summary['whole_episode_penetration_peak_mm'] = float(data['penetration_depth_m'][:,:,0].max()*1000)
        summary['whole_episode_penetration_over_2mm_samples'] = int(data['penetration_over_2mm'][:,:,0].sum())
        summary['penetration_scope'] = 'Reported pad-can contacts only, all physics steps; 2 mm is a reporting threshold, not a hard constraint'
        summary['sensor_saturation_samples'] = int(data['sensor_saturated'].sum())
        summary['sensor_incomplete_pair_samples'] = int((~data['sensor_pair_coverage_valid']).sum())
        summary['normal_sum_interpretation'] = 'Sum of pad compression loads; not net object force or maximum hand grip rating'
        (self.output/(stem+'.json')).write_text(json.dumps(summary, indent=2)+'\n')
        print('[tactile]', json.dumps(summary), flush=True)
        self.rows.clear()


class TactileWorld:
    def __init__(self, world, recorder):
        self._world, self._recorder = world, recorder

    def __getattr__(self, name):
        return getattr(self._world, name)

    def step(self, *args, **kwargs):
        result = self._world.step(*args, **kwargs)
        self._recorder.capture()
        return result


def attach_tactile_recording(env, output):
    recorder = TactileRecorder(env, output)
    env.world = TactileWorld(env.world, recorder)
    env.tactile_recorder = recorder
    return recorder
