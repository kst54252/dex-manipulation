"""Runtime playback timing, separate from saved training/checkpoint contracts."""
import copy


def prepare_playback(reference, config, speed=1.0):
    dt = config['physics_dt'] * config['control_decimation']
    applied = reference.for_playback(speed, dt)
    runtime = copy.deepcopy(config)
    if speed != 1.0:
        runtime['episode_length_s'] /= speed
        # These intervals belong to reference phase. Physical controller gains,
        # filters, contact properties and the simulation/control dt stay fixed.
        for key in ('blend_start_s', 'blend_end_s'):
            if key in runtime['augmentation']:
                runtime['augmentation'][key] /= speed
        for key in ('approach_lead_s', 'preload_ramp_s'):
            if key in runtime.get('grasp_task', {}):
                runtime['grasp_task'][key] /= speed
    timing = dict(speed=float(speed), input_duration_s=reference.duration,
                  duration_s=applied.duration, physics_dt_s=config['physics_dt'],
                  control_dt_s=dt, differs_from_training=speed != 1.0,
                  restore_training_rsi_histogram=speed == 1.0,
                  interpretation='reference clock retimed; physical grasp performance requires separate validation')
    return applied, runtime, timing
