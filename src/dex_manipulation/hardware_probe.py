"""Read raw arm/hand feedback without calibration or opening RB3's command port."""

import asyncio
import importlib.metadata
import inspect
import json
from pathlib import Path
import time

import numpy as np

from .hardware import VERSIONS, rb_status, transport_plan
from .sensors.hardware_tactile import decode_sdk_sample
from .sensors.session import TactileCSV


async def probe(config, output, seconds=5., record_tactile=False):
    transport_plan(config)
    if not np.isfinite(seconds) or seconds <= 0:
        raise ValueError('Probe duration must be positive')
    for name, version in VERSIONS.items():
        if importlib.metadata.version(name) != version:
            raise RuntimeError(f'Use {name}=={version}')
    import rbpodo
    import bc_stark_sdk.main_mod as sdk

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    rb, revo = config['rb3'], config['revo2']
    timeout = config['io_timeout_s']
    device = data = tactile = None
    metadata = dict(schema='dex_hardware_probe_v1', readonly=True, hardware=True,
                    motor_commands=0, requested_hz=10., port=revo['port'], address=rb['address'])
    async def call(method):
        value = getattr(device, method)(revo['slave_id'])
        return await asyncio.wait_for(value, timeout) if inspect.isawaitable(value) else value
    count = 0
    try:
        data = rbpodo.CobotData(rb['address'], rb['data_port'])
        device = await asyncio.wait_for(sdk.modbus_open(revo['port'], getattr(sdk.Baudrate, revo['baudrate_enum'])), 2.)
        metadata.update(hand_type=str(await call('get_hand_type')),
                        position_units=str(await call('get_finger_unit_mode')),
                        uses_revo2_motor_api=bool(await call('uses_revo2_motor_api')))
        if not metadata['uses_revo2_motor_api']:
            raise ValueError('Connected hand does not support Revo2 motor feedback')
        if record_tactile:
            if not await call('is_touch_hand') or int(await call('get_touch_sensor_enabled')) & 31 != 31:
                raise ValueError('Five enabled Revo2 tactile channels required')
            metadata['tactile_firmware'] = str(await call('get_touch_sensor_fw_versions'))
            tactile = TactileCSV(output / 'tactile.csv')
        start = time.monotonic()
        metadata['start_monotonic_s'] = start
        metadata['time_zero'] = 'probe start (NO motion); not a policy start marker'
        with (output / 'raw_feedback.jsonl').open('x') as log:
            while time.monotonic() - start < seconds:
                cycle_started = before = time.monotonic()
                arm, hand = await asyncio.gather(
                    asyncio.to_thread(data.request_data, timeout), call('get_motor_status'))
                after = time.monotonic()
                if arm is None:
                    raise TimeoutError('RB3 feedback timeout')
                state = dict(time_s=(before + after) / 2 - start, read_window_s=after - before,
                             rb3_joint_deg=list(map(float, arm.sdata.jnt_ang)),
                             rb3_device_time_s=float(arm.sdata.time), rb3_status=rb_status(arm.sdata),
                             revo2_positions_raw=list(hand.positions),
                             revo2_motor_states=list(map(str, hand.states)))
                log.write(json.dumps(state) + '\n')
                log.flush()
                if count == 0:
                    print(json.dumps(state, indent=2), flush=True)
                if tactile:
                    before = time.monotonic()
                    sample = decode_sdk_sample(await call('get_touch_sensor_status'))
                    tactile.append(sample, before - start, time.monotonic() - start)
                count += 1
                await asyncio.sleep(max(0., .1 - (time.monotonic() - cycle_started)))
        metadata.update(completed=True, elapsed_s=time.monotonic() - start)
    except BaseException as error:
        metadata.update(completed=False, error=f'{type(error).__name__}: {error}')
        raise
    finally:
        metadata['samples'] = count
        if tactile:
            metadata['tactile'] = tactile.summary()
            tactile.close()
        (output / 'probe.json').write_text(json.dumps(metadata, indent=2) + '\n')
        if device is not None:
            await asyncio.wait_for(sdk.modbus_close(device), timeout)
        data = None
    return metadata
