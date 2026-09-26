"""Read-only Revo2 capacitive tactile collection via the official SDK.

No motor, sensor enable, calibration, reset or current-limit setters are used.
An existing controller should call decode_sdk_sample inside its own bus owner,
not open a second client on the same RS485 device.
"""

import asyncio
import csv
import importlib.metadata
import inspect
import json
from pathlib import Path
import time

import numpy as np

from dex_manipulation.sensors.tactile import FINGERS, decode_capacitive


def decode_sdk_sample(value):
    items = value if isinstance(value, (list, tuple)) else getattr(value, 'items', None)
    if items is None:
        raise ValueError('Expected Revo2 capacitive TouchFingerItem records')
    if len(items) != 5:
        raise ValueError('Revo2 capacitive tactile requires five finger records')
    fields = ('normal_force1', 'tangential_force1', 'tangential_direction1')
    raw = np.array([[int(getattr(item, key)) for key in fields] for item in items])
    status = np.array([int(item.status) for item in items])
    return dict(decode_capacitive(raw, status), raw_registers=raw,
                raw_status=status, proximity_raw=np.array([int(i.self_proximity1) for i in items]))


async def record(port, baudrate_enum, slave_id, output, seconds=10., hz=100., sdk=None):
    if not port or not 1 <= slave_id <= 254 or not np.isfinite([seconds, hz]).all() or seconds <= 0 or hz <= 0:
        raise ValueError('Explicit port/slave ID and positive duration/rate required')
    if sdk is None:
        if importlib.metadata.version('bc-stark-sdk') != '2.0.3':
            raise RuntimeError('Use the project hardware extra: bc-stark-sdk==2.0.3')
        import bc_stark_sdk.main_mod as sdk
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    async def wait(value):
        return await asyncio.wait_for(value, 2.) if inspect.isawaitable(value) else value
    device = await wait(sdk.modbus_open(port, getattr(sdk.Baudrate, baudrate_enum)))
    metadata = dict(schema='revo2_capacitive_hardware_v1', source='real SDK tactile feedback',
                    finger_order=list(FINGERS), requested_poll_hz=hz, duration_s=seconds,
                    port=port, slave_id=slave_id, baudrate_enum=baudrate_enum,
                    units=dict(force='N', angle='degree', force_register='.01 N/count'),
                    timestamp='host monotonic midpoint of request/response; align start to simulation explicitly',
                    readonly=True, proximity='raw uncalibrated units; not distance in meters')
    count = 0
    try:
        if not await wait(device.uses_revo2_motor_api(slave_id)) or not await wait(device.is_touch_hand(slave_id)):
            raise ValueError('Connected hand is not a Revo2 tactile model')
        enabled = int(await wait(device.get_touch_sensor_enabled(slave_id)))
        if enabled & 31 != 31:
            raise ValueError('All five tactile channels must already be enabled; recorder does not modify sensors')
        metadata['firmware_versions'] = str(await wait(device.get_touch_sensor_fw_versions(slave_id)))
        previous = None
        start = time.monotonic()
        columns = ['time_s', 'read_started_s', 'read_completed_s', 'finger', 'normal_n',
                   'tangential_n', 'direction_deg', 'valid', 'fresh', 'sequence',
                   'status_error', 'proximity_raw', 'normal_raw', 'tangential_raw', 'direction_raw']
        with (output/'tactile.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            while time.monotonic()-start < seconds:
                before = time.monotonic()-start
                sample = decode_sdk_sample(await wait(device.get_touch_sensor_status(slave_id)))
                after = time.monotonic()-start
                fresh = np.ones(5, bool) if previous is None else sample['sequence'] != previous
                previous = sample['sequence'].copy()
                for i, name in enumerate(FINGERS):
                    writer.writerow(dict(time_s=(before+after)/2, read_started_s=before, read_completed_s=after,
                        finger=name, normal_n=sample['normal_n'][i], tangential_n=sample['shear_n'][i],
                        direction_deg=sample['direction_deg'][i], valid=bool(sample['valid'][i]),
                        fresh=bool(fresh[i]), sequence=int(sample['sequence'][i]),
                        status_error=int(sample['status_error'][i]), proximity_raw=int(sample['proximity_raw'][i]),
                        normal_raw=int(sample['raw_registers'][i,0]), tangential_raw=int(sample['raw_registers'][i,1]),
                        direction_raw=int(sample['raw_registers'][i,2])))
                f.flush()
                count += 1
                await asyncio.sleep(max(0., start + count/hz - time.monotonic()))
        metadata.update(samples=count, elapsed_s=time.monotonic()-start)
        metadata['actual_poll_hz'] = count/metadata['elapsed_s']
    except BaseException as error:
        metadata.update(error=str(error), samples=count)
        raise
    finally:
        (output/'metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
        await wait(sdk.modbus_close(device))
    return output
