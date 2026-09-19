"""Offline playback of measured physics link poses, with failure markers."""
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from ..transforms import transform


def write_replay(output, model, reference):
    output = Path(output)
    times = np.linspace(0, reference.duration, round(reference.duration * 30) + 1)
    sampled = reference.sample(times)
    link_names = list(model.link_transforms(model.lower))
    poses = []
    for i, q in enumerate(sampled["q"]):
        wrist = transform(Rotation.from_quat(sampled["wrist_quaternion"][i]).as_matrix(), sampled["wrist_position"][i])
        links = model.link_transforms(q, wrist)
        poses.append([np.r_[links[n][:3, 3], Rotation.from_matrix(links[n][:3, :3]).as_quat()].tolist() for n in link_names])
    human = reference.data["human_keypoints"] @ reference.camera_to_world[:3, :3].T + reference.camera_to_world[:3, 3]
    human_sampled = np.stack([np.interp(times, reference.times, human[:, k, xyz]) for k in range(21) for xyz in range(3)], axis=-1).reshape(-1, 21, 3)
    tracks = [dict(label="기준 궤적 · 사람 21점 + 기하학적 Revo2", times=times, body_names=link_names,
                   links=poses, object=np.c_[sampled["object_position"], sampled["object_quaternion"]],
                   human=human_sampled, status="기하학적 reference; 물리 성공 아님")]
    for stem, label in (("zero_residual", "무보정 · 실제 PhysX"), ("trained_residual", "Residual PPO · 실제 PhysX")):
        with np.load(output / f"{stem}_rollout.npz", allow_pickle=False) as d:
            report = json.loads((output / f"{stem}_evaluation.json").read_text())
            tracks.append(dict(label=label, times=d["time_s"], body_names=d["body_names"], links=d["link_transforms"],
                               object=np.c_[d["object_position"], d["object_quaternion"]],
                               status="실패" if report["failure_reasons"][0] else "추종 기준 통과",
                               success=report["failure_reasons"][0] is None))
    payload = dict(duration=reference.duration, tracks=tracks, semantic_names=model.semantic_names, collision_shapes=reference.collision_shapes,
                   geometry=[dict(link=c["link"], vertices=c["vertices"], faces=c["faces"]) for c in model.colliders],
                   keypoints=model.keypoints)
    def convert(value):
        if isinstance(value, np.ndarray):
            return np.round(value, 7).tolist() if value.dtype.kind == "f" else value.tolist()
        raise TypeError(type(value))
    bundle = (Path(__file__).resolve().parents[1] / "static/policy.bundle.js").read_text()
    content = json.dumps(payload, default=convert, separators=(",", ":"), allow_nan=False).replace("</", "<\\/")
    html = '''<!doctype html><html lang="ko"><meta charset="utf-8"><title>Residual RL — measured PhysX replay</title>
<style>body{margin:20px;background:#101820;color:#e6edf3;font:15px system-ui}h1{font-size:21px}#views{display:flex;gap:12px}.view{width:33.33%;min-width:0}canvas{width:100%;height:480px;border-radius:8px}p{line-height:1.6}.status{min-height:40px;color:#f6c88b}input{width:50%}button,select{padding:8px}#time{margin:12px}a{color:#91c8ed}</style>
<h1>Residual RL · 실제 물리 궤적 비교</h1><p>왼쪽은 기준 동작, 가운데·오른쪽은 측정된 rigid-body pose입니다. 손 표면은 충돌 메시입니다.<br>
실패한 궤적은 기록 종료 시점에서 멈추고 흐리게 표시합니다. 목표 캔 pose로 강제 이동하지 않았습니다. 드래그: 시점 회전 · 휠: 확대.</p>
<button id="play">재생</button> <select id="speed"><option value="0.5">0.5×</option><option value="1" selected>1×</option></select>
<input id="slider" type="range" min="0" step="0.01"><span id="time"></span><div id="views"></div>
<p>초기 캔 +Z를 table up으로 둔 가상 환경입니다. 사람의 실측 테이블/중력 좌표계는 미확인입니다. 원 촬영 fps가 아닌 RL episode 설정의 시간 배정입니다.</p>
<script type="application/json" id="data">''' + content + '</script><script>' + bundle + '</script></html>'
    (output / "comparison.html").write_text(html)
    return output / "comparison.html"
