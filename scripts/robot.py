#!/usr/bin/env python3
"""One-command arm/hand control, virtual devices and measured-state RViz."""

import argparse
import fcntl
import socket
from datetime import datetime
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import webbrowser

from dex_manipulation.configuration import read_config

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    parser = argparse.ArgumentParser(prog="./run.sh robot")
    parser.add_argument(
        "mode", choices=("virtual", "sim", "hardware", "vcb", "description", "_panel", "_sim")
    )
    parser.add_argument("--config", type=Path, default=ROOT / "config/robot.json")
    parser.add_argument(
        "--recording",
        type=Path,
        help="Selected immutable recording; defaults to config/execution.json",
    )
    parser.add_argument("--hardware-config", type=Path, default=ROOT / "local/hardware.json")
    parser.add_argument("--vcb-config", type=Path, default=ROOT / "local/vcb.json")
    parser.add_argument(
        "--enable-motion", action="store_true", help="Hardware/VCB only; otherwise read-only"
    )
    parser.add_argument(
        "--allow-jog", action="store_true", help="Explicitly enable bounded hardware/VCB jogging"
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="No RViz window or browser; control panel remains on localhost",
    )
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--no-panel", action="store_true")
    parser.add_argument("--port", type=int)
    parser.add_argument("--seconds", type=float, default=0.0)
    parser.add_argument("--output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    session_lock = None
    children = []
    logs = []

    def launch(name, command, output):
        log = (output / (name + ".log")).open("w")
        logs.append(log)
        process = subprocess.Popen(
            command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        children.append((name, process))
        return process

    def wait_for(predicate, process, log, timeout):
        start = time.monotonic()
        while not predicate():
            if process.poll() is not None:
                raise RuntimeError(
                    f"Process exited: {log}\n{log.read_text(errors='replace')[-3000:]}"
                )
            if time.monotonic() - start > timeout:
                raise TimeoutError(f"Startup timed out; log: {log}")
            time.sleep(0.1)

    try:
        settings = read_config(ROOT / args.config)
        defaults = read_config(ROOT / "config/execution.json")
        recording_path = (ROOT / (args.recording or defaults["recording"])).resolve()
        if args.mode.startswith("_"):
            from dex_manipulation.execution import RecordedCommands

            recording = RecordedCommands(recording_path)
            if args.mode == "_panel":
                from dex_manipulation.robot.panel import run_panel

                run_panel(
                    ROOT,
                    settings,
                    recording,
                    defaults["hold_s"],
                    args.port,
                    ROOT / "src/dex_manipulation/robot/panel.html",
                )
            else:
                from dex_manipulation.robot.sim import run_device

                run_device(ROOT, settings, recording, args.output)
            return 0
        if not math.isfinite(args.seconds) or args.seconds < 0:
            raise ValueError("--seconds must be finite and nonnegative")
        if args.mode in ("virtual", "sim", "description") and (
            args.enable_motion or args.allow_jog
        ):
            raise ValueError(
                "Virtual devices already enable motion; hardware flags are not applicable"
            )
        if args.allow_jog and not args.enable_motion:
            raise ValueError("--allow-jog also requires --enable-motion")
        output = ROOT / "local/robot/sessions" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        output.mkdir(parents=True, exist_ok=False)
        from dex_manipulation.robot.description import export_description

        urdf = export_description(ROOT, settings, output / "description")
        if args.mode == "description":
            print(urdf)
            return 0
        from dex_manipulation.execution import RecordedCommands

        recording = RecordedCommands(recording_path)
        namespace = {
            "virtual": "/dex_virtual",
            "sim": "/dex_sim",
            "hardware": "/dex",
            "vcb": "/dex_vcb",
        }[args.mode]
        backend = "isaac" if args.mode == "sim" else args.mode
        locks = ROOT / "local/robot/locks"
        locks.mkdir(parents=True, exist_ok=True)
        # Hardware lock spans ROS domains: there must be only one device owner.
        identity = (
            args.mode
            if args.mode in ("hardware", "vcb")
            else args.mode + "_" + os.environ.get("ROS_DOMAIN_ID", "0")
        )
        session_lock = (locks / (identity + ".lock")).open("w")
        try:
            fcntl.flock(session_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("This robot session is already running") from error
        settings.update(namespace=namespace, backend=backend)
        port = args.port if args.port is not None else settings["panel_port"]
        if not 1024 <= port <= 65535:
            raise ValueError("Panel port must be in [1024,65535]")
        if not args.no_panel:
            with socket.socket() as probe:
                probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                probe.bind(("127.0.0.1", port))
        if (
            not args.headless
            and not os.environ.get("DISPLAY")
            and not os.environ.get("WAYLAND_DISPLAY")
        ):
            raise ValueError("No desktop display; use --headless")
        from ament_index_python.packages import get_package_prefix

        rsp = (
            Path(get_package_prefix("robot_state_publisher"))
            / "lib/robot_state_publisher/robot_state_publisher"
        )
        rviz = Path(get_package_prefix("rviz2")) / "lib/rviz2/rviz2" if not args.headless else None
        settings_path = output / "settings.json"
        settings_path.write_text(json.dumps(settings, indent=2) + "\n")
        env = dict(os.environ)
        env.setdefault("ROS_AUTOMATIC_DISCOVERY_RANGE", "LOCALHOST")
        env.setdefault("ROS_LOG_DIR", str(ROOT / "local/logs/ros"))
        env["PYTHONPATH"] = str(ROOT / "src") + ":" + str(ROOT) + ":" + env.get("PYTHONPATH", "")
        if args.mode == "hardware":
            hardware_path = (ROOT / args.hardware_config).resolve()
            from dex_manipulation.hardware import connection_plan, hardware_plan

            cfg = read_config(hardware_path)
            cfg["simulation_validation"] = str(ROOT / cfg["simulation_validation"])
            connection_plan(recording.names, cfg)
            if args.enable_motion:
                hardware_plan(recording, cfg)
        elif args.mode == "vcb":
            hardware_path = (ROOT / args.vcb_config).resolve()
            from dex_manipulation.vcb import motion_plan

            motion_plan(recording, read_config(hardware_path))
        else:
            cfg = read_config(ROOT / "config/hardware.example.json")
            # Desktop scheduling is not hardware real-time; keep this override virtual-only.
            cfg["guard"].update(maximum_feedback_age_s=0.2, maximum_lateness_s=0.03)
            hardware_path = output / "virtual_guard.json"
            hardware_path.write_text(json.dumps(cfg, indent=2) + "\n")

        def interrupted(signum, frame):
            raise KeyboardInterrupt

        signal.signal(signal.SIGTERM, interrupted)
        base = [sys.executable, str(ROOT / "scripts/robot.py")]
        if args.mode == "sim":
            process = launch(
                "physics",
                base
                + [
                    "_sim",
                    "--config",
                    str(settings_path),
                    "--recording",
                    str(recording_path),
                    "--output",
                    str(output),
                ],
                output,
            )
            print("[robot] Initializing Isaac physics (headless)…", flush=True)
            wait_for(
                lambda: (output / "ready.json").is_file(), process, output / "physics.log", 120.0
            )
        command = [
            sys.executable,
            str(ROOT / "scripts/ros.py"),
            "bridge",
            "--backend",
            backend,
            "--config",
            str(settings_path),
            "--recording",
            str(recording_path),
            "--hardware-config",
            str(hardware_path),
            "--vcb-config",
            str(hardware_path),
        ]
        if args.enable_motion:
            command.append("--enable-motion")
        if args.allow_jog or args.mode in ("virtual", "sim"):
            command.append("--allow-jog")
        process = launch("bridge", command, output)
        wait_for(
            lambda: "ROS ready:" in (output / "bridge.log").read_text(errors="replace"),
            process,
            output / "bridge.log",
            15.0,
        )
        import yaml

        params = output / "state_publisher.yaml"
        params.write_text(
            yaml.safe_dump(
                {
                    "/**": {
                        "ros__parameters": {
                            "robot_description": urdf.read_text(),
                            "publish_frequency": 60.0,
                        }
                    }
                }
            )
        )
        launch(
            "state_publisher",
            [
                str(rsp),
                "--ros-args",
                "--params-file",
                str(params),
                "-r",
                f"__ns:={namespace}",
                "-r",
                f"joint_states:={namespace}/model_joint_states",
                "-r",
                f"/tf:={namespace}/tf",
                "-r",
                f"/tf_static:={namespace}/tf_static",
            ],
            output,
        )
        if rviz:
            configuration = yaml.safe_load(
                (ROOT / "config/robot.rviz").read_text().replace("__NAMESPACE__", namespace)
            )
            view = output / "robot.rviz"
            view.write_text(yaml.safe_dump(configuration))
            launch(
                "rviz",
                [
                    str(rviz),
                    "-d",
                    str(view),
                    "--ros-args",
                    "-r",
                    f"/tf:={namespace}/tf",
                    "-r",
                    f"/tf_static:={namespace}/tf_static",
                ],
                output,
            )
        if not args.no_panel:
            process = launch(
                "panel",
                base
                + [
                    "_panel",
                    "--config",
                    str(settings_path),
                    "--recording",
                    str(recording_path),
                    "--port",
                    str(port),
                ],
                output,
            )
            wait_for(
                lambda: "Control panel:" in (output / "panel.log").read_text(errors="replace"),
                process,
                output / "panel.log",
                15.0,
            )
            print(f"[robot] Panel: http://127.0.0.1:{port}", flush=True)
            if not args.headless and not args.no_browser:
                webbrowser.open(f"http://127.0.0.1:{port}")
        print(f"[robot] {args.mode} | {namespace} | arm 6 + hand 6 | log: {output}", flush=True)
        print(
            "[robot] No automatic movement. Use the panel or ROS actions; Ctrl+C stops the session.",
            flush=True,
        )
        started = time.monotonic()
        while not args.seconds or time.monotonic() - started < args.seconds:
            for name, process in children:
                if process.poll() is not None:
                    raise RuntimeError(
                        f"{name} exited ({process.returncode}); see {output / (name + '.log')}"
                    )
            time.sleep(0.2)
        return 0
    except KeyboardInterrupt:
        return 0
    except (OSError, ValueError, RuntimeError, ImportError, TimeoutError) as error:
        print(f"Robot session failed: {error}", file=sys.stderr, flush=True)
        return 2
    finally:
        # Reverse order leaves the device alive until the bridge acknowledges stop.
        for name, process in reversed(children):
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGINT)
                try:
                    process.wait(timeout=12)
                except subprocess.TimeoutExpired:
                    print(f"[robot] {name} did not stop; terminating process", file=sys.stderr)
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
        for log in logs:
            log.close()
        if session_lock is not None:
            session_lock.close()


if __name__ == "__main__":
    sys.exit(main())
