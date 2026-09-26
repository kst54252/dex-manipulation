"""SPIDER-inspired sampling optimization with removable physical contact forces.

Independent PhysX adaptation, not the paper's full implementation. No policy
network, RL updates, object pose pins or post-reset teleports are used.
"""

import json
from pathlib import Path

import numpy as np
import torch

from .contact_retargeting import PAD_LINKS, HardContacts, extract_contacts, adapt_contact_anchors
from .fk import HandModel
from dex_manipulation.geometry import CollisionScene
from .policy.math3d import quat_apply, rotation_error
from dex_manipulation.policy.trajectory import ReferenceMotion, digest
from .data import resolve_demo_path
from .configuration import read_config


def spring_pair_force(error, relative_velocity, stiffness, damping, cap, strength):
    """Force on hand; object receives its exact negative (world-frame SI)."""
    direction = error / error.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    speed = (relative_velocity * direction).sum(dim=-1, keepdim=True)
    # Central pair damping conserves total angular momentum as well as force.
    force = -stiffness * error - damping * speed * direction
    force = force * (cap / force.norm(dim=-1, keepdim=True).clamp_min(1e-12)).clamp(max=1)
    return force * strength


def sampling_update(controls, candidates, costs, temperature):
    """Boltzmann weighted trajectory perturbations, stabilized by min cost."""
    if temperature <= 0 or not torch.isfinite(costs).all():
        raise ValueError("Finite costs and positive temperature required")
    weights = torch.softmax(-(costs - costs.min()) / temperature, dim=0)
    return (controls + torch.einsum("n,nha->ha", weights, candidates - controls)).clamp(-1, 1)


class VirtualContacts:
    def __init__(self, env, plan, pad_points, config):
        self.cfg, self.plan = config, plan
        self.ids = torch.tensor(
            [env.robot.body_names.index(n) for n in PAD_LINKS], device=env.device
        )
        self.pad = torch.tensor(pad_points, dtype=torch.float32, device=env.device)
        self.object = torch.tensor(plan.anchors, dtype=torch.float32, device=env.device)
        self.times = torch.tensor(plan.times, dtype=torch.float32, device=env.device)
        self.active = torch.tensor(plan.active, device=env.device)
        self.strength = 0.0
        self.maximum_force = torch.zeros(env.num_envs, device=env.device)
        self.impulse = torch.zeros(env.num_envs, device=env.device)

    def points(self, state):
        links = state["link_transforms"][:, self.ids]
        hand_offset = quat_apply(links[..., 3:7], self.pad[None].expand(len(links), -1, -1))
        hand = links[..., :3] + hand_offset
        obj_offset = quat_apply(
            state["object_quaternion"][:, None].expand(-1, 5, -1),
            self.object[None].expand(len(links), -1, -1),
        )
        obj = state["object_position"][:, None] + obj_offset
        return hand, obj, hand_offset, obj_offset

    def contact_mask(self, time):
        frame = (
            torch.searchsorted(self.times, time + 1e-6, right=True)
            .sub(1)
            .clamp(0, len(self.times) - 1)
        )
        return self.active[frame]

    def __call__(self, env, state, substep):
        hand, obj, hand_offset, obj_offset = self.points(state)
        link_v = state["link_velocities"][:, self.ids]
        vh = link_v[..., :3] + torch.cross(link_v[..., 3:], hand_offset, dim=-1)
        vo = state["object_velocity"][:, None] + torch.cross(
            state["object_angular_velocity"][:, None].expand(-1, 5, -1), obj_offset, dim=-1
        )
        mask = self.contact_mask(env.time + substep * env.cfg["physics_dt"])
        force = spring_pair_force(
            hand - obj,
            vh - vo,
            self.cfg["spring_n_m"],
            self.cfg["damping_ns_m"],
            self.cfg["force_cap_n"],
            self.strength,
        )
        force *= mask[..., None]
        magnitude = force.norm(dim=-1)
        self.maximum_force = torch.maximum(self.maximum_force, magnitude.amax(dim=1))
        self.impulse += magnitude.sum(dim=1) * env.cfg["physics_dt"]
        hand_forces, hand_torques = (
            torch.zeros_like(env.body_mass[..., None].expand(-1, -1, 3)),
            torch.zeros_like(env.body_mass[..., None].expand(-1, -1, 3)),
        )
        hand_com = quat_apply(
            state["link_transforms"][:, self.ids, 3:7], env.body_com_local[:, self.ids]
        )
        hand_forces[:, self.ids] = force
        hand_torques[:, self.ids] = torch.cross(hand_offset - hand_com, force, dim=-1)
        obj_com = quat_apply(state["object_quaternion"], env.can_com_local)
        object_force = -force.sum(dim=1)
        object_torque = torch.cross(obj_offset - obj_com[:, None], -force, dim=-1).sum(dim=1)
        return hand_forces, hand_torques, object_force, object_torque


def optimize_virtual_contacts(root, settings, output, *, controls_path=None):
    from .policy.floating_env import PhysxResidualEnv
    from .policy.contact_reward import PadCanContacts, ContactWorld

    root, output = Path(root), Path(output)
    cfg = read_config(root / settings["physics_config"])
    cfg["reference"] = settings["reference"]
    cfg["domain_randomization"]["enabled"] = False
    cfg["augmentation"]["enabled"] = False
    cfg["rsi"]["enabled"] = False
    cfg["gravity_curriculum"]["enabled"] = False
    cfg["reference_timing"] = "input_timestamps"
    model = HandModel.load(root / settings["model"])
    reference = ReferenceMotion(
        resolve_demo_path(settings["reference"], root),
        model,
        root / settings["object_geometry"],
        cfg["world_frame"],
    )
    reference.configure_timing(cfg)
    replay_controls = None
    if controls_path is not None:
        controls_path = Path(controls_path)
        with np.load(controls_path, allow_pickle=False) as saved:
            metadata = json.loads(str(saved["metadata_json"]))
            if metadata["reference_sha256"] != reference.metadata["reference_sha256"]:
                raise ValueError("Saved controls belong to a different reference")
            replay_controls = saved["residual_actions"].copy()
        if json.loads((controls_path.parent / "physics.json").read_text()) != cfg:
            raise ValueError("Saved controls require identical physics/controller configuration")
    plan = extract_contacts(reference.data, root / settings["object_mesh"], settings["contacts"])
    scene = CollisionScene(model, root / settings["object_geometry"])
    if settings.get("anchor_mode", "human_surface_proxy") == "robot_surface_seed":
        plan = adapt_contact_anchors(plan, model, scene, reference.data)
    elif settings.get("anchor_mode", "human_surface_proxy") != "human_surface_proxy":
        raise ValueError("Unknown anchor mode")
    plan.save(output / "contacts.npz")
    vcfg = settings["virtual"]
    stages = vcfg["strength_stages"]
    if vcfg["num_envs"] < 8 or vcfg["updates_per_stage"] < 1 or vcfg["hold_s"] <= 0:
        raise ValueError("Use at least eight environments, positive iterations and hold duration")
    if not stages or stages[-1] != 0 or any(b > a or b < 0 for a, b in zip(stages, stages[1:])):
        raise ValueError("Virtual assistance must monotonically decay to exactly zero")
    torch.set_num_threads(1)
    torch.manual_seed(vcfg["seed"])
    np.random.seed(vcfg["seed"])
    env = PhysxResidualEnv(root, model, reference, cfg, vcfg["num_envs"], False)
    env.set_training(False)
    env.reset(randomize=False)
    sensor = PadCanContacts(env)
    env.world = ContactWorld(env.world, sensor)
    query = HardContacts(model, scene, plan)
    starts = np.flatnonzero(plan.active.any(axis=1))
    if not len(starts):
        raise ValueError("No stable contact proxies; inspect contacts.npz")
    at = starts[0]
    links = model.link_transforms(reference.q[at], reference.wrist[at])
    _, witnesses = query.distances(links, reference.object[at])
    pad_points = np.array(
        [links[n][:3, :3].T @ (p - links[n][:3, 3]) for n, p in zip(PAD_LINKS, witnesses)]
    )
    guide = VirtualContacts(env, plan, pad_points, vcfg)
    env.external_wrench_provider = guide
    dt, n = env.task.dt, env.num_envs
    horizon = round(reference.duration / dt) + 1
    hold = round(vcfg["hold_s"] / dt)
    device = env.device
    nominal = reference.sample(np.arange(horizon) * dt)
    generator = torch.Generator(device=device).manual_seed(vcfg["seed"])
    times = torch.arange(horizon, device=device) * dt
    envelope = ((times - 0.2) / 0.3).clamp(0, 1)[None, :, None]
    (output / "physics.json").write_text(json.dumps(cfg, indent=2))
    (output / "contact_mapping.json").write_text(
        json.dumps(
            dict(
                object_anchors_local=plan.anchors.tolist(),
                pad_anchors_local=pad_points.tolist(),
                pad_anchor_source="closest actual pad collision-surface point to human proxy, at annotated grasp onset",
                plan=plan.metadata,
            ),
            indent=2,
        )
    )

    @torch.no_grad()
    def rollout(controls, strength, record=False):
        if controls.shape != (n, horizon, 12):
            raise ValueError("Control sequence shape mismatch")
        guide.strength = float(strength)
        guide.maximum_force.zero_()
        guide.impulse.zero_()
        env.reset(randomize=False)
        reset_count = env.object_reset_count
        cost = torch.zeros(n, device=device)
        errors, contacts, gaps, tables, coupling = [], [], [], [], []
        histories = {
            k: []
            for k in (
                "wrist_position",
                "wrist_quaternion",
                "object_position",
                "object_quaternion",
                "q",
                "full_q",
            )
        }
        target_history = {k: [] for k in ("position", "quaternion", "active_q")}
        force_history = []
        for step in range(horizon + hold):
            command = controls[:, min(step, horizon - 1)]
            _, _, _, _, info = env.step(command, auto_reset=False)
            s, ref = info["state"], info["reference"]
            actual_contacts = sensor.consume()
            mask = guide.contact_mask(ref["time"])
            hp, op, _, _ = guide.points(s)
            gap = (hp - op).norm(dim=-1)
            object_error = (
                (
                    env.motion.points(s["object_position"], s["object_quaternion"])
                    - env.motion.points(ref["object_position"], ref["object_quaternion"])
                )
                .norm(dim=-1)
                .mean(dim=-1)
            )
            present = actual_contacts >= vcfg["minimum_force_n"]
            all_present = (present | ~mask[None]).all(dim=-1).float().mean(dim=0)
            present_fraction = present.float().mean(dim=0)
            contact_cost = ((gap / 0.01).square() * mask).sum(dim=-1) / mask.sum(dim=-1).clamp_min(
                1
            )
            wrist = (s["wrist_position"] - ref["wrist_position"]).norm(dim=-1) / 0.02
            orientation = (
                rotation_error(ref["wrist_quaternion"], s["wrist_quaternion"]).norm(dim=-1) / 0.3
            )
            cost += (
                (object_error / 0.01).square()
                + 0.1 * wrist.square()
                + 0.05 * orientation.square()
                + 0.2 * contact_cost
                + 2 * (1 - all_present) * mask.any(dim=-1)
                + 0.02 * command.square().mean(dim=-1)
                + 0.1
                * (command - controls[:, max(0, min(step - 1, horizon - 1))]).square().mean(dim=-1)
            ) / (horizon + hold)
            errors.append(object_error)
            contacts.append(present_fraction)
            gaps.append(gap)
            coupling.append(
                (s["full_q"] - s["q"] @ env.task.coupling.T - env.task.offset).abs().amax(dim=-1)
            )
            # Conservative collider-box table clearance, already measured at physics substeps.
            tables.append(
                info["metrics"].get(
                    "table_clearance_m", torch.full((n,), float("nan"), device=device)
                )
            )
            if record:
                for key in histories:
                    histories[key].append(s[key].clone())
                for key in target_history:
                    target_history[key].append(info["applied_targets"][key].clone())
                force_history.append(actual_contacts.clone())
        if env.object_reset_count != reset_count:
            raise RuntimeError("Object reset during trajectory optimization")
        errors, contacts, gaps = map(torch.stack, (errors, contacts, gaps))
        phase = torch.arange(horizon + hold, device=device) * dt >= plan.times[at] - 1e-6
        metrics = dict(
            cost=cost,
            mean_object_error_m=errors.mean(dim=0),
            max_object_error_m=errors.amax(dim=0),
            mean_anchor_gap_m=gaps[phase].mean(dim=(0, 2)),
            each_finger_contact_fraction=contacts[phase].mean(dim=0),
            final_object_lift_m=env.state()["object_position"][:, 2] - reference.object[0, 2, 3],
            max_coupling_error_rad=torch.stack(coupling).amax(dim=0),
            max_virtual_force_n=guide.maximum_force.clone(),
            virtual_impulse_ns=guide.impulse.clone(),
        )
        # Simultaneous contact measured at physics substeps; temporal pad averages alone are insufficient.
        if record:
            force_history = torch.stack(force_history)  # time, substep, env, finger
            metrics["all_five_contact_fraction"] = (
                (force_history[phase] >= vcfg["minimum_force_n"])
                .all(dim=-1)
                .float()
                .mean(dim=(0, 1))
            )
            metrics["hold_all_five_contact_fraction"] = (
                (force_history[-hold:] >= vcfg["minimum_force_n"])
                .all(dim=-1)
                .float()
                .mean(dim=(0, 1))
            )
            histories = {k: torch.stack(vals) for k, vals in histories.items()}
            histories.update(
                {"target_" + k: torch.stack(vals) for k, vals in target_history.items()}
            )
            histories.update(
                object_error_m=errors,
                pad_anchor_gap_m=gaps,
                pad_force_n=force_history.permute(0, 2, 1, 3),
                table_clearance_m=torch.stack(tables),
                command_timestamp_s=torch.arange(horizon + hold, device=device) * dt,
                timestamp_s=(torch.arange(horizon + hold, device=device) + 1) * dt,
            )
        return cost, metrics, histories if record else None

    def save_rollout(name, metrics, hist, index):
        report = {k: v[index].detach().cpu().tolist() for k, v in metrics.items()}
        report["virtual_assistance_off"] = report["max_virtual_force_n"] == 0.0
        report["object_lift_tracking_pass"] = bool(
            report["virtual_assistance_off"]
            and report["final_object_lift_m"] > 0.10
            and report["max_object_error_m"] < 0.025
        )
        report["physical_grasp_success"] = bool(
            report["object_lift_tracking_pass"] and report["max_coupling_error_rad"] <= 0.01
        )
        report["five_finger_hold_success"] = bool(
            report["physical_grasp_success"] and report["hold_all_five_contact_fraction"] >= 0.95
        )
        report["classification"] = (
            "physics_rollout; no guarantee for hardware, other placements or arm IK"
        )
        (output / (name + ".json")).write_text(json.dumps(report, indent=2))
        data = {
            k: (v if k in ("timestamp_s", "command_timestamp_s") else v[:, index])
            .detach()
            .cpu()
            .numpy()
            for k, v in hist.items()
        }
        np.savez_compressed(
            output / (name + ".npz"),
            **data,
            active_joint_names=np.array(model.active_names),
            full_joint_names=np.array(model.full_names),
            pad_link_names=np.array(PAD_LINKS),
            metadata_json=np.array(
                json.dumps(
                    dict(
                        length_unit="m",
                        angle_unit="rad",
                        quaternion_order="xyzw",
                        coordinate_frame=cfg["world_frame"],
                        timestamps="state timestamp is end of control tick; command timestamp is start",
                        physics_dt_s=cfg["physics_dt"],
                        control_dt_s=dt,
                        classification="physics_debug; use separate geometric and coupling validation",
                    )
                )
            ),
        )
        return report

    zero = torch.zeros((n, horizon, 12), device=device)
    seed_controls = zero.clone()
    grip_ramp = ((times - plan.times[at] + 0.2) / 0.3).clamp(0, 1)
    for slot, closing in enumerate((0.25, 0.5, 0.75, 1.0), start=1):
        seed_controls[slot, :, 7:] = grip_ramp[:, None] * closing
    seed_cost, baseline_metrics, baseline_hist = rollout(seed_controls, 0.0, True)
    baseline_report = save_rollout("baseline", baseline_metrics, baseline_hist, 0)
    seed_index = 1 + int(seed_cost[1:5].argmin())
    seed_report = save_rollout("grip_seed_only", baseline_metrics, baseline_hist, seed_index)
    controls = zero[0].clone()
    records = []
    count = len(stages) * vcfg["updates_per_stage"]
    best_final = controls.clone()
    if replay_controls is not None:
        if (
            replay_controls.shape != (horizon, 12)
            or not np.isfinite(replay_controls).all()
            or np.abs(replay_controls).max() > 1.000001
        ):
            raise ValueError("Invalid saved residual control sequence")
        count = 0
        best_final = torch.tensor(replay_controls, dtype=torch.float32, device=device)
        controls = best_final.clone()
    best_final_cost = float("inf")
    for i in range(count):
        strength = stages[i // vcfg["updates_per_stage"]]
        fraction = i / max(count - 1, 1)
        sigma = vcfg["noise_initial"] * (vcfg["noise_final"] / vcfg["noise_initial"]) ** fraction
        noise = (
            torch.randn((n, 12, vcfg["control_knots"]), generator=generator, device=device) * sigma
        )
        noise = torch.nn.functional.interpolate(
            noise, size=horizon, mode="linear", align_corners=True
        ).transpose(1, 2)
        candidates = (controls[None] + noise * envelope).clamp(-1, 1)
        candidates[0] = controls
        candidates[1] = zero[0]
        if vcfg.get("grip_seeds", False):
            # Explicit independent initialization prior: coherent finger flexion
            # through the annotated grasp, not an imported policy or learned action.
            grip_ramp = ((times - plan.times[at] + 0.2) / 0.3).clamp(0, 1)
            for slot, closing in enumerate((0.25, 0.5, 0.75, 1.0), start=3):
                candidates[slot].zero_()
                candidates[slot, :, 7:] = grip_ramp[:, None] * closing
        if strength == 0 and best_final_cost < float("inf"):
            candidates[2] = best_final
        costs, metrics, _ = rollout(candidates, strength)
        best = int(costs.argmin())
        if strength == 0 and float(costs[best]) < best_final_cost:
            best_final_cost, best_final = float(costs[best]), candidates[best].clone()
        controls = sampling_update(controls, candidates, costs, vcfg["temperature"])
        row = dict(
            update=i + 1,
            strength=strength,
            sigma=sigma,
            best_cost=float(costs[best]),
            mean_object_error_mm=float(metrics["mean_object_error_m"][best]) * 1000,
            max_virtual_force_n=float(metrics["max_virtual_force_n"][best]),
        )
        records.append(row)
        print("[virtual-contact] " + json.dumps(row), flush=True)
    final_candidates = best_final[None].expand(n, -1, -1).clone()
    final_candidates[1] = controls
    final_candidates[2] = zero[0]
    costs, metrics, hist = rollout(final_candidates, 0.0, True)
    # Select among the best sampled sequence, weighted mean and unchanged baseline.
    selected = int(costs[:3].argmin()) if replay_controls is None else 0
    result = save_rollout("unassisted", metrics, hist, selected)
    save_rollout("baseline_recheck", metrics, hist, 2)
    np.savez_compressed(
        output / "controls.npz",
        timestamps_s=times.cpu().numpy(),
        residual_actions=final_candidates[selected].cpu().numpy(),
        action_scales=env.task.scales.cpu().numpy(),
        wrist_reference_position=nominal["wrist_position"],
        wrist_reference_quaternion_xyzw=nominal["wrist_quaternion"],
        finger_reference_q_rad=nominal["q"],
        object_reference_transform=reference.object,
        metadata_json=np.array(
            json.dumps(
                dict(
                    reference=settings["reference"],
                    reference_sha256=digest(resolve_demo_path(settings["reference"], root)),
                    method="SPIDER-inspired sampling plus annealed pair forces",
                    assistance_at_execution=0,
                    closed_loop_controller="PD plus measured-angle finger governor",
                    requires_object_pose_feedback=False,
                )
            )
        ),
    )
    report = dict(
        method="SPIDER-inspired, independently implemented PhysX adaptation",
        baseline=baseline_report,
        grip_seed_only=seed_report,
        unassisted=result,
        selected_candidate=selected,
        iterations=records,
        replay_source=None if controls_path is None else str(controls_path),
        all_final_candidates_virtual_force_max_n=float(metrics["max_virtual_force_n"].max()),
        limitations=[
            "proxy skeleton contacts",
            "nominal physics only; no worst-case perturbation optimizer",
            "bounded sampling budget; not an optimality certificate",
            "physical self-collision setting inherited; no arm IK",
            "rubber/motor parameters are not hardware calibrated",
        ],
        env_metadata=env.metadata,
    )
    (output / "report.json").write_text(json.dumps(report, indent=2))
    env.external_wrench_provider = None
    env.close()
    return report
