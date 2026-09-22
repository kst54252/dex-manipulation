"""Opt-in five-pad/can rewards using filtered, per-physics-step PhysX forces.

The shared runner selects this adapter for a five_finger_grasp configuration.
It preserves the base controller, observations and saved checkpoint contract.
"""

import math
import numpy as np
import torch

from ..materials import PAD_BODIES

FINGERS = ("thumb", "index", "middle", "ring", "little")
SCHEMA = "demo2_five_pad_contact_v1"


def contact_schedule(reference, settings):
    """Use source frame IDs, never slider indices or guessed capture FPS."""
    if settings.get("schema") != SCHEMA:
        raise ValueError("Unsupported five-finger reward schema")
    ids = np.asarray(reference.data["frame_ids"])
    found = np.flatnonzero(ids == settings["start_frame_id"])
    if len(found) != 1:
        raise ValueError("Contact start frame must occur exactly once in this reference")
    start, end = float(reference.times[found[0]]), float(reference.duration)
    if not 0 <= start < end:
        raise ValueError("Contact reward requires a nonempty reference interval")
    for key in ("minimum_force_n", "maximum_force_n", "tracking_std_m"):
        if not math.isfinite(settings[key]) or settings[key] <= 0:
            raise ValueError(f"Invalid contact parameter: {key}")
    if settings["maximum_force_n"] <= settings["minimum_force_n"]:
        raise ValueError("Maximum contact force must exceed the detection threshold")
    for key in ("finger_weight", "all_fingers_weight", "missing_weight", "excess_force_weight"):
        if not math.isfinite(settings[key]) or settings[key] < 0:
            raise ValueError(f"Invalid contact weight: {key}")
    return dict(
        start_frame_id=int(ids[found[0]]),
        start_time_s=start,
        end_frame_id=int(ids[-1]),
        end_time_s=end,
        clock="reference command time; RSI and augmentation preserve this clock",
    )


def contact_terms(forces, time, object_error, early_failure, settings, schedule):
    """forces [physics substeps, environments, 5], in newtons, can pairs only.

    Simultaneity is evaluated before averaging over substeps. Alternating
    fingers cannot earn the all-five bonus. Force above threshold earns no
    extra contact reward; excessive squeezing is penalized. All outputs are
    per-environment, suitable for the existing rollout logger.
    """
    if forces.ndim != 3 or forces.shape[-1] != 5 or not forces.shape[0]:
        raise ValueError("Expected [physics substeps, environments, five fingers] forces")
    on = (
        (time + 1e-6 >= schedule["start_time_s"]) & (time <= schedule["end_time_s"] + 1e-6)
    ).float()
    contacts = forces >= settings["minimum_force_n"]
    duty = contacts.float().mean(0)
    simultaneous = contacts.all(-1).float().mean(0)
    fraction = duty.mean(-1)
    # Dense force onset shaping stays below the physical-contact threshold.
    # No proximity/keypoint or table contact is counted as a contact.
    onset = (forces / settings["minimum_force_n"]).clamp(0, 1).mean((0, 2))
    quality = torch.exp(-object_error / settings["tracking_std_m"]) * (~early_failure).float()
    excess = (forces / settings["maximum_force_n"] - 1).clamp(0, 10).square().mean((0, 2))
    terms = dict(
        grasp_fingers=on * quality * settings["finger_weight"] * onset,
        grasp_all_five=on * quality * settings["all_fingers_weight"] * simultaneous,
        grasp_missing=-on * settings["missing_weight"] * (1 - fraction),
        grasp_excess_force=-on * settings["excess_force_weight"] * excess,
    )
    metrics = dict(
        grasp_active=on,
        grasp_contact_count=duty.sum(-1),
        grasp_all_five_duty=simultaneous,
        grasp_window_all_five_duty=on * simultaneous,
        grasp_window_contact_fraction=on * fraction,
        grasp_tracking_quality=quality,
        grasp_max_force_n=forces.amax((0, 2)),
    )
    for i, name in enumerate(FINGERS):
        metrics[f"grasp_{name}_duty"] = duty[:, i]
        metrics[f"grasp_window_{name}_duty"] = on * duty[:, i]
        metrics[f"grasp_{name}_force_n"] = forces[:, :, i].mean(0)
    metrics.update({"reward_" + key: value for key, value in terms.items()})
    return sum(terms.values()), metrics


class PadCanContacts:
    """Linear-memory, explicitly paired pads and same-environment cans."""

    def __init__(self, env):
        from pxr import PhysxSchema

        self.env = env
        self.samples = []
        links = env.robot._physics_view.link_paths
        indices = [env.robot.body_names.index(name) for name in PAD_BODIES]
        cans = list(env.can.prim_paths)
        if len(links) != env.num_envs or len(cans) != env.num_envs:
            raise ValueError("Cannot map contact pairs to environment rows")
        self.paths = [row[i] for row in links for i in indices]
        filters = [[can] for can in cans for _ in indices]
        for path in self.paths:
            prim = env.world.stage.GetPrimAtPath(path)
            PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.0)
        self.view = env.world.physics_sim_view.create_rigid_contact_view(self.paths, filters)
        if self.view.sensor_count != 5 * env.num_envs or self.view.filter_count != 1:
            raise RuntimeError(
                "Expected exactly five pad sensors and one can filter per environment"
            )
        # Tensor order is verified rather than assumed from regex/numeric sorting.
        reported = list(self.view.sensor_paths)
        if len(set(reported)) != len(self.paths) or set(reported) != set(self.paths):
            raise RuntimeError("Contact view sensor paths differ from named pad bodies")
        lookup = {path: i for i, path in enumerate(reported)}
        self.order = torch.tensor([lookup[p] for p in self.paths], device=env.device)
        actual_filters = list(self.view.filter_paths)
        if actual_filters != filters:
            # Some API versions return a flat row for the single filter.
            if actual_filters != [f[0] for f in filters]:
                raise RuntimeError(
                    "PhysX can filter mapping differs from the requested explicit pairs"
                )
        self.metadata = dict(
            schema=SCHEMA,
            finger_order=list(FINGERS),
            pad_bodies=list(PAD_BODIES),
            sensors=self.view.sensor_count,
            filters_per_sensor=self.view.filter_count,
            force_unit="N",
            impulse_dt_s=env.cfg["physics_dt"],
            pair_filter="one exact same-environment can per pad; table/other fingers excluded",
            aggregation="norm of pair normal-force vector at every physics step; simultaneous threshold then temporal mean",
        )

    def capture(self):
        # PhysX reuses its tensor buffer: copy the scalar magnitudes immediately.
        force = self.view.get_contact_force_matrix(self.env.cfg["physics_dt"])
        self.samples.append(force[self.order, 0].norm(dim=-1).reshape(self.env.num_envs, 5))

    def consume(self):
        expected = self.env.cfg["control_decimation"]
        if len(self.samples) != expected:
            raise RuntimeError(
                f"Expected {expected} contact substeps; received {len(self.samples)}"
            )
        samples = torch.stack(self.samples)
        self.samples.clear()
        # A missing/invalid sensor must never silently masquerade as no contact.
        if not torch.isfinite(samples).all():
            raise RuntimeError("Nonfinite PhysX contact forces")
        return samples


class ContactWorld:
    """Delegate the existing world, adding a read AFTER each physics step."""

    def __init__(self, world, contacts):
        self._world, self._contacts = world, contacts

    def __getattr__(self, name):
        return getattr(self._world, name)

    def step(self, *args, **kwargs):
        result = self._world.step(*args, **kwargs)
        self._contacts.capture()
        return result


class ContactTask:
    """Compose the existing task and reward; no changed reset or PPO lifecycle."""

    def __init__(self, task, contacts, settings, schedule):
        self.base, self.contacts, self.settings, self.schedule = task, contacts, settings, schedule

    def __getattr__(self, name):
        return getattr(self.base, name)

    def score(self, state, ref, *args, **kwargs):
        reward, term, trunc, metrics = self.base.score(state, ref, *args, **kwargs)
        extra, contact = contact_terms(
            self.contacts.consume(),
            ref["time"],
            metrics["object_keypoint_error_m"],
            metrics["early_failure"],
            self.settings,
            self.schedule,
        )
        metrics.update(contact)
        return reward + extra * self.dt, term, trunc, metrics


def attach_contact_reward(env):
    if isinstance(env.task, ContactTask):
        raise ValueError("Contact reward already attached")
    settings = env.cfg["five_finger_grasp"]
    schedule = contact_schedule(env.reference, settings)
    contacts = PadCanContacts(env)
    env.world = ContactWorld(env.world, contacts)
    env.task = ContactTask(env.task, contacts, settings, schedule)
    env.metadata["five_finger_grasp"] = dict(
        schedule=schedule, sensor=contacts.metadata, settings=settings
    )
    return contacts


def contact_summary(stats):
    """Conditional rates: the approach interval must not dilute grasp metrics."""
    m = stats["metrics"]
    active = m["grasp_active"]["mean"]
    return dict(
        active_sample_fraction=active,
        all_five_fraction=(m["grasp_window_all_five_duty"]["mean"] / active if active else None),
        finger_contact_fraction=(
            m["grasp_window_contact_fraction"]["mean"] / active if active else None
        ),
        per_finger_fraction={
            name: (m[f"grasp_window_{name}_duty"]["mean"] / active if active else None)
            for name in FINGERS
        },
    )
