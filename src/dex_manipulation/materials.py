"""Localized Revo2 pad contact; source USD assets remain read-only.

The five touch bodies have separate collision meshes. Tip landmark frames and
distal finger shells must never be used as a substitute for these pad bodies.
Values are an explicitly uncalibrated contact approximation, not measured rubber.
"""

import math

PAD_BODIES = tuple(
    "right_" + finger + "_touch_link" for finger in ("thumb", "index", "middle", "ring", "pinky")
)


def pad_settings(config):
    settings = config.get("contact_materials")
    if not settings or not settings.get("enabled", False):
        return None
    if tuple(settings["pad_bodies"]) != PAD_BODIES:
        raise ValueError("Rubber contact requires the five verified Revo2 touch bodies")
    for key in ("static_friction", "dynamic_friction", "stiffness_n_m", "damping_ns_m"):
        value = settings[key]
        if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"Invalid pad material parameter: {key}")
    if settings["dynamic_friction"] > settings["static_friction"]:
        raise ValueError("Pad dynamic friction must not exceed static friction")
    if settings["friction_combine_mode"] != "average":
        raise ValueError("This profile preserves the existing average friction combine rule")
    return settings


def bind_pad_material(stage, robot_path, config, material_path):
    """Bind only verified pad collision subtrees, including instance proxies."""
    settings = pad_settings(config)
    if settings is None:
        return None
    from pxr import Usd, UsdPhysics, UsdShade, PhysxSchema

    root = stage.GetPrimAtPath(robot_path)
    bodies = [
        p
        for p in Usd.PrimRange(root)
        if p.HasAPI(UsdPhysics.RigidBodyAPI) and p.GetName() in PAD_BODIES
    ]
    if sorted(p.GetName() for p in bodies) != sorted(PAD_BODIES):
        raise ValueError("Missing or duplicate physical Revo2 pad bodies in this assembly")
    material = UsdShade.Material.Define(stage, material_path)
    physics = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    physics.CreateStaticFrictionAttr(settings["static_friction"])
    physics.CreateDynamicFrictionAttr(settings["dynamic_friction"])
    physics.CreateRestitutionAttr(0.0)
    physx = PhysxSchema.PhysxMaterialAPI.Apply(material.GetPrim())
    physx.CreateFrictionCombineModeAttr("average")
    physx.CreateRestitutionCombineModeAttr("average")
    physx.CreateDampingCombineModeAttr("average")
    physx.CreateCompliantContactStiffnessAttr(settings["stiffness_n_m"])
    physx.CreateCompliantContactDampingAttr(settings["damping_ns_m"])
    physx.CreateCompliantContactAccelerationSpringAttr(False)
    paths = []
    for body in bodies:
        colliders = [
            p
            for p in Usd.PrimRange(body, Usd.TraverseInstanceProxies())
            if p.HasAPI(UsdPhysics.CollisionAPI)
        ]
        if len(colliders) != 1:
            raise ValueError(f"Expected one separate collision mesh on pad {body.GetPath()}")
        # Binding the non-instance body avoids copying or changing shared meshes.
        UsdShade.MaterialBindingAPI.Apply(body).Bind(
            material, UsdShade.Tokens.strongerThanDescendants, "physics"
        )
        resolved, _ = UsdShade.MaterialBindingAPI(colliders[0]).ComputeBoundMaterial("physics")
        if resolved.GetPath() != material.GetPath():
            raise ValueError(f"Pad material did not resolve on {colliders[0].GetPath()}")
        paths.append(str(colliders[0].GetPath()))
    return dict(
        profile=settings,
        material_path=material_path,
        pad_colliders=paths,
        model="Rigid bodies with localized implicit spring-damper contacts; no deformable mesh",
    )


def pad_shape_mask(env):
    """Map named bodies to PhysX shape slots, checking the runtime layout."""
    import torch

    if not hasattr(env, "body_shape_counts"):
        env.body_shape_counts = [
            env.world.physics_sim_view.create_rigid_body_view(path).max_shapes
            for path in env.robot._physics_view.link_paths[0]
        ]
    counts = env.body_shape_counts
    if sum(counts) != env.robot._physics_view.max_shapes:
        raise ValueError("Unknown articulation material shape layout")
    for name in PAD_BODIES:
        if env.robot.body_names.count(name) != 1 or counts[env.robot.body_names.index(name)] != 1:
            raise ValueError(f"Cannot isolate physical pad collider: {name}")
    return torch.tensor(
        [
            name in PAD_BODIES
            for name, count in zip(env.robot.body_names, counts)
            for _ in range(count)
        ],
        dtype=torch.bool,
    )


def enforce_pad_contact(env):
    """Restore pad physics after randomization/checkpoint material tensor writes.

    The legacy material tensor also writes restitution, which can clear compliant
    stiffness internally. Set compliance last and verify with the PhysX readback.
    Other shape friction, masses, drives and geometry are never changed here.
    """
    settings = pad_settings(env.cfg)
    if settings is None:
        return
    import torch

    view = env.robot._physics_view
    mask = pad_shape_mask(env)
    if not hasattr(env, "_pad_physics_view"):
        paths = [view.link_paths[0][env.robot.body_names.index(name)] for name in PAD_BODIES]
        if hasattr(env, "env_paths"):
            paths = [p.replace(env.env_paths[0], "/World/envs/env_*", 1) for p in paths]
        env._pad_physics_view = env.world.physics_sim_view.create_rigid_body_view(paths)
    pads = env._pad_physics_view
    if pads.count != 5 * env.num_envs or pads.max_shapes != 1:
        raise ValueError(
            "Pad-only PhysX view did not isolate five single-shape bodies per environment"
        )
    # The setter takes (mu_static,mu_dynamic,k,c), unlike the two-value getter.
    # Use a pad-only view: never pass rigid material's +inf stiffness sentinel
    # back through a compliant-material setter for the other robot shapes.
    properties = (
        torch.tensor(
            [
                settings["static_friction"],
                settings["dynamic_friction"],
                settings["stiffness_n_m"],
                settings["damping_ns_m"],
            ]
        )
        .expand(pads.count, 1, 4)
        .contiguous()
    )
    modes = torch.zeros((pads.count, 1, 3), dtype=torch.uint8)
    pads.set_compliant_material_properties(
        properties, modes, torch.arange(pads.count, dtype=torch.int32)
    )
    actual, _ = view.get_compliant_material_properties()
    expected = torch.tensor([settings["stiffness_n_m"], settings["damping_ns_m"]])
    if not torch.allclose(actual.cpu()[:, mask], expected.expand(env.num_envs, 5, 2)):
        raise RuntimeError("PhysX did not retain the requested compliant pad material")
    friction = view.get_material_properties().cpu()[:, mask, :2]
    if not torch.allclose(
        friction,
        torch.tensor([settings["static_friction"], settings["dynamic_friction"]]).expand(
            env.num_envs, 5, 2
        ),
    ):
        raise RuntimeError("PhysX did not retain the requested pad friction")
    rigid = actual.cpu()[:, ~mask, 0]
    if not torch.all((rigid == 0) | torch.isposinf(rigid)):
        raise RuntimeError("A non-pad robot collider has unexpectedly become compliant")


def pad_contact_report(env):
    if pad_settings(env.cfg) is None:
        return None
    import torch

    mask = pad_shape_mask(env)
    view = env.robot._physics_view
    material = view.get_material_properties().cpu()
    compliant, _ = view.get_compliant_material_properties()
    offsets = view.get_rest_offsets().cpu()
    report = dict(
        settings=env.cfg["contact_materials"],
        pad_shape_count=int(mask.sum()),
        other_robot_shape_count=int((~mask).sum()),
        groups={},
    )
    for label, selected in (("pads", mask), ("other_robot", ~mask)):
        report["groups"][label] = dict(
            static_friction_range=[
                float(material[:, selected, 0].min()),
                float(material[:, selected, 0].max()),
            ],
            dynamic_friction_range=[
                float(material[:, selected, 1].min()),
                float(material[:, selected, 1].max()),
            ],
            stiffness_n_m=float(env.cfg["contact_materials"]["stiffness_n_m"])
            if label == "pads"
            else None,
            contact_model="compliant" if label == "pads" else "rigid",
            rest_offset_range_m=[float(offsets[:, selected].min()), float(offsets[:, selected].max())],
        )
    for label, rigid in (("can", env.can), ("table", env.table)):
        values, _ = rigid._physics_view.get_compliant_material_properties()
        if not torch.all((values[..., 0] == 0) | torch.isposinf(values[..., 0])):
            raise RuntimeError(f"{label} must retain rigid contact material")
        values = rigid._physics_view.get_material_properties().cpu()
        report["groups"][label] = dict(
            stiffness_n_m=0.0,
            rigid_contact=True,
            rest_offset_range_m=[float(rigid._physics_view.get_rest_offsets().min()),
                                 float(rigid._physics_view.get_rest_offsets().max())],
            static_friction_range=[float(values[..., 0].min()), float(values[..., 0].max())],
            dynamic_friction_range=[float(values[..., 1].min()), float(values[..., 1].max())],
        )
    return report
