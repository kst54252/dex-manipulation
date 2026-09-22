"""Attach the configured grasp objective without changing saved policy contracts."""

import hashlib
import json

from .contact_reward import SCHEMA, contact_schedule


def attach_grasp_reward(env, metadata, training_reference):
    dense = bool(env.cfg.get("grasp_task"))
    contact_only = bool(env.cfg.get("five_finger_grasp"))
    if dense and contact_only:
        raise ValueError("Choose one grasp objective; contact rewards must not be counted twice")
    if dense:
        from .grasp_reward import attach_grasp_task

        attach_grasp_task(env)
    elif contact_only:
        from .contact_reward import attach_contact_reward

        attach_contact_reward(env)
        # Legacy dedicated demo2 runs added this task signature after computing
        # the base contract. Keep it byte-for-byte, even when playback is retimed.
        schedule = contact_schedule(training_reference, env.cfg["five_finger_grasp"])
        contract = dict(base_contract=metadata["contract_hash"], task=SCHEMA, schedule=schedule)
        metadata["contract_hash"] = hashlib.sha256(
            json.dumps(contract, sort_keys=True).encode()
        ).hexdigest()
        metadata["task"] = dict(
            schema=SCHEMA,
            schedule=schedule,
            implementation="dedicated demo2 contact task; shared unchanged PhysX environment and RSL-RL PPO",
        )
