# Execution inputs

`manifest.json` lists the SHA256 and original modification time of every packaged file.
`run.sh` restores missing files into ignored `local/` without overwriting existing runs.
`./run.sh check` verifies the archives and restored files.

- `policy_demo2.zip`: latest completed demo2 checkpoint, saved settings and completion metadata.
- `references.zip`: derived policy inputs and arm IK trajectories.
- `placements.zip`: demo2 reference-matched random-placement map and required provenance inputs.
- `execution.zip`: recorded 12-joint commands, matching simulation validation and baseline tactile data.

The files retain their original bytes and physics contracts. Current playback pad physics can differ from training;
`--contact-materials checkpoint` selects the trained profile. Robot calibration and connection settings are supplied on site.
Source human demonstrations are derived from [DexYCB](../docs/dataset.md). No raw RGB images or annotations are included.
See [installation](../docs/installation.md).
