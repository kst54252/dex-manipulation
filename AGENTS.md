# Repository instructions

## Commits

- Use Conventional Commits: `<type>(<scope>): <summary>` with short imperative summaries.
- Group commits by feature. Use `!` and a `BREAKING CHANGE:` footer for breaking interfaces.
- Run relevant checks before committing; report failures accurately.

## Files and data

- Reusable code: `src/dex_manipulation/`; entry points: `scripts/`; settings: `config/`; documentation: `docs/`.
- Keep tests, troubleshooting, analyses, reports, runs, checkpoints and share packages under ignored `local/`. Never force-add them.
- Preserve source assets, data and checkpoint contracts. Record methodology sources in `docs/PROVENANCE.md`.
- Keep `data/demo*/raw/` images and NPZ untracked; track only placement READMEs. Keep DexYCB attribution in `docs/dataset.md`.
- Demo 1 is `data/demo1/` (40 poses, historical `original`); demo 2 is `data/demo2/` (27 poses, historical `current`). Use `1`/`2` in user-facing commands.
- Resolve historical input paths with `resolve_demo_path` rather than editing saved metadata.
- Group task-specific settings under `config/tasks/<task>/` and adapters under `src/dex_manipulation/tasks/`. Default task is `can_pick`; its existing `data/demo1/` and `data/demo2/` remain stable.
- Add new task inputs under `data/<task>/demoN/` and task assets under `assets/tasks/<task>/`. Raw inputs remain untracked. Keep task identity separate from demo number and never silently reuse another task's policy or physics.

## Documentation

- Explain project purpose, structure, configuration and commands concisely.
- Use `run.sh` for standard playback and training examples.
- Omit progress diaries, experiment histories and repeated implementation/verification status from project Markdown.
- Keep operational requirements, dataset attribution and methodology sources.

## Training

- Run long training only when explicitly requested. Continue authorized training through follow-up messages.
- Train headlessly, without intermediate reports or policy analysis; monitor execution health and output saving only.
- After training, briefly report completion and saved log metrics. Do not automatically run a simulator, GUI, rollout evaluation or extra experiment.
- Distinguish training metrics from independently evaluated performance.
