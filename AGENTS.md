# Repository instructions

## Demo names

- Use **demo 1 / 1번 데모** for `data/demo1/` (imported 40 poses), and **demo 2 / 2번 데모**
  for `data/demo2/` (retained 27 poses). The user swapped their numbers on
  2026-09-21; historical `original` means demo 1 and `current` means demo 2.
  Both demos remain in active use.
- Use `1` / `2` in playback commands and user-facing labels. Keep source names
  when recording provenance. Preserve input bytes and checkpoint contracts;
  resolve historical data paths through `resolve_demo_path` when reading saved runs.

## Commit convention

- Use Conventional Commits for all future commits in this repository:
  `<type>(<scope>): <summary>`. The scope is optional.
- Choose the type that describes the change, such as `feat`, `fix`, `refactor`,
  `perf`, `docs`, `test`, `build`, `ci`, or `chore`.
- Prefer short, imperative summaries and clear scopes such as `fk`, `data`,
  `retargeting`, `ik`, `sim`, and `policy`.
- Keep each commit focused on one coherent feature or change. When asked to
  commit a large batch, group it by major functionality.
- Use `!` and a `BREAKING CHANGE:` footer when a change breaks a public interface.

## Project files and local artifacts

- Keep reusable code in `src/dex_manipulation/`, entry points in `scripts/`,
  configuration in `config/`, and usage/methodology documentation in `docs/`.
- Keep tests, troubleshooting tools, analyses, validation reports, generated runs,
  checkpoints, and share packages under the ignored `local/` directory. Do not
  force-add these files to Git; this separation is an explicit user preference.
- Preserve original assets and input data. Record methodology sources and
  implementation choices in `docs/PROVENANCE.md`.
- Keep DexYCB raw images and NPZ annotations in `data/demo*/raw/` untracked;
  only its placement README belongs in Git. Keep dataset attribution in
  `docs/dataset.md` and preserve local raw bytes when changing Git tracking.
- Run checks appropriate to the changed features before committing, and report
  failures or unverified behavior honestly. Long policy training is run by the
  user unless they explicitly request it.
