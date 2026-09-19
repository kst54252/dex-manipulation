# Repository instructions

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
- Run checks appropriate to the changed features before committing, and report
  failures or unverified behavior honestly. Long policy training is run by the
  user unless they explicitly request it.
