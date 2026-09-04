# Fork CI policy

This note is for maintainers of `rickyren/qlib`. Stop reading if you are
working in the upstream `microsoft/qlib` repository: upstream owns its own
support matrix and merge requirements.

This fork is immutable source infrastructure for a macOS 15 Intel / Python
3.14 application. Pull requests therefore run one source gate that matches that
environment:

- `test_qlib_from_source.yml` for lint, configuration smoke tests, and the
  non-slow test suite.

The gate installs the fork runtime plus only the tools it needs. It
intentionally excludes the inherited `docs`, `package`, and `rl` extras, which
are not part of the application runtime; in particular, the upstream
documentation-only SciPy pin does not support Python 3.14 and PyTorch does not
publish a Python 3.14 Intel macOS wheel.

`test_qlib_from_source_slow.yml` and `test_qlib_from_pip.yml` are manual-only;
the application neither uses the inherited slow/RL coverage as a product gate
nor installs Qlib from PyPI. `release.yml` is also manual-only; merging to the
fork's `main` never publishes a package or starts a release compatibility
matrix. The inherited stale-issue workflow is manual-only as well. The source
gate runs on pull requests, not again after the squash merge.

The application's local macOS virtual environment is the final runtime
authority. If a fork change is proposed back to `microsoft/qlib`, validate it
through the upstream pull request and its current cross-platform policy instead
of expanding this fork's default PR matrix.
