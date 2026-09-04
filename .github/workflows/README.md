# Fork CI policy

This note is for maintainers of `rickyren/qlib`. Stop reading if you are
working in the upstream `microsoft/qlib` repository: upstream owns its own
support matrix and merge requirements.

This fork is immutable source infrastructure for a macOS 15 Intel / Python
3.14 application. Pull requests therefore run only the two source gates that
match that environment:

- `test_qlib_from_source.yml` for lint, configuration smoke tests, and the
  non-slow test suite;
- `test_qlib_from_source_slow.yml` for slow tests.

Both gates install the fork runtime plus only the tools needed by that gate.
They intentionally exclude the inherited `docs`, `package`, and `rl` extras,
which are not part of the application runtime; in particular, the upstream
documentation-only SciPy pin does not support Python 3.14.

`test_qlib_from_pip.yml` is manual-only because the application installs this
fork from a pinned Git revision rather than from PyPI. `release.yml` is also
manual-only; merging to the fork's `main` never publishes a package or starts a
release compatibility matrix. The inherited stale-issue workflow is manual-only
as well. The source gates run on pull requests, not again after the squash merge.

The application's local macOS virtual environment is the final runtime
authority. If a fork change is proposed back to `microsoft/qlib`, validate it
through the upstream pull request and its current cross-platform policy instead
of expanding this fork's default PR matrix.
