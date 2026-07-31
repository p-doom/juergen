# Coarse official-pilot infrastructure (do not execute yet)

This package is the ROADMAP 3.5 release boundary and CPU aggregator. It is not
an official task runner. In particular, it contains no source adapter and no
model or VM invocation.

The only safe pre-release action is running the mock-only unit tests. Once
ROADMAP 3.1--3.4 have passed, release operators must independently produce the
two signed gates described in `PREREGISTRATION.md`. The release recipe validates
them and emits a sanitized launch authorization. A separately reviewed broker
implementation may then be injected through `with_authorized_source`; the
broker must emit rows conforming to `records.py`.

The aggregate recipe re-verifies both gates before opening rows, requires the
complete paired 32-episode grid, and writes only preregistered metrics. Neither
recipe accepts an official task path, task file, model, checkpoint, VM, or KVM
input.
