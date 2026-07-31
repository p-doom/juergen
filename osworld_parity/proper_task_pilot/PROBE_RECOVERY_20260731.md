# Seed-503 infrastructure recovery amendment (2026-07-31)

This operational amendment does not change the preregistered estimand, model,
sampling, task list, or 96 intended seed/task cells. Job 135555 (seed 503)
published two infrastructure-valid results for canonical task indices 0 and 1,
then exited after 897 GPU-seconds while attempting index 2 because the OSWorld
screenshot service returned HTTP 500/connection reset and no byte screenshot.
The failed run has no trusted `run_manifest.json`; its `failure.json` SHA-256 is
`fba50a7f3351f5ace7e4132bd1266994f926832f7ea230d82d892e2481763331`.

Recovery job 135575 is authorized to reuse only the sealed valid results at
indices 0 and 1 and to run exactly indices 2 through 11 with seed 503,
temperature 0.7, top-p 0.95, the same model/runtime, the canonical forward task
order, and fresh VM/session state per rollout. The infrastructure-invalid
index-2 attempt is unobserved: it produced no result record and is excluded,
not counted as an intended cell. No valid index 0 or 1 rollout may be rerun.
The recovery writes a separate artifact, and the final CPU merge must require
one and only one infrastructure-valid result for every intended seed/task key.

The first half consumed 6,400 GPU-seconds. Failed job 135555 consumed 897.
Seed-601 job 135556 later completed in 1,074 seconds, proving 906 seconds unused
relative to its 1,980-second reservation. Slurm rounds a 19:23 request upward,
so job 135575 was first corrected while pending to 19:00 (1,140 seconds), then
extended while still pending to 34:00 (2,040 seconds) after job 135556 finished.
At that point the fail-closed worst case was
`6400 + 897 + 1074 + 1980 + 1980 + 2040 = 14371` GPU-seconds (29 seconds margin).

Seed-809 job 135559 then completed in 1,249 seconds. This proved a theoretical
safe recovery ceiling of
`14400 - (6400 + 897 + 1074 + 1249 + 1980) = 2800` seconds while seed 701 was
still running. Slurm denied increasing the already-running recovery job, so its
limit remained 2,040 seconds. The updated worst case is therefore
`6400 + 897 + 1074 + 1249 + 1980 + 2040 = 14360` GPU-seconds (40 seconds margin).
Only final `sacct` elapsed seconds may be used in the sealed GPU-accounting
record, whose validated total must not exceed 14,400.

Seed-701 job 135558 subsequently completed in 1,409 seconds. With all intact
second-half seeds complete, the recovery's full 2,040-second allocation gives
the final fail-closed bound
`6400 + 897 + 1074 + 1249 + 1409 + 2040 = 13069` GPU-seconds, leaving 1,331
seconds below the cap. The recovery limit remains 2,040 seconds.
