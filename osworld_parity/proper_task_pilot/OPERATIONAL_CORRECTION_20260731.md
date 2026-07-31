# Probe operational correction — 2026-07-31

This correction was recorded at 03:01 CEST, before any second-half seed was
submitted. It changes scheduling only; the preregistered cells, tasks, model,
runtime, sampling parameters, stopping rules, and four aggregate GPU-hour cap
remain unchanged.

The earlier 04:40 CEST deadline was based on an assumed Nishant reservation.
The live Slurm check found no reservation matching `nishant` or job `134957`.
Job `134957` was an ordinary `PENDING (Resources)` job with `Deadline=N/A` and
`SchedNodeList=hai[001-002,004]`; it did not reserve or request the probe nodes
`hai003`, `hai007`, or `hai008`.

Accordingly, second-half GPU seeds and the final CPU aggregate use an
operational deadline of 2026-07-31 09:30 Europe/Berlin and are restricted to
`hai003,hai007,hai008`. First-half jobs and their already-submitted rendezvous
barrier are not modified. The second-half per-job time limits must additionally
be chosen after first-half completion so that the worst-case cumulative GPU
allocation remains at or below 4.0 GPU-hours.

The four first-half jobs subsequently consumed 6,400 GPU-seconds in total
(37:33, 21:32, 27:54, and 19:41). This leaves 8,000 seconds under the 14,400
GPU-second cap. Each of the four second-half jobs is therefore limited to
33:00 (1,980 seconds): the worst-case cumulative allocation is 14,320 seconds
(3:58:40), leaving an 80-second fail-closed margin.
