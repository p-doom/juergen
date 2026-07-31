# Pre-PhaseB real-VM short-task pass@k launch bundle

This integration binds the approved c603 curriculum, approved fail-closed
paired evaluator, and reviewed executor successor without accessing held-out tasks.
The five task-affine shard recipes each reserve two H100s (one model server per complete
system arm), 16 CPUs, and one KVM-capable Slurm task. After both servers are
healthy the parent process clears `CUDA_VISIBLE_DEVICES` before creating the
VM, so executor isolation remains CPU/KVM-only while the already-started model
children retain their allocated GPUs.

This is a complete-system parity comparison, not an interface-only causal
effect. The native-absolute arm pins the absraw-pre r32 model consumed by TF
job 136209; the compact-relative arm pins the A-to-B r256/lr5e-5 model consumed
by TF job 136207. Both use the executor's native PyAutoGUI-A click backend so
release mechanics are held constant. `config/checkpoint_identity.json` records
the immutable artifact, manifest, and weight hashes.

The committed evaluation manifest remains blocked on the reviewed successor's
`EXECUTOR_READY.json` artifact ID/hash and the c603 task
setup-validation artifact ID/hash are explicit `UNRESOLVED_PIN:`/zero-SHA
placeholders. Replace those only with independently registered artifacts, then
recompute `evaluation_manifest_payload_sha256` and re-run all five plan checks. The
offline audit performs no submission and reports every remaining placeholder.

Exactly eight attempts per cell support pass@1, pass@4, and pass@8 from one
preregistered trial set. A task and all its cells/attempts/arms stay together
on one deterministic shard. `config/shard_inventory.json` pins the five task
assignments, trial counts, and deterministic plan-payload hashes. The aggregate
recipe is CPU-only and consumes all
five complete shard artifacts; partial or typed invalid evaluations remain
fail-closed under the approved evaluator.
