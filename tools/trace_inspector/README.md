# Static trace inspector

This is a dependency-free human inspection surface for the sealed mouse-movement,
typing, and closed-loop evaluation artifacts produced by labctl. It shows run-level
metrics, cross-cutting filters, raw and parsed actions, gold actions, outcomes,
typing diffs/events, step navigation, and mouse geometry overlays.

The generator fails closed. Before a run is indexed it checks:

- the artifact `.meta.json` identity, owner, alias, and producer run;
- a complete/valid result manifest equal to the registered result payload;
- every declared rows, report, nested generation, teacher-forced, and chunk hash;
- the labctl context run/recipe/output path and exactly one submitted Slurm job ID;
- referenced screenshot containment and existence.

On any mismatch it writes an error-only index that the browser renders prominently,
then exits non-zero. It never publishes a partial collection.

## Regenerate and open

On the cluster, one command creates a ready-to-open bundle with compact JSON and
relative symlinks to sealed screenshots, then serves it locally:

```bash
python tools/trace_inspector/generate_index.py \
  --output-dir /fast/home/franz.srambical/tmp/relative_mouse_trace_inspector \
  --serve --port 8765
```

Open <http://127.0.0.1:8765/>. If port forwarding is needed, forward local port
8765 to port 8765 on the login host. The output folder contains `index.html`,
`app.js`, `styles.css`, `data/index.json`, and relative asset symlinks; no image is
copied into Git or into the generated bundle.

Run the tests with:

```bash
python -m unittest discover -s tools/trace_inspector/tests -v
```

`source_rules.json` pins the currently completed eval families and includes
zero-minimum discovery rules for the compact-scale and typing-prose eval aliases.
A refresh will include those later results once their artifacts expose one of the
audited supported schemas; an unsupported or half-written match is a visible error.
Known superseded pre-final directories are listed as exact, reasoned exclusions in
that file and surfaced in the generated index; they are never skipped implicitly.
