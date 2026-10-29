# EMG innovation evidence snapshot

[English report](report.md) · [Personal research](../../README.md) · [Workflow](../../docs/emg_innovation_suite.md)

Completed offline Preview right-hand study, fitted only on wmh train. zbk/wsj
are fixed review recordings, not unknown-participant benchmarks. The selected
family was locked before formal evaluation errors. All tables/figures here are
newly executed results, not archival tests. Batch128 was explicitly authorized
by the user after hardware profiling; the original 30-epoch/lr protocol remains.

Valid execution: `myself/outputs/2026-10-29_emg_innovation_full_02`.
Execution checkout: `d7f55cfd30a93807641d93190f93e2ce240f71fa` on `main`.
Execution identity: `c49b1625007ade78fc323ec0bcca509e1357c01d417aaf0966626a94778c69e0`.

The interrupted full_01 and smoke_01 failure are preserved separately and are
not combined with this result. Supplied checkpoints and inputs were read-only.
Batch features were never written to disk. Full predictions, optimizer states,
weights, orders, logs and source copies remain in ignored outputs and are
intentionally omitted. No full logs, input copies or per-window features are
included in this snapshot.

The manifest maps every exported evidence file to its repository-relative
source and SHA-256. Export date and current checkout commit are distinct from
execution identity. The sources are local until separately committed/pushed;
this task did neither. Prediction-level reproduction needs the original data,
checkpoints and full output directory; compact tables and figures are readable
on GitHub independently of outputs. See workflow for smoke/full/resume/verify/export.
