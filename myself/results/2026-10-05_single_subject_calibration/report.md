# Offline single-participant calibration suite

All eligible data in preview recording episode_000020, participant wmh, right hand.
Fixed evaluation supports from user/gesture/both splits; EMG-only evaluation.
Fitting and evaluation input support unions are disjoint despite interleaved intervals.
Overlapping windows within pools are allowed and are not independent replications.

| Condition | Supervision / pool | Seed | MAE (deg) | Δ baseline (deg) | Fit centers | Parameters | Fit seconds | Peak GPU MiB |
|---|---|---|---:|---:|---:|---:|---:|---:|
| M0 | none / none | None | 15.349226 | +0.000000 | 0 | 0 | 0.000 | 0.00 |
| M1 | reference / T_all | None | 15.338320 | -0.010906 | 4174 | 20 | 0.001 | 0.00 |
| M2 | reference / T_all | None | 15.340363 | -0.008863 | 4174 | 40 | 0.003 | 0.00 |
| M3_mean | reference / T_all | 42/43/44 | 15.638044 ± 0.021868 | +0.288818 | 4174 | 10260 | 1.264 | 65.72 |
| M4 | RGB / T_rgb | None | 15.338340 | -0.010886 | 4174 | 20 | 0.001 | 0.00 |
| M5 | RGB / T_rgb | None | 15.332166 | -0.017060 | 4174 | 40 | 0.002 | 0.00 |
| M6_mean | RGB / T_rgb | 42/43/44 | 15.484483 ± 0.013651 | +0.135257 | 4174 | 10260 | 1.271 | 65.72 |

## Every head seed

| Condition | Supervision / pool | Seed | MAE (deg) | Δ baseline (deg) | Fit centers | Parameters | Fit seconds | Peak GPU MiB |
|---|---|---|---:|---:|---:|---:|---:|---:|
| M3_seed42 | reference / T_all | 42 | 15.627354 | +0.278128 | 4174 | 10260 | 1.344 | 65.72 |
| M3_seed43 | reference / T_all | 43 | 15.663200 | +0.313974 | 4174 | 10260 | 1.228 | 65.72 |
| M3_seed44 | reference / T_all | 44 | 15.623576 | +0.274350 | 4174 | 10260 | 1.219 | 65.72 |
| M6_seed42 | RGB / T_rgb | 42 | 15.493918 | +0.144692 | 4174 | 10260 | 1.263 | 65.72 |
| M6_seed43 | RGB / T_rgb | 43 | 15.490701 | +0.141475 | 4174 | 10260 | 1.234 | 65.72 |
| M6_seed44 | RGB / T_rgb | 44 | 15.468830 | +0.119604 | 4174 | 10260 | 1.316 | 65.72 |

## Matched supervision

| Condition | Supervision / pool | Seed | MAE (deg) | Δ baseline (deg) | Fit centers | Parameters | Fit seconds | Peak GPU MiB |
|---|---|---|---:|---:|---:|---:|---:|---:|
| M0 | none / none | None | 15.349226 | +0.000000 | 0 | 0 | 0.000 | 0.00 |
| M1 | reference / T_all | None | 15.338320 | -0.010906 | 4174 | 20 | 0.001 | 0.00 |
| M2 | reference / T_all | None | 15.340363 | -0.008863 | 4174 | 40 | 0.003 | 0.00 |
| M3_mean | reference / T_all | 42/43/44 | 15.638044 ± 0.021868 | +0.288818 | 4174 | 10260 | 1.264 | 65.72 |
| M4 | RGB / T_rgb | None | 15.338340 | -0.010886 | 4174 | 20 | 0.001 | 0.00 |
| M5 | RGB / T_rgb | None | 15.332166 | -0.017060 | 4174 | 40 | 0.002 | 0.00 |
| M6_mean | RGB / T_rgb | 42/43/44 | 15.484483 ± 0.013651 | +0.135257 | 4174 | 10260 | 1.271 | 65.72 |

## Coverage and evidence

T_all: 4174 centers, 417.400 nominal center seconds, 476.900 support union seconds.

E: 3654 centers, 365.400 nominal center seconds, 489.300 support union seconds.

Wrist MAE: 12.704364 degrees, identical across every condition.

EMG extraction: 39.772s; peak allocated GPU 149.88 MiB. Original head/cache maximum absolute discrepancy 3.58e-07 radians, tolerance atol=1e-5 rtol=1e-4.

RGB: 4174/4174 centers; 4174 unique frames. Teacher MAE 1.7607952151997557 degrees. Construction 11.056s; peak allocated GPU 145.73 MiB.

GPU peaks are torch.cuda.max_memory_allocated after reset per stage/seed, including resident model/parameters and transient tensors; allocator reserved memory, driver allocations, CPU caches and other processes are excluded. Shared extraction includes model construction/read/load; per-method fitting includes optimization or analytic solve, without shared extraction.

Full interval coverage: coverage.json and rgb_interval_coverage.csv. Raw IDs, active flags and vocabulary names: gesture_coverage.csv and rgb_gesture_coverage.csv. Reference ranges/variance: reference_variation.csv. All scores can be recomputed from evaluation_predictions.npz without data/models. Train residuals/teacher quality are diagnostics and did not choose parameters. Head final epoch 50 for each seed; sample seed SD uses ddof=1 and is not a population confidence interval.

Released whole-recording FFT filtering is noncausal; train intervals may occur after evaluated intervals. Checkpoint exposure of this participant remains unresolved. This is an offline within-participant engineering study in one preview recording; it does not establish unseen-user generalization, short-calibration retention, causal deployment, or REACT reproduction.

Blocked conditions: {}

# Interval and gesture coverage

All positions are local half-open row intervals. T_all and T_rgb are fitting pools; E is fixed evaluation.

| Interval | Split | Rows [a,b) | Candidate | Eligible | RGB | Exclusions |
|---:|---|---|---:|---:|---:|---|
| 0 | user | [0,97375) | 427 | 427 | — | {} |
| 1 | both | [97375,120745) | 57 | 57 | — | {} |
| 2 | user | [120745,140220) | 38 | 38 | — | {} |
| 3 | train | [140220,296020) | 720 | 720 | 720 | {} |
| 4 | gesture | [296020,323285) | 77 | 77 | — | {} |
| 5 | train | [323285,416765) | 408 | 408 | 408 | {} |
| 6 | gesture | [416765,440135) | 57 | 57 | — | {} |
| 7 | train | [440135,498560) | 233 | 233 | 233 | {} |
| 8 | gesture | [498560,521930) | 57 | 57 | — | {} |
| 9 | train | [521930,595935) | 311 | 311 | 311 | {} |
| 10 | gesture | [595935,623200) | 77 | 77 | — | {} |
| 11 | train | [623200,716680) | 408 | 408 | 408 | {} |
| 12 | user | [716680,1016595) | 1440 | 1440 | — | {} |
| 13 | both | [1016595,1039965) | 57 | 57 | — | {} |
| 14 | user | [1039965,1078915) | 135 | 135 | — | {} |
| 15 | both | [1078915,1102285) | 57 | 57 | — | {} |
| 16 | user | [1102285,1117865) | 18 | 18 | — | {} |
| 17 | both | [1117865,1141235) | 57 | 57 | — | {} |
| 18 | user | [1141235,1277560) | 622 | 622 | — | {} |
| 19 | both | [1277560,1300930) | 57 | 57 | — | {} |
| 20 | user | [1300930,1320405) | 38 | 38 | — | {} |
| 21 | train | [1320405,1476205) | 720 | 720 | 720 | {} |
| 22 | gesture | [1476205,1503470) | 77 | 77 | — | {} |
| 23 | train | [1503470,1596950) | 408 | 408 | 408 | {} |
| 24 | gesture | [1596950,1620320) | 57 | 57 | — | {} |
| 25 | train | [1620320,1678745) | 233 | 228 | 228 | {"invalid_center_supervision": 5} |
| 26 | gesture | [1678745,1702115) | 57 | 57 | — | {} |
| 27 | train | [1702115,1780015) | 330 | 330 | 330 | {} |
| 28 | gesture | [1780015,1803385) | 57 | 57 | — | {} |
| 29 | train | [1803385,1896865) | 408 | 408 | 408 | {} |
| 30 | user | [1896865,1935815) | 135 | 135 | — | {} |
| 31 | train | [1935815,1935926) | 0 | 0 | 0 | insufficient length |

| Raw ID | Name | Active | T_all | T_rgb | E |
|---:|---|---|---:|---:|---:|
| -1 | inactive | False | 4 | 4 | 68 |
| 0 | ASL1 | True | 0 | 0 | 173 |
| 1 | ASL2 | True | 0 | 0 | 148 |
| 2 | ASL3 | True | 0 | 0 | 100 |
| 3 | ASL4 | True | 0 | 0 | 100 |
| 4 | ASL5 | True | 0 | 0 | 100 |
| 5 | ASL6 | True | 0 | 0 | 100 |
| 6 | ASL7 | True | 0 | 0 | 100 |
| 7 | ASL8 | True | 0 | 0 | 100 |
| 8 | ASL9 | True | 0 | 0 | 100 |
| 9 | Claw3 | True | 0 | 0 | 100 |
| 10 | Claw5 | True | 0 | 0 | 100 |
| 11 | FreeAction | True | 0 | 0 | 100 |
| 12 | ILY | True | 0 | 0 | 100 |
| 13 | IndexBow | True | 0 | 0 | 100 |
| 14 | IndexMiddleClaw | True | 0 | 0 | 54 |
| 15 | JoystickCircle | True | 0 | 0 | 57 |
| 16 | JoystickSlide | True | 0 | 0 | 70 |
| 17 | MiddleBow | True | 0 | 0 | 65 |
| 18 | Nine | True | 0 | 0 | 57 |
| 19 | PalmYaw | True | 0 | 0 | 18 |
| 20 | PinchMiddle | True | 0 | 0 | 57 |
| 21 | PinkyBow | True | 0 | 0 | 64 |
| 22 | Rest | True | 0 | 0 | 100 |
| 23 | RingAndThumb | True | 0 | 0 | 100 |
| 24 | RingBow | True | 0 | 0 | 200 |
| 25 | Rock | True | 0 | 0 | 200 |
| 26 | Thumb | True | 0 | 0 | 200 |
| 27 | nocontact_disperse_palm | True | 0 | 0 | 117 |
| 28 | nocontact_free | True | 0 | 0 | 114 |
| 29 | nocontact_grab | True | 0 | 0 | 76 |
| 30 | Clap | True | 200 | 200 | 0 |
| 31 | CrossHand | True | 200 | 200 | 0 |
| 32 | CrossStretch | True | 200 | 200 | 0 |
| 33 | FingerPullLeft | True | 200 | 200 | 0 |
| 34 | FingerPullRight | True | 200 | 200 | 0 |
| 35 | FingerTipTouch | True | 104 | 104 | 0 |
| 36 | FistBump | True | 0 | 0 | 154 |
| 37 | Gaming | True | 105 | 105 | 0 |
| 38 | HandClasp | True | 200 | 200 | 0 |
| 39 | HandRub | True | 200 | 200 | 0 |
| 40 | IndexTapping | True | 200 | 200 | 0 |
| 41 | PalmRoll | True | 129 | 129 | 0 |
| 42 | PalmStack | True | 0 | 0 | 114 |
| 43 | PinkyHook | True | 200 | 200 | 0 |
| 44 | Prayer | True | 121 | 121 | 0 |
| 45 | Squeeze | True | 0 | 0 | 134 |
| 46 | SymOpen | True | 107 | 107 | 0 |
| 47 | SymSwing | True | 200 | 200 | 0 |
| 48 | ThumbWrestle | True | 200 | 200 | 0 |
| 49 | Typing | True | 108 | 108 | 0 |
| 51 | Kiss | True | 110 | 110 | 0 |
| 52 | MiddleOppo | True | 0 | 0 | 114 |
| 53 | Beijing | True | 136 | 136 | 0 |
| 54 | Checky | True | 200 | 200 | 0 |
| 55 | PairClaw | True | 133 | 133 | 0 |
| 56 | PairOK | True | 199 | 199 | 0 |
| 57 | Picture | True | 119 | 119 | 0 |
| 58 | PinchWring | True | 199 | 199 | 0 |
| 59 | ThumbOppo | True | 200 | 200 | 0 |
