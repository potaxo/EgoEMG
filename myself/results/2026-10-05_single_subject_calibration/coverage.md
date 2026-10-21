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
