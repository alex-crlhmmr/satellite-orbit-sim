# Sentinel-1A drag validation

Evidence gate: **PASS**

Fitted effective drag area: 2.2743 m²

Equivalent density/ballistic scale versus the 10 m² nominal baseline: 0.2274

> This is an effective drag scale, not a recovered physical surface area. It absorbs density, attitude, coefficient, and unmodelled-force errors.

| Arc | Split | Regime | Model | RMS 3D | RMS along-track | Final 3D |
|---|---|---|---:|---:|---:|---:|
| 20240401 | train | quiet | nominal | 35.645 m | 35.187 m | 87.302 m |
| 20240401 | train | quiet | fitted | 93.235 m | 93.037 m | 188.701 m |
| 20240402 | train | quiet | nominal | 268.229 m | 268.166 m | 544.682 m |
| 20240402 | train | quiet | fitted | 160.321 m | 160.258 m | 300.821 m |
| 20240403 | train | quiet | nominal | 104.803 m | 104.686 m | 186.221 m |
| 20240403 | train | quiet | fitted | 25.140 m | 24.622 m | 81.579 m |
| 20240404 | train | quiet | nominal | 153.966 m | 153.786 m | 350.040 m |
| 20240404 | train | quiet | fitted | 32.152 m | 31.354 m | 76.543 m |
| 20240405 | train | quiet | nominal | 976.109 m | 976.010 m | 1753.699 m |
| 20240405 | train | quiet | fitted | 849.432 m | 849.351 m | 1460.472 m |
| 20240406 | train | quiet | nominal | 76.394 m | 76.226 m | 210.464 m |
| 20240406 | train | quiet | fitted | 58.453 m | 58.229 m | 76.522 m |
| 20240407 | train | quiet | nominal | 151.369 m | 151.307 m | 331.312 m |
| 20240407 | train | quiet | fitted | 23.464 m | 23.235 m | 34.742 m |
| 20240408 | train | quiet | nominal | 3253.184 m | 3252.105 m | 6982.003 m |
| 20240408 | train | quiet | fitted | 3394.574 m | 3393.485 m | 7300.024 m |
| 20240415 | validation | quiet | nominal | 238.771 m | 238.695 m | 491.011 m |
| 20240415 | validation | quiet | fitted | 87.593 m | 87.422 m | 231.656 m |
| 20240509 | validation | storm | nominal | 562.512 m | 562.404 m | 1207.142 m |
| 20240509 | validation | storm | fitted | 42.373 m | 42.072 m | 34.207 m |
| 20240510 | validation | storm | nominal | 1463.561 m | 1463.352 m | 2945.874 m |
| 20240510 | validation | storm | fitted | 605.747 m | 605.651 m | 998.944 m |
| 20241009 | test | storm | nominal | 772.084 m | 771.940 m | 1701.885 m |
| 20241009 | test | storm | fitted | 62.187 m | 62.113 m | 101.126 m |
| 20241010 | test | storm | nominal | 431.631 m | 431.501 m | 1042.280 m |
| 20241010 | test | storm | fitted | 351.728 m | 351.646 m | 682.602 m |
| 20241011 | test | storm | nominal | 3237.331 m | 3236.980 m | 6102.294 m |
| 20241011 | test | storm | fitted | 2721.243 m | 2720.970 m | 4950.697 m |

Parameters are fitted only on `train`; validation and storm test arcs are locked.
