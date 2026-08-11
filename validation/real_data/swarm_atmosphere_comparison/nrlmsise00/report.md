# Swarm-A reduced-dynamic precise orbits drag validation

Evidence gate: **PASS**

Fitted effective drag area: 1.7229 m²

Equivalent density/ballistic scale versus the nominal baseline: 1.7229

Fitted effective CdA/m: 0.008099 m²/kg

> This is an effective drag scale, not a recovered physical surface area. It absorbs density, attitude, coefficient, and unmodelled-force errors.

| Arc | Split | Regime | Model | RMS 3D | RMS along-track | Final 3D |
|---|---|---|---:|---:|---:|---:|
| SW_OPER_SP3ACOM_2__20240331T235942_20240401T235942_0201.ZIP | train | quiet | nominal | 619.644 m | 619.476 m | 1324.989 m |
| SW_OPER_SP3ACOM_2__20240331T235942_20240401T235942_0201.ZIP | train | quiet | fitted | 35.831 m | 34.638 m | 17.657 m |
| SW_OPER_SP3ACOM_2__20240401T235942_20240402T235942_0201.ZIP | train | quiet | nominal | 496.206 m | 496.012 m | 1137.905 m |
| SW_OPER_SP3ACOM_2__20240401T235942_20240402T235942_0201.ZIP | train | quiet | fitted | 29.729 m | 28.113 m | 41.938 m |
| SW_OPER_SP3ACOM_2__20240402T235942_20240403T235942_0201.ZIP | train | quiet | nominal | 471.152 m | 470.987 m | 979.044 m |
| SW_OPER_SP3ACOM_2__20240402T235942_20240403T235942_0201.ZIP | train | quiet | fitted | 30.391 m | 28.551 m | 73.748 m |
| SW_OPER_SP3ACOM_2__20240415T235942_20240416T235942_0201.ZIP | validation | quiet | nominal | 917.435 m | 917.141 m | 2153.359 m |
| SW_OPER_SP3ACOM_2__20240415T235942_20240416T235942_0201.ZIP | validation | quiet | fitted | 119.255 m | 118.414 m | 107.281 m |
| SW_OPER_SP3ACOM_2__20240510T235942_20240511T235942_0201.ZIP | validation | storm | nominal | 3231.328 m | 3230.911 m | 6704.338 m |
| SW_OPER_SP3ACOM_2__20240510T235942_20240511T235942_0201.ZIP | validation | storm | fitted | 1157.347 m | 1157.182 m | 2093.399 m |
| SW_OPER_SP3ACOM_2__20241010T235942_20241011T235942_0201.ZIP | test | storm | nominal | 2807.165 m | 2806.855 m | 5539.505 m |
| SW_OPER_SP3ACOM_2__20241010T235942_20241011T235942_0201.ZIP | test | storm | fitted | 576.434 m | 576.301 m | 495.037 m |

Parameters are fitted only on `train`; validation and storm test arcs are locked.
