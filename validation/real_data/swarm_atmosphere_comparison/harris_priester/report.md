# Swarm-A reduced-dynamic precise orbits drag validation

Evidence gate: **PASS**

Fitted effective drag area: 1.9245 m²

Equivalent density/ballistic scale versus the nominal baseline: 1.9245

Fitted effective CdA/m: 0.009047 m²/kg

> This is an effective drag scale, not a recovered physical surface area. It absorbs density, attitude, coefficient, and unmodelled-force errors.

| Arc | Split | Regime | Model | RMS 3D | RMS along-track | Final 3D |
|---|---|---|---:|---:|---:|---:|
| SW_OPER_SP3ACOM_2__20240331T235942_20240401T235942_0201.ZIP | train | quiet | nominal | 825.664 m | 825.488 m | 1781.898 m |
| SW_OPER_SP3ACOM_2__20240331T235942_20240401T235942_0201.ZIP | train | quiet | fitted | 259.250 m | 259.087 m | 530.263 m |
| SW_OPER_SP3ACOM_2__20240401T235942_20240402T235942_0201.ZIP | train | quiet | nominal | 567.807 m | 567.616 m | 1294.035 m |
| SW_OPER_SP3ACOM_2__20240401T235942_20240402T235942_0201.ZIP | train | quiet | fitted | 30.110 m | 28.511 m | 36.219 m |
| SW_OPER_SP3ACOM_2__20240402T235942_20240403T235942_0201.ZIP | train | quiet | nominal | 503.647 m | 503.480 m | 1051.005 m |
| SW_OPER_SP3ACOM_2__20240402T235942_20240403T235942_0201.ZIP | train | quiet | fitted | 78.623 m | 77.828 m | 228.849 m |
| SW_OPER_SP3ACOM_2__20240415T235942_20240416T235942_0201.ZIP | validation | quiet | nominal | 1756.826 m | 1756.451 m | 4019.462 m |
| SW_OPER_SP3ACOM_2__20240415T235942_20240416T235942_0201.ZIP | validation | quiet | fitted | 1236.044 m | 1235.725 m | 2857.709 m |
| SW_OPER_SP3ACOM_2__20240510T235942_20240511T235942_0201.ZIP | validation | storm | nominal | 5532.544 m | 5531.802 m | 11809.088 m |
| SW_OPER_SP3ACOM_2__20240510T235942_20240511T235942_0201.ZIP | validation | storm | fitted | 5001.188 m | 5000.519 m | 10635.119 m |
| SW_OPER_SP3ACOM_2__20241010T235942_20241011T235942_0201.ZIP | test | storm | nominal | 5214.686 m | 5214.039 m | 10893.575 m |
| SW_OPER_SP3ACOM_2__20241010T235942_20241011T235942_0201.ZIP | test | storm | fitted | 4539.931 m | 4539.378 m | 9395.270 m |

Parameters are fitted only on `train`; validation and storm test arcs are locked.
