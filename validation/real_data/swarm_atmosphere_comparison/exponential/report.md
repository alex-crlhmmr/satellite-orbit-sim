# Swarm-A reduced-dynamic precise orbits drag validation

Evidence gate: **PASS**

Fitted effective drag area: 1.6672 m²

Equivalent density/ballistic scale versus the nominal baseline: 1.6672

Fitted effective CdA/m: 0.007837 m²/kg

> This is an effective drag scale, not a recovered physical surface area. It absorbs density, attitude, coefficient, and unmodelled-force errors.

| Arc | Split | Regime | Model | RMS 3D | RMS along-track | Final 3D |
|---|---|---|---:|---:|---:|---:|
| SW_OPER_SP3ACOM_2__20240331T235942_20240401T235942_0201.ZIP | train | quiet | nominal | 726.806 m | 726.610 m | 1555.930 m |
| SW_OPER_SP3ACOM_2__20240331T235942_20240401T235942_0201.ZIP | train | quiet | fitted | 252.559 m | 252.257 m | 501.928 m |
| SW_OPER_SP3ACOM_2__20240401T235942_20240402T235942_0201.ZIP | train | quiet | nominal | 473.508 m | 473.297 m | 1091.346 m |
| SW_OPER_SP3ACOM_2__20240401T235942_20240402T235942_0201.ZIP | train | quiet | fitted | 32.027 m | 30.171 m | 49.488 m |
| SW_OPER_SP3ACOM_2__20240402T235942_20240403T235942_0201.ZIP | train | quiet | nominal | 414.305 m | 414.148 m | 853.610 m |
| SW_OPER_SP3ACOM_2__20240402T235942_20240403T235942_0201.ZIP | train | quiet | fitted | 68.570 m | 67.897 m | 201.610 m |
| SW_OPER_SP3ACOM_2__20240415T235942_20240416T235942_0201.ZIP | validation | quiet | nominal | 1604.871 m | 1604.516 m | 3686.784 m |
| SW_OPER_SP3ACOM_2__20240415T235942_20240416T235942_0201.ZIP | validation | quiet | fitted | 1127.741 m | 1127.436 m | 2626.404 m |
| SW_OPER_SP3ACOM_2__20240510T235942_20240511T235942_0201.ZIP | validation | storm | nominal | 5374.197 m | 5373.478 m | 11452.880 m |
| SW_OPER_SP3ACOM_2__20240510T235942_20240511T235942_0201.ZIP | validation | storm | fitted | 4885.096 m | 4884.446 m | 10367.976 m |
| SW_OPER_SP3ACOM_2__20241010T235942_20241011T235942_0201.ZIP | test | storm | nominal | 5063.028 m | 5062.400 m | 10551.131 m |
| SW_OPER_SP3ACOM_2__20241010T235942_20241011T235942_0201.ZIP | test | storm | fitted | 4474.953 m | 4474.406 m | 9241.340 m |

Parameters are fitted only on `train`; validation and storm test arcs are locked.
