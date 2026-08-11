# GRACE-FO force-level density validation

Evidence gate: **PASS**

Training-only static scale: `0.789619`

Validation-selected process-noise PSD: `1.00e-05 s⁻¹`

| Split | Raw MAPE | Static MAPE | Online one-step MAPE | Mean NIS |
|---|---:|---:|---:|---:|
| validation | 51.26% | 35.37% | 14.89% | 1.162 |
| test | 92.56% | 59.01% | 11.98% | 0.731 |

Online predictions are evaluated before assimilating the observation at that epoch.
Process noise is selected on validation only; October is untouched test data.
