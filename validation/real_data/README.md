# Real-data drag validation

The first benchmark uses Sentinel-1A Precise Orbit Ephemerides from the Copernicus
POD Service, mirrored in the public NASA Alaska Satellite Facility `s1-orbits`
AWS Open Data bucket. Products contain 10-second Earth-fixed states, carry a
5 cm 3D RMS accuracy requirement, and are typically accurate below 1 cm.

The immutable checksummed manifest prevents silent dataset changes. Quiet
April 2024 arcs are split chronologically into train and validation sets. The
10–11 May Gannon storm is validation data because it was observed during
estimator development. The later 10–12 October 2024 storm was frozen before
its first benchmark run and is the untouched final test set.

```bash
python validation/real_data/fetch.py /tmp/sentinel1-pod
python validation/real_data/benchmark.py --data /tmp/sentinel1-pod
```

An independent, lower-altitude benchmark uses ESA Swarm-A reduced-dynamic SP3
orbits. Its manifest was frozen before evaluation and uses separate April quiet
training/validation, May storm validation, and October storm test arcs:

```bash
python validation/real_data/fetch.py /tmp/swarm-pod \
  --manifest validation/real_data/swarm_manifest.yaml
python validation/real_data/benchmark.py --data /tmp/swarm-pod \
  --manifest validation/real_data/swarm_manifest.yaml \
  --protocol validation/real_data/swarm_protocol.yaml \
  --output validation/real_data/swarm_results
```

The benchmark compares a documented nominal spacecraft assumption against one
effective drag-area parameter fitted exclusively on training arcs. A robust
median of per-arc linearized estimates prevents an orbit-maintenance maneuver
from dominating the fit. The result must improve the quiet validation RMS by
at least 5% and may not degrade mean storm-test RMS by more than 10%.

The fitted value must not be interpreted as physical area: drag observations
identify a combination of atmospheric density, projected area, drag
coefficient, attitude and other modelling errors. The current result motivates
estimating a time-varying density/ballistic scale with calibrated uncertainty.

That warning is especially important across missions. Sentinel-1A and Swarm-A
have different shapes, attitudes, masses, and altitudes; their fitted scale
factors are not a universal correction to atmospheric density. The reported
effective `CdA/m` is the observable combination under the stated constant-area
assumption. Separating density from spacecraft aerodynamics requires attitude
and geometry (or accelerometer) data.

Dataset provenance: [ASF Sentinel-1 POD Open Data](https://registry.opendata.aws/s1-orbits/).
Swarm provenance: [ESA Swarm SP3 COM product catalogue](https://swarm-disc.github.io/product-catalogue-tools/SW_SP3xCOM_2_.html).

## Atmosphere ablation

The same frozen Swarm arcs can be evaluated with every atmosphere currently
provided by the validated Brahe backend. The exponential case is a deliberately
simple single-scale-height control anchored at 400 km, not a modern empirical
climatology:

```bash
python validation/real_data/compare_atmospheres.py \
  --data /tmp/swarm-pod \
  --manifest validation/real_data/swarm_manifest.yaml \
  --protocol validation/real_data/swarm_protocol.yaml \
  --output validation/real_data/swarm_atmosphere_comparison
```

This is an apples-to-apples force-model ablation, but it still refits one
effective ballistic parameter per atmosphere. It ranks predictive orbit
residuals under that protocol; it does not independently identify true density.
JB2008 and DTM2020 remain future adapters and must not be claimed until their
implementations and required indices are versioned and validated.

The committed comparisons intentionally retain a mixed cross-mission result.
NRLMSISE-00 is decisively best on both frozen Swarm evaluation splits. For
Sentinel-1A it is the only model to pass the existing acceptance gate and has
the lowest untouched storm-test RMS, while the simpler controls have slightly
lower absolute validation RMS. Therefore the evidence supports keeping
NRLMSISE-00 as the default; it does not support claiming one universal winner
for every altitude, regime, and scoring rule.
