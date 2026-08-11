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
