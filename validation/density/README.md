# Force-level atmospheric-density validation

This benchmark compares Brahe NRLMSISE-00 predictions directly with TU Delft
version-02c GRACE-FO accelerometer-derived neutral-density observations. The
source products are licensed CC BY 4.0 and are stored outside the repository;
the immutable manifest records their SHA-256 checksums.
NRLMSISE-00 is driven by the CSSI space-weather file committed in the immutable
Brahe v1.7.0 source revision, also checksum-locked by the manifest. The benchmark
does not use Brahe's refreshable live cache.

The protocol was frozen before evaluation. April 1–3 trains one static scale,
April 15–16 and the May 10–11 Gannon storm select the random-walk process noise,
and October 10–12 is untouched test data. Samples are reduced from 10 seconds
to five minutes. The online prediction is scored before the observation at the
same epoch is assimilated.

```bash
python validation/density/fetch.py /tmp/gracefo-density
python validation/density/benchmark.py --data /tmp/gracefo-density
```

The estimated state remains an effective density/ballistic scale. Although the
reference density is accelerometer-derived using detailed TU Delft aerodynamic
and radiation-pressure models, its uncertainty still includes those retrieval
assumptions. The benchmark therefore reports innovation consistency as well as
MAPE and does not describe the observations as error-free truth.

Dataset: [TU Delft Thermosphere Data](https://thermosphere.tudelft.nl/).
