# Independent dynamics validation

This harness compares the Python research engine (Brahe) against Orekit 13.1.7
in a common GCRF Cartesian state representation. The initial validation profiles
are deliberately model-matched:

- `two_body`: identical Earth gravitational parameter
- `gravity`: identical ICGEM coefficient file and 20x20 truncation, with each
  engine independently implementing integration, spherical harmonics, Earth
  orientation, and frame transformations
- `third_body`: point Earth plus DE440 Sun and Moon point-mass attraction
- `srp_sunlight`: point Earth plus a 1 m², Cr=1.5 isotropic spacecraft and
  100 kg mass, with occultation disabled to isolate radiation acceleration
- `srp`: the same spacecraft with conical Earth eclipse enabled

Five orbit regimes and 1/3/7-day horizons are defined in `scenarios.yaml`.
Never interpret a close comparison as proof that both engines are correct; it
is one line of evidence alongside analytic tests and comparisons to measured
precise-orbit ephemerides.

The runner requires Java 21, the shaded Orekit reference JAR, an Orekit data
directory, and an ICGEM `.gfc` gravity file. These external datasets and
toolchains are intentionally not committed. Bootstrap the pinned toolchain and
datasets into a disposable absolute directory (about 500 MB):

```bash
validation/bootstrap.sh /tmp/orbit-validation-tools
```

```bash
python validation/run.py \
  --java /path/to/java \
  --orekit-jar validation/orekit/target/orekit-reference-1.0.0.jar \
  --orekit-data /path/to/orekit-data \
  --gravity-file /path/to/eigen-6s.gfc
```

Use `--quick` for the ISS-like one-day smoke matrix. Results are written as
both `validation/results/results.json` and `validation/results/report.md`.
The committed matrix uses a one-metre limit for gravitational profiles, five
metres for continuous-sunlight SRP, and an explicit 25-metre limit for conical
SRP. Small ephemeris/radiation-direction differences accumulate at MEO/GEO;
independent limb/event implementations also differ near eclipse transitions.
The report records both limitations. Regenerate the matrix whenever a dynamics
dependency or configuration changes.
