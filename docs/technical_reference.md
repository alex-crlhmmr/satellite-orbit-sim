# High-Fidelity LEO Satellite Orbital Simulation

## Technical Reference

---

### 1. Overview

This project implements a high-fidelity low-Earth orbit (LEO) satellite propagator with a Gymnasium-compatible reinforcement learning environment, real-time 3D rendering, and TCP video/telemetry streaming. It is designed to run on NVIDIA Jetson Orin hardware with headless EGL GPU rendering.

**Core capabilities:**

- 6DOF orbital propagation with J2--J6 zonal harmonics, atmospheric drag, solar radiation pressure, and Sun/Moon third-body perturbations
- RK4 fixed-step integration in float64 precision
- Gymnasium RL environment with quasi-nonsingular relative orbital elements (ROE) as state
- Headless 3D rendering via moderngl/EGL with day/night Earth textures
- TCP streaming server for remote visualization

---

### 2. Architecture

```
orbital_simulation/
├── core/                    # Physics engine
│   ├── constants.py         # WGS-84, IAU constants
│   ├── frames.py            # Reference frame transforms
│   ├── elements.py          # Orbital element conversions
│   ├── gravity.py           # Zonal harmonic accelerations
│   ├── atmosphere.py        # Exponential drag model
│   ├── srp.py               # Solar radiation pressure
│   ├── third_body.py        # Sun/Moon perturbations
│   └── propagator.py        # RK4 integrator + force assembly
├── env/                     # Reinforcement learning
│   ├── orbital_env.py       # Gymnasium environment
│   └── rewards.py           # Reward functions
├── render/                  # Visualization
│   ├── renderer.py          # moderngl EGL headless renderer
│   ├── earth.py             # Textured Earth sphere
│   ├── camera.py            # View/projection matrices
│   └── shaders/             # GLSL vertex/fragment shaders
├── stream/                  # Network streaming
│   ├── server.py            # Asyncio TCP server
│   ├── client.py            # Receiver/display client
│   └── protocol.py          # Binary wire protocol
├── config/default.yaml      # Configuration
├── main.py                  # Entry point
├── viewer.py                # Remote viewer client
└── tests/test_core.py       # Validation suite
```

---

### 3. Coordinate Systems

**Earth-Centered Inertial (ECI, J2000):** The primary frame for propagation. Origin at Earth's center of mass, X-axis toward the vernal equinox at J2000.0 epoch, Z-axis along Earth's rotation axis (north), Y completing the right-handed triad.

**Earth-Centered Earth-Fixed (ECEF):** Co-rotates with Earth. Related to ECI by a rotation through the Greenwich Mean Sidereal Time (GMST):

$$
\mathbf{r}_{\text{ECEF}} = R_z(\theta_{\text{GMST}}) \, \mathbf{r}_{\text{ECI}}
$$

**Radial-Transverse-Normal (RTN):** Local orbital frame centered on the satellite.
- **R** (radial): along position vector
- **T** (transverse/along-track): perpendicular to R in the orbital plane, in the direction of motion
- **N** (normal/cross-track): along the angular momentum vector

$$
\hat{\mathbf{R}} = \frac{\mathbf{r}}{|\mathbf{r}|}, \quad
\hat{\mathbf{N}} = \frac{\mathbf{r} \times \mathbf{v}}{|\mathbf{r} \times \mathbf{v}|}, \quad
\hat{\mathbf{T}} = \hat{\mathbf{N}} \times \hat{\mathbf{R}}
$$

**Perifocal (PQW):** Orbit-plane frame with P toward periapsis, Q 90 degrees ahead in the orbit plane, W along the angular momentum. Used as an intermediate frame for Keplerian-to-Cartesian conversion.

---

### 4. Orbital Elements

#### 4.1 Classical Keplerian Elements

The state of a satellite is described by six classical orbital elements:

| Symbol | Element | Description |
|--------|---------|-------------|
| $a$ | Semi-major axis | Size of the orbit ellipse [m] |
| $e$ | Eccentricity | Shape of the ellipse (0 = circle, 1 = parabola) |
| $i$ | Inclination | Tilt of the orbital plane relative to the equator [rad] |
| $\Omega$ | RAAN | Right ascension of the ascending node [rad] |
| $\omega$ | Argument of periapsis | Orientation of the ellipse within the orbital plane [rad] |
| $\nu$ | True anomaly | Position of the satellite along the orbit [rad] |

#### 4.2 Cartesian to Keplerian Conversion

Given position $\mathbf{r}$ and velocity $\mathbf{v}$ in ECI:

**Angular momentum:**
$$\mathbf{h} = \mathbf{r} \times \mathbf{v}$$

**Node vector:**
$$\mathbf{n} = \hat{\mathbf{k}} \times \mathbf{h}$$

**Eccentricity vector:**
$$\mathbf{e} = \frac{1}{\mu}\left[\left(v^2 - \frac{\mu}{r}\right)\mathbf{r} - (\mathbf{r} \cdot \mathbf{v})\mathbf{v}\right]$$

**Semi-major axis:**
$$a = -\frac{\mu}{2\varepsilon}, \quad \varepsilon = \frac{v^2}{2} - \frac{\mu}{r}$$

**Inclination:**
$$i = \arccos\left(\frac{h_z}{|\mathbf{h}|}\right)$$

**RAAN:**
$$\Omega = \arccos\left(\frac{n_x}{|\mathbf{n}|}\right), \quad \text{if } n_y < 0 \text{ then } \Omega = 2\pi - \Omega$$

**Argument of periapsis:**
$$\omega = \arccos\left(\frac{\mathbf{n} \cdot \mathbf{e}}{|\mathbf{n}||\mathbf{e}|}\right), \quad \text{if } e_z < 0 \text{ then } \omega = 2\pi - \omega$$

**True anomaly:**
$$\nu = \arccos\left(\frac{\mathbf{e} \cdot \mathbf{r}}{|\mathbf{e}||\mathbf{r}|}\right), \quad \text{if } \mathbf{r} \cdot \mathbf{v} < 0 \text{ then } \nu = 2\pi - \nu$$

#### 4.3 Keplerian to Cartesian Conversion

Position and velocity in the perifocal frame:

$$
\mathbf{r}_{PQW} = \frac{p}{1 + e\cos\nu}
\begin{pmatrix} \cos\nu \\ \sin\nu \\ 0 \end{pmatrix}, \quad
\mathbf{v}_{PQW} = \sqrt{\frac{\mu}{p}}
\begin{pmatrix} -\sin\nu \\ e + \cos\nu \\ 0 \end{pmatrix}
$$

where $p = a(1 - e^2)$ is the semi-latus rectum. Transform to ECI via the 3-1-3 rotation matrix $R(\Omega, i, \omega)$.

#### 4.4 Anomaly Conversions

**True to eccentric anomaly:**
$$E = 2\arctan\left(\sqrt{\frac{1-e}{1+e}} \tan\frac{\nu}{2}\right)$$

**Eccentric to mean anomaly (Kepler's equation):**
$$M = E - e\sin E$$

**Mean to eccentric anomaly** — solved iteratively via Newton-Raphson:
$$E_{n+1} = E_n - \frac{E_n - e\sin E_n - M}{1 - e\cos E_n}$$

Convergence tolerance: $10^{-12}$ rad, maximum 50 iterations.

#### 4.5 Quasi-Nonsingular Relative Orbital Elements (ROE)

Following D'Amico's formulation, the relative state between a deputy and chief satellite is expressed as:

$$
\delta\boldsymbol{\alpha} = \begin{pmatrix} \delta a \\ \delta\lambda \\ \delta e_x \\ \delta e_y \\ \delta i_x \\ \delta i_y \end{pmatrix} = \begin{pmatrix} (a_d - a_c) / a_c \\ (u_d - u_c) + (\Omega_d - \Omega_c)\cos i_c \\ e_d\cos\omega_d - e_c\cos\omega_c \\ e_d\sin\omega_d - e_c\sin\omega_c \\ i_d - i_c \\ (\Omega_d - \Omega_c)\sin i_c \end{pmatrix}
$$

where $u = \omega + \nu$ is the argument of latitude. This parameterization is nonsingular for circular and near-equatorial orbits and varies slowly over time, making it suitable as an RL observation.

---

### 5. Force Models

The total acceleration on the satellite is:

$$
\mathbf{a}_{\text{total}} = \mathbf{a}_{\text{Kepler}} + \mathbf{a}_{J_n} + \mathbf{a}_{\text{drag}} + \mathbf{a}_{\text{SRP}} + \mathbf{a}_{\text{3rd body}} + \mathbf{a}_{\text{thrust}}
$$

Each perturbation can be independently enabled or disabled via configuration.

#### 5.1 Two-Body (Keplerian) Gravity

$$
\mathbf{a}_{\text{Kepler}} = -\frac{\mu}{r^3}\mathbf{r}
$$

where $\mu = GM_\oplus = 3.986004418 \times 10^{14}$ m$^3$/s$^2$ (WGS-84).

#### 5.2 Zonal Harmonics (J2--J6)

The Earth's gravitational potential includes zonal harmonic terms arising from the planet's oblateness. Accelerations are computed in closed-form Cartesian expressions (Vallado).

**J2 perturbation** (dominant term):

$$
\mathbf{a}_{J_2} = -\frac{3}{2} J_2 \frac{\mu R_\oplus^2}{r^5}
\begin{pmatrix}
x\left(1 - 5\frac{z^2}{r^2}\right) \\[4pt]
y\left(1 - 5\frac{z^2}{r^2}\right) \\[4pt]
z\left(3 - 5\frac{z^2}{r^2}\right)
\end{pmatrix}
$$

where $J_2 = 1.08263 \times 10^{-3}$ and $R_\oplus = 6{,}378{,}137$ m.

**J3 perturbation:**

$$
\mathbf{a}_{J_3} = -\frac{1}{2} J_3 \frac{\mu R_\oplus^3}{r^7}
\begin{pmatrix}
x\left(15z - 35\frac{z^3}{r^2}\right) \\[4pt]
y\left(15z - 35\frac{z^3}{r^2}\right) \\[4pt]
6r^2 - 45z^2 + 35\frac{z^4}{r^2}
\end{pmatrix}
$$

**J4 perturbation:**

$$
\mathbf{a}_{J_4} = \frac{5}{8} J_4 \frac{\mu R_\oplus^4}{r^9}
\begin{pmatrix}
x\left(3r^2 - 42z^2 + 63\frac{z^4}{r^2}\right) \\[4pt]
y\left(3r^2 - 42z^2 + 63\frac{z^4}{r^2}\right) \\[4pt]
z\left(15r^2 - 70z^2 + 63\frac{z^4}{r^2}\right)
\end{pmatrix}
$$

J5 and J6 follow analogous closed-form expressions with increasing polynomial order in $z/r$.

**Zonal harmonic constants (unnormalized):**

| Coefficient | Value |
|-------------|-------|
| $J_2$ | $1.08263 \times 10^{-3}$ |
| $J_3$ | $-2.53881 \times 10^{-6}$ |
| $J_4$ | $-1.61988 \times 10^{-6}$ |
| $J_5$ | $-2.27141 \times 10^{-7}$ |
| $J_6$ | $5.40788 \times 10^{-7}$ |

**Secular effects of J2** (analytical, used for validation):

$$
\dot{\Omega} = -\frac{3}{2} n J_2 \left(\frac{R_\oplus}{p}\right)^2 \cos i
$$

$$
\dot{\omega} = \frac{3}{2} n J_2 \left(\frac{R_\oplus}{p}\right)^2 \left(\frac{5}{2}\sin^2 i - 2\right)
$$

where $n = \sqrt{\mu/a^3}$ is the mean motion and $p = a(1-e^2)$.

#### 5.3 Atmospheric Drag

**Density model:** Exponential atmosphere with 28 altitude bands from the US Standard Atmosphere 1976. For altitude $h$ in a band with base altitude $h_0$, base density $\rho_0$, and scale height $H$:

$$
\rho(h) = \rho_0 \exp\left(-\frac{h - h_0}{H}\right)
$$

| Altitude [km] | $\rho_0$ [kg/m$^3$] | $H$ [km] |
|:-:|:-:|:-:|
| 0 | 1.225 | 7.249 |
| 100 | $5.297 \times 10^{-7}$ | 5.877 |
| 200 | $2.789 \times 10^{-10}$ | 37.105 |
| 400 | $3.725 \times 10^{-12}$ | 58.515 |
| 600 | $1.454 \times 10^{-13}$ | 71.835 |
| 800 | $1.170 \times 10^{-14}$ | 124.64 |
| 1000 | $3.019 \times 10^{-15}$ | 268.00 |

(Selected bands shown; the model uses all 28 bands.)

**Drag acceleration:**

$$
\mathbf{a}_{\text{drag}} = -\frac{1}{2} \rho \, C_D \frac{A}{m} \, |\mathbf{v}_{\text{rel}}| \, \mathbf{v}_{\text{rel}}
$$

where the velocity relative to the atmosphere accounts for Earth's rotation:

$$
\mathbf{v}_{\text{rel}} = \mathbf{v} - \boldsymbol{\omega}_\oplus \times \mathbf{r}, \quad \boldsymbol{\omega}_\oplus = (0, 0, 7.2921150 \times 10^{-5})^T \text{ rad/s}
$$

Default parameters: $C_D = 2.2$, $A/m = 0.01$ m$^2$/kg.

#### 5.4 Solar Radiation Pressure

**Solar ephemeris:** Low-precision algorithm from Meeus, "Astronomical Algorithms". Computes the Sun's geocentric ECI position from the Julian date using mean anomaly, equation of center, ecliptic longitude, and obliquity. Accuracy approximately 0.01 degrees.

**Cylindrical shadow model:** The satellite is in Earth's shadow when:
1. It is on the anti-Sun side of Earth (projection of $\mathbf{r}_{\text{sat}}$ onto $\hat{\mathbf{r}}_{\text{sun}}$ is negative), and
2. Its perpendicular distance from the Earth-Sun line is less than $R_\oplus$.

Shadow factor: $\nu_{\text{shadow}} \in \{0, 1\}$ (umbra or full sunlight).

**SRP acceleration:**

$$
\mathbf{a}_{\text{SRP}} = -P_\odot \left(\frac{AU}{|\mathbf{r}_\odot|}\right)^2 C_R \frac{A}{m} \, \hat{\mathbf{e}}_{\text{sat} \to \text{sun}} \cdot \nu_{\text{shadow}}
$$

where $P_\odot = 4.56 \times 10^{-6}$ N/m$^2$ is the solar radiation pressure at 1 AU, $C_R = 1.5$ is the reflectivity coefficient.

#### 5.5 Third-Body Perturbations (Sun and Moon)

**Lunar ephemeris:** Simplified Brown's theory evaluating the five fundamental arguments (mean longitude, mean anomaly, mean elongation, argument of latitude, solar mean anomaly) with the dominant periodic correction terms. Accuracy approximately 0.5 degrees.

**Third-body acceleration:**

$$
\mathbf{a}_{\text{3rd}} = \mu_{\text{body}} \left(\frac{\mathbf{r}_{\text{body}} - \mathbf{r}_{\text{sat}}}{|\mathbf{r}_{\text{body}} - \mathbf{r}_{\text{sat}}|^3} - \frac{\mathbf{r}_{\text{body}}}{|\mathbf{r}_{\text{body}}|^3}\right)
$$

Applied for both the Sun ($\mu_\odot = 1.327 \times 10^{20}$ m$^3$/s$^2$) and Moon ($\mu_\leftmoon = 4.903 \times 10^{12}$ m$^3$/s$^2$).

---

### 6. Numerical Integration

#### 6.1 RK4 Fixed-Step Integrator

The equations of motion are integrated using a classical fourth-order Runge-Kutta scheme. The state vector is $\mathbf{y} = (\mathbf{r}, \mathbf{v})^T \in \mathbb{R}^6$ with derivative:

$$
\dot{\mathbf{y}} = f(t, \mathbf{y}) = \begin{pmatrix} \mathbf{v} \\ \mathbf{a}_{\text{total}}(t, \mathbf{r}, \mathbf{v}) \end{pmatrix}
$$

**RK4 update:**

$$
\begin{aligned}
\mathbf{k}_1 &= f(t_n, \mathbf{y}_n) \\
\mathbf{k}_2 &= f\!\left(t_n + \tfrac{\Delta t}{2},\; \mathbf{y}_n + \tfrac{\Delta t}{2}\mathbf{k}_1\right) \\
\mathbf{k}_3 &= f\!\left(t_n + \tfrac{\Delta t}{2},\; \mathbf{y}_n + \tfrac{\Delta t}{2}\mathbf{k}_2\right) \\
\mathbf{k}_4 &= f(t_n + \Delta t,\; \mathbf{y}_n + \Delta t\,\mathbf{k}_3) \\[6pt]
\mathbf{y}_{n+1} &= \mathbf{y}_n + \frac{\Delta t}{6}(\mathbf{k}_1 + 2\mathbf{k}_2 + 2\mathbf{k}_3 + \mathbf{k}_4)
\end{aligned}
$$

Default timestep: $\Delta t = 10$ s, suitable for LEO where the orbital period is approximately 90 minutes. The method is chosen for its simplicity, numerical stability at this step size, and compatibility with GPU batch parallelism (no adaptive step rejection logic).

#### 6.2 Precision

All propagation is performed in **float64** (double precision). At LEO distances ($\sim$7000 km), float32 provides only $\sim$1 m position precision due to the limited mantissa, whereas float64 gives sub-micrometer precision. The state tensor dtype is `torch.float64` throughout the physics engine.

---

### 7. Reinforcement Learning Environment

#### 7.1 Gymnasium Interface

The environment follows the `gymnasium.Env` API and passes `gymnasium.utils.env_checker.check_env`.

**Observation space:** `Box(shape=(10,), dtype=float32)`

| Index | Component | Description |
|:-----:|-----------|-------------|
| 0--5 | $\delta\boldsymbol{\alpha}$ | Quasi-nonsingular ROE (6 elements) |
| 6 | Altitude | $(|\mathbf{r}| - R_\oplus) / a_{\text{ref}}$ (normalized) |
| 7 | Period | $T / T_{\text{ref}}$ (normalized) |
| 8 | Eclipse | 0.0 (sunlit) or 1.0 (shadow) |
| 9 | Time | $t / t_{\max}$ (fraction of episode elapsed) |

**Action space:** `Box(-1, 1, shape=(3,), dtype=float32)`

Actions represent continuous thrust commands in the RTN frame, scaled by $a_{\max} = F_{\max} / m$. The RTN-to-ECI rotation converts the commanded thrust to inertial frame before propagation.

**Step dynamics:** Each `env.step()` propagates $\Delta t_{\text{env}} = 60$ s of simulation time using multiple RK4 substeps of $\Delta t = 10$ s.

#### 7.2 Reward Functions

Three reward modes are provided:

**Station-keeping:**
$$r = -\|\mathbf{W} \cdot \delta\boldsymbol{\alpha}\| - \lambda\|\mathbf{a}_{\text{cmd}}\|$$

**Orbit raising:**
$$r = \kappa \, \Delta a - \lambda\|\mathbf{a}_{\text{cmd}}\|$$

**Deorbit:**
$$r = -|h - h_{\text{target}}| - \lambda\|\mathbf{a}_{\text{cmd}}\|$$

where $\lambda$ is the fuel penalty coefficient (default 0.1) and $\mathbf{W}$ is a diagonal weight matrix.

---

### 8. Rendering Pipeline

#### 8.1 Architecture

The renderer uses **moderngl** with the **EGL** backend for fully headless GPU rendering (no display server required). The pipeline:

1. Create off-screen framebuffer (1280x720, RGB + depth)
2. Render Earth sphere (UV sphere mesh, 64x128 subdivisions, radius $R_\oplus$)
3. Render orbit trail (line strip with fading alpha)
4. Render satellite marker (point sprite)
5. Read pixels back to CPU as numpy array

#### 8.2 Earth Shading

The fragment shader blends between day and night textures based on the Sun illumination angle:

```glsl
float NdotL = dot(normalize(v_normal), sun_direction);
float blend = smoothstep(-0.1, 0.3, NdotL);
vec3 color = mix(night_color, day_color, blend);
```

An atmospheric rim glow is added at grazing angles:

```glsl
float rim = pow(1.0 - max(dot(N, V), 0.0), 3.0);
color += vec3(0.3, 0.5, 1.0) * rim * 0.3;
```

Earth orientation is set by rotating the model matrix through the GMST angle about the Z-axis.

---

### 9. Streaming Protocol

#### 9.1 Wire Format

Binary protocol over TCP with a 21-byte header:

```
Offset  Size  Type     Field
0       4     bytes    Magic ("ORBT")
4       1     uint8    Channel ID (1=video, 2=telemetry)
5       4     uint32   Payload length [bytes]
9       4     uint32   Sequence number
13      8     float64  Simulation time [s]
21      var   bytes    Payload
```

All multi-byte fields are big-endian.

**Video channel (ID=1):** Payload is a JPEG-encoded RGB frame (quality 85).

**Telemetry channel (ID=2):** Payload is a UTF-8 JSON dictionary containing orbital elements, altitude, velocity, and simulation time.

#### 9.2 Server

Two asyncio TCP servers on separate ports (default 9100 for video, 9101 for telemetry). Multiple clients are supported via broadcast. Disconnected clients are detected during write and removed from the subscriber list.

#### 9.3 Remote Viewing

The standalone `viewer.py` client connects via TCP (or SSH tunnel if ports are firewalled), decodes JPEG frames, overlays telemetry text, and displays via OpenCV.

---

### 10. Validation Results

| Test | Metric | Result | Requirement |
|------|--------|--------|-------------|
| Element round-trip | Relative error (all 6 elements) | $< 10^{-10}$ | $< 10^{-6}$ |
| Anomaly solver | Closure error at $e = 0.9$ | $< 10^{-12}$ rad | $< 10^{-10}$ rad |
| Two-body energy | Conservation over 100 orbits | $3.2 \times 10^{-11}$ relative | $< 10^{-6}$ |
| Kepler period | Position return error | 0.02 m | < 1 m |
| J2 RAAN drift | vs. analytical formula | 0.85% error | < 1% |
| Sun-synchronous | RAAN drift at $i=97.8°$, 600 km | $\approx 0.99°$/day | $0.9856 \pm 5\%$ |
| Frame transforms | ECI-ECEF-ECI round-trip | $1.2 \times 10^{-10}$ m | $< 10^{-9}$ m |
| RTN orthogonality | $\|RR^T - I\|_F$ | $< 10^{-15}$ | $< 10^{-12}$ |
| ROE (identical) | All elements | $< 10^{-14}$ | $< 10^{-12}$ |
| Atmospheric density | Sea-level | 1.225 kg/m$^3$ | $\pm 1\%$ |
| Sun ephemeris | Distance at J2000 | 0.983 AU | $\pm 2\%$ of 1 AU |
| Gymnasium | `check_env` | Pass | Pass |

---

### 11. Configuration

All parameters are set in `config/default.yaml`:

```yaml
orbit:
  semi_major_axis_m: 6778137.0    # ~400 km altitude
  eccentricity: 0.0001
  inclination_deg: 51.6           # ISS-like
  raan_deg: 0.0
  arg_periapsis_deg: 0.0
  true_anomaly_deg: 0.0

satellite:
  mass_kg: 100.0
  drag_coefficient: 2.2
  reflectivity_coefficient: 1.5
  area_to_mass_ratio: 0.01        # m^2/kg
  max_thrust_n: 0.1

propagator:
  dt: 10.0                        # RK4 timestep [s]
  enable_j2: true
  max_j_degree: 6
  enable_drag: true
  enable_srp: true
  enable_third_body: true
```

---

### 12. Usage

**Run simulation with streaming:**
```bash
python main.py
```

**Propagation only (no rendering or streaming):**
```bash
python main.py --no-render --no-stream --steps 1440
```

**Remote viewer (from another machine):**
```bash
# If ports are firewalled, tunnel through SSH:
ssh -L 9100:localhost:9100 -L 9101:localhost:9101 user@jetson-ip -N -f

# Run viewer:
python viewer.py --host localhost
python viewer.py --host localhost --headless       # terminal only
python viewer.py --host localhost --save-frames    # save PNGs
```

**Run tests:**
```bash
python -m pytest tests/test_core.py -v
```

---

### 13. Dependencies

| Package | Purpose |
|---------|---------|
| PyTorch >= 2.0 | Tensor computation, batched propagation |
| NumPy >= 1.24 | Array operations, rendering math |
| Gymnasium >= 0.29 | RL environment interface |
| moderngl >= 5.8 | OpenGL rendering (EGL backend) |
| Pillow >= 10.0 | JPEG encoding/decoding |
| PyYAML >= 6.0 | Configuration loading |
| OpenCV >= 4.8 | Viewer display (optional) |

---

### 14. References

1. Vallado, D.A. *Fundamentals of Astrodynamics and Applications*, 4th ed. Microcosm Press, 2013.
2. Meeus, J. *Astronomical Algorithms*, 2nd ed. Willmann-Bell, 1998.
3. D'Amico, S. "Autonomous Formation Flying in Low Earth Orbit." PhD Thesis, TU Delft, 2010.
4. US Standard Atmosphere, 1976. NOAA/NASA/USAF.
5. Montenbruck, O. and Gill, E. *Satellite Orbits: Models, Methods, Applications*. Springer, 2000.

---

*Generated for the orbital_simulation project. All equations implemented in PyTorch with float64 precision.*
