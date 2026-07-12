# Mathematical Model: IMU-Based Double-Tap Recognition on PC

## 1. Problem Statement

A user performs a **double-tap** on a laptop touchpad, palm rest, or lid. An onboard IMU (6-DoF: 3-axis accelerometer + 3-axis gyroscope) records the resulting mechanical vibrations. The goal is to detect the double-tap event **online** (causally) from the streaming IMU signal.

**Challenges:**

- Single taps, typing, trackpad clicks, and fan vibration produce similar impulsive signatures
- Orientation changes affect the gravity component in accelerometer readings
- Inter-tap interval and tap strength vary across users
- Detection must be **causal**: only past and present samples may be used

The mathematical model serves two roles: it defines the **physical signal** and it guides **dataset construction, soft labeling, network architecture, training, and inference**.

---

## 2. Coordinate Frames and IMU Measurement Model

### 2.1 Frames

| Frame | Description |
|-------|-------------|
| **World** $W$ | Inertial frame, gravity $\mathbf{g}^W = [0,\,0,\,-9.81]^\top$ m/s² |
| **Body** $B$ | IMU sensor frame fixed to chassis |
| **Tap** $T$ | Local impact direction at contact point |

The body frame rotates with respect to the world according to orientation $\mathbf{R}_{WB}(\boldsymbol{\theta})$ (rotation matrix from Euler angles or quaternion).

### 2.2 Accelerometer

The accelerometer measures **specific force** (includes gravity):

$$
\mathbf{a}^B(t) = \mathbf{R}_{BW}(\boldsymbol{\theta}(t))\,\mathbf{g}^W + \mathbf{a}_{\text{vib}}^B(t) + \mathbf{b}_a + \mathbf{n}_a(t)
$$

where:

- $\mathbf{a}_{\text{vib}}^B$ — vibration-induced linear acceleration from taps
- $\mathbf{b}_a$ — slowly varying bias
- $\mathbf{n}_a \sim \mathcal{N}(0,\,\sigma_a^2 \mathbf{I})$ — white sensor noise

### 2.3 Gyroscope

$$
\boldsymbol{\omega}^B(t) = \boldsymbol{\omega}_{\text{vib}}^B(t) + \mathbf{b}_g + \mathbf{n}_g(t)
$$

Gyroscope captures rotational recoil of the chassis during impact.

### 2.4 Stacked Observation

$$
\mathbf{y}(t) = \begin{bmatrix} \mathbf{a}^B(t) \\ \boldsymbol{\omega}^B(t) \end{bmatrix} \in \mathbb{R}^6
$$

Sampled at rate $f_s$ (typically 100–400 Hz on laptop IMUs), giving sequence $\{\mathbf{y}_n\}_{n=0}^{N-1}$ with $t_n = n/f_s$.

---

## 3. Physical Model of a Single Tap

A tap at time $t_k$ with parameters $\mathbf{p}_k = (A_k,\, f_k,\, \zeta_k,\, \boldsymbol{\phi}_k)$ excites a **damped second-order oscillator**:

$$
h(t - t_k;\, \mathbf{p}_k) = A_k \cdot e^{-\zeta_k \omega_k (t - t_k)} \cdot \sin\bigl(\omega_k (t - t_k) + \phi_k\bigr) \cdot \mathbf{u}_k \cdot H(t - t_k)
$$

where:

- $\omega_k = 2\pi f_k$ — natural angular frequency (chassis resonance, typically 30–120 Hz)
- $\zeta_k \in (0,1)$ — damping ratio
- $\mathbf{u}_k \in \mathbb{R}^6$ — unit direction vector (impact geometry)
- $H(\cdot)$ — Heaviside step (causal impulse response)
- $A_k$ — impact amplitude (user-dependent)

The initial impulse can be approximated as a short burst (2–10 ms) followed by ring-down.

### 3.1 Simplified Impulse + Ring-Down

For synthesis and analysis, we use a two-stage model:

**Stage 1 — Contact impulse** (duration $\tau_{\text{imp}} \approx 3$ ms):

$$
\mathbf{i}_k(t) = \mathbf{I}_k \cdot \mathrm{rect}\!\left(\frac{t - t_k}{\tau_{\text{imp}}}\right)
$$

**Stage 2 — Chassis response** (convolution with impulse response):

$$
\mathbf{h}_k(t) = \mathbf{i}_k(t) * \mathbf{s}_k(t), \quad s_k(t) = e^{-\alpha_k t}\sin(2\pi f_k t)\,\mathbf{u}_k
$$

---

## 4. Double-Tap Signal Model

A **double-tap** is the superposition of two tap responses plus background:

$$
\mathbf{y}(t) = \mathbf{y}_{\text{bg}}(t) + h(t;\, t_1, \mathbf{p}_1) + h(t;\, t_2, \mathbf{p}_2) + \boldsymbol{\eta}(t)
$$

subject to the **double-tap constraint**:

$$
\Delta t = t_2 - t_1 \in [T_{\min},\, T_{\max}], \quad T_{\min} \approx 0.12\,\text{s},\; T_{\max} \approx 0.45\,\text{s}
$$

In practice, $\Delta t$ is enforced as a **hard constraint on training data**: only samples with $\Delta t \in [T_{\min}, T_{\max}]$ are included as positives. Invalid intervals are excluded rather than down-weighted.

### 4.1 Background Activity

$$
\mathbf{y}_{\text{bg}}(t) = \mathbf{R}_{BW}\mathbf{g}^W + \mathbf{b} + \mathbf{r}_{\text{typing}}(t) + \mathbf{r}_{\text{fan}}(t)
$$

- $\mathbf{r}_{\text{typing}}$ — sparse low-amplitude impulses from keyboard
- $\mathbf{r}_{\text{fan}}$ — low-frequency periodic vibration (~30–80 Hz)

### 4.2 Negative Class: Single Tap

$$
\mathbf{y}_{\text{single}}(t) = \mathbf{y}_{\text{bg}}(t) + h(t;\, t_1, \mathbf{p}_1) + \boldsymbol{\eta}(t)
$$

Only one impulse within the observation window. With second-tap Gaussian labeling, single taps receive $y_n = 0$ everywhere (no $t_2$), which naturally distinguishes them from double-taps.

---

## 5. Detection as a Sequential Hypothesis Test

### 5.1 Event Definition

Detection is anchored on the **second tap** $t_2$: the double-tap is complete when the second impulse occurs. The training target is a soft bump centered at $t_2$, not a hard step function.

At frame $n$:

$$
\hat{p}_n = P(\text{double-tap at } t_2 \mid \mathbf{y}_{0:n}) \approx f_\theta(\mathbf{y}_{0:n})
$$

**Hard labels** $l_n \in \{0, 1\}$ (e.g. a step at $t_2$) are discouraged: they impose a discontinuous target and create sparse positive frames. **Gaussian soft labels** (Section 5.4) provide a smooth, localized target around $t_2$.

At **inference**, hysteresis (Section 8) converts $\hat{p}_n$ to a hard trigger.

### 5.2 Likelihood Ratio (Generative View)

Under hypothesis $\mathcal{H}_1$ (double-tap) vs $\mathcal{H}_0$ (not):

$$
\Lambda_n = \frac{p(\mathbf{y}_{0:n} \mid \mathcal{H}_1)}{p(\mathbf{y}_{0:n} \mid \mathcal{H}_0)} \gtrless \gamma
$$

For nonlinear dynamics and unknown parameters, we use a **learned discriminative** approximation (Section 5.1). The Gaussian label (Section 5.4) localizes the positive target at $t_2$, matching the moment the likelihood ratio would peak.

### 5.3 Causal Constraint (Formal)

A function $f$ is causal if:

$$
\frac{\partial f(\mathbf{y}_{0:n})}{\partial \mathbf{y}_m} = 0 \quad \forall\, m > n
$$

**Enforcement in neural networks:**

1. **Causal convolution**: kernel only accesses past samples via left-padding
2. **Unidirectional GRU**: processes $n = 0, 1, \ldots$ in order
3. **No bidirectional layers**, no centered padding

### 5.4 Gaussian Soft Labeling (Second-Tap Centric)

The simplest model-guided soft label places a **Gaussian bump** at the second tap. Define:

$$
t_{\text{peak}} = t_2 + \tau_{\text{offset}}
$$

where $\tau_{\text{offset}} \approx 20$ ms accounts for contact + initial ring-down after the second impact. The frame-level target is:

$$
y_n = \exp\!\left(-\frac{(t_n - t_{\text{peak}})^2}{2\sigma^2}\right)
$$

| Parameter | Typical value | Role |
|-----------|---------------|------|
| $\sigma$ | 30–50 ms | Label width; wider = more timing tolerance |
| $\tau_{\text{offset}}$ | 15–25 ms | Peak slightly after $t_2$ |

**Why this works:**

- **Localized**: peak at $t_2$; values before $t_1$ are near zero (Gaussian tail decays quickly)
- **Smooth**: no step discontinuity; BCE receives graded targets
- **Simple**: one formula, two parameters ($\sigma$, $\tau_{\text{offset}}$)
- **Causal training**: $t_2$ comes from synthesis metadata; the network sees only $\mathbf{y}_{0:n}$ at runtime

Example at $\sigma = 40$ ms, $\Delta t = 300$ ms: at $t = t_1$, $|t - t_{\text{peak}}| \approx 280$ ms, so $y_n \approx e^{-24.5} \approx 0$; at $t = t_{\text{peak}}$, $y_n \approx 1$.

**Inter-tap interval constraint:** $\Delta t$ validity is enforced by **excluding** out-of-range samples from the training set (Section 5.6–5.7), not by scaling label amplitude.

### 5.5 Labels for Negative Classes

| Class | Soft target $y_n$ | Rationale |
|-------|-------------------|-----------|
| Background | $0$ | No second tap |
| Single tap | $0$ | No $t_2$; first tap alone does not produce a bump |
| Valid double-tap | Gaussian at $t_2$ | Clear positive |

Negatives stay at zero — the network learns to produce a response **only** when a second-tap transient appears, not on the first tap alone.

### 5.6 Synthetic Dataset Design

The generative model in `tap_recognition/physics.py` defines **what to simulate** and **how to parameterize diversity** for `IMUDoubleTapDataset`:

| Parameter | Sampling range | Purpose |
|-----------|----------------|---------|
| $\Delta t$ | $[T_{\min}, T_{\max}]$ | Only valid double-tap intervals |
| $A_k$ | 0.5–3.0 | Amplitude invariance |
| $f_k$ | 30–120 Hz | Chassis resonance diversity |
| $\zeta_k$ | 0.05–0.4 | Ring-down duration variation |
| $\mathbf{u}_k$ | random unit vector in $\mathbb{R}^6$ | Tap location / direction |
| $\mathbf{R}_{BW}$ | varied tilt angles | Gravity / orientation robustness |

**Sampling policy:**

- ~50% double-tap ($\Delta t \in [T_{\min}, T_{\max}]$)
- ~25% single tap
- ~25% background

Double-tap samples with $\Delta t \notin [T_{\min}, T_{\max}]$ are **not generated** — boundary ambiguity is avoided by data filtering, not label scaling.

Each sample returns:

$$
\mathcal{D} = \{ \mathbf{y}_{:N-1},\; \{y_n\}_{n=0}^{N-1},\; \text{metadata}(t_1, t_2, \Delta t,\, \mathbf{p}_k) \}
$$

Metadata is used for stratified evaluation, not as a hard training class label.

**Window length** follows from physical timescales (Section 7.2): $T = 64$ samples at $f_s = 100$ Hz (640 ms) covers the full double-tap envelope ($T_{\max} + 2 \times$ ring-down).

### 5.7 Recorded Data: Annotation and Window Extraction

Collected IMU recordings use a complementary pipeline implemented in `tools/label_imu.py` and `tap_recognition/dataset.py`.

#### 5.7.1 CSV format

Each recording is a CSV with columns:

| Column | Description |
|--------|-------------|
| `timestamp_ms` | Sample timestamp (ms) |
| `acc_x`, `acc_y`, `acc_z` | Accelerometer (m/s²) |
| `gyro_x`, `gyro_y`, `gyro_z` | Gyroscope (rad/s) |
| `segment_index` | Segment ID within the session |
| `action_label` | Human-readable action name |

Frame indices used in labels are **global row indices** over the full CSV (0 to $N-1$), matching the contiguous timeline shown in the labeling GUI.

#### 5.7.2 Interactive trigger annotation

Human annotators mark the **second tap** in each double-tap gesture. The GUI computes energy features:

$$
E_{\text{acc},\Delta}(n) = \bigl| \lVert \mathbf{a}_n \rVert - \lVert \mathbf{a}_{n-1} \rVert \bigr|, \qquad
E_{\text{gyro}}(n) = \lVert \boldsymbol{\omega}_n \rVert
$$

For each user-selected span $[n_{\min}, n_{\max}]$, the trigger frame is:

$$
n_2 = \arg\min_{n \in [n_{\min}, n_{\max}]} \hat{E}_{\text{gyro}}(n)
$$

where $\hat{E}_{\text{gyro}}$ is min–max normalized within the span. This picks the gyro energy trough associated with the second impact transient.

Labels are stored in a sidecar file `<recording>.txt`, one line per event:

```
frame_index, action_label
```

with `action_label = 1` for the second tap. Recordings of type `knock_once` or `arbitrary` receive **empty** label files and serve as negative-class sources during training.

#### 5.7.3 From trigger frames to training windows

`RecordedIMUDataset` converts annotations into fixed-length windows ($T = 64$ samples):

**Positive windows** (from `knock_twice` label files):

1. For each trigger frame $n_2$, extract a window starting at $\max(0,\, n_2 - 48)$.
2. Estimate the first tap $n_1$ as the peak of $E_{\text{acc},\Delta}(n)$ for $n < n_2$ within the window.
3. **Filter:** keep the window only if $(n_2 - n_1) / f_s \in [T_{\min}, T_{\max}]$.
4. Convert the in-window second-tap time: $t_2 = (n_2 - n_{\text{start}}) / f_s$.
5. Apply Gaussian soft labels (Section 5.4).

**Negative windows**:

- From `knock_once` / `arbitrary` recordings (empty labels): random windows with $y_n = 0$.
- From `knock_twice` recordings: random windows outside an exclusion zone $\pm 80$ frames around each $n_2$.

Each window is high-pass filtered (Section 6.1) before training. The dataset returns:

$$
\{ \mathbf{y}_{0:T-1},\; \{y_n\}_{n=0}^{T-1},\; \ell_{\text{window}} \}
$$

where $\ell_{\text{window}} \in \{0, 1\}$ is a hard window-level label (1 if the window contains a labeled second tap).

#### 5.7.4 Train / validation split

Recordings are organized into:

| Directory | Role |
|-----------|------|
| `data/train_data/` | Training CSVs + `.txt` labels |
| `data/valid_data/` | Held-out validation CSVs + `.txt` labels |

This file-level split avoids leakage across segments from the same session.

---

## 6. Feature Domain Preprocessing

### 6.1 Gravity Removal (High-Pass)

$$
\tilde{\mathbf{a}}_n = \mathbf{a}_n - \mathrm{LPF}_\alpha(\mathbf{a}_n)
$$

A first-order IIR high-pass with cutoff $f_c \approx 0.5$ Hz removes slow orientation/gravity drift while preserving tap transients. At inference, a **causal** one-pole high-pass must be used (not zero-phase `filtfilt`).

### 6.2 Normalization

Per-channel z-score using running statistics (causal exponential moving average):

$$
\hat{y}_{n,d} = \frac{y_{n,d} - \mu_{n,d}}{\sigma_{n,d} + \epsilon}, \quad \mu_{n,d} = \alpha \mu_{n-1,d} + (1-\alpha) y_{n,d}
$$

### 6.3 Optional Derived Features

- **Jerk**: $j_n = (\mathbf{a}_n - \mathbf{a}_{n-1}) / \Delta t$
- **Magnitude**: $\lVert \mathbf{a}_n \rVert$, $\lVert \boldsymbol{\omega}_n \rVert$

This demo uses the raw 6-channel vector after high-pass for simplicity.

---

## 7. Neural Network: Causal CNN + GRU

The model separates two timescales: **local tap transients** (CNN) and **inter-tap timing** (GRU).

### 7.1 Architecture

$$
\begin{aligned}
\text{Input:} \quad & x_n \in \mathbb{R}^6 \quad \text{(one IMU sample)} \\
\downarrow \quad & \text{Causal Conv1D} \times L \text{ layers} \\
& \text{kernel size } k,\ \text{dilation } d_l \\
& \text{receptive field: } R = 1 + \sum_l (k-1)\,d_l \\
\downarrow \quad & \text{GRU hidden state } h_n \in \mathbb{R}^H \\
& h_n = \mathrm{GRU}(c_n,\, h_{n-1}) \\
\downarrow \quad & \text{Linear} \to \sigma(\text{logit}) \to \hat{p}_n \in (0,1)
\end{aligned}
$$

| Component | Physical role |
|-----------|---------------|
| Causal Conv1D | Detect impulse + ring-down of each tap (~50–150 ms) |
| Dilated kernels | Expand receptive field without future leakage |
| GRU | Remember first tap; recognize second tap within $\Delta t$ window |

### 7.2 Receptive Field vs. Physical Timescales

| Quantity | Typical value |
|----------|---------------|
| Single tap ring-down | 50–150 ms |
| Inter-tap interval | 120–450 ms |
| Double-tap total duration | 200–600 ms |
| At $f_s = 100$ Hz | 20–60 samples |

Receptive field $R = 29$ samples ($\approx 290$ ms) covers one tap's ring-down; the GRU carries first-tap information across the inter-tap gap.

### 7.3 Loss Function

Train against **soft labels** $y_n \in [0, 1]$ with weighted binary cross-entropy:

$$
\mathcal{L} = -\frac{1}{N}\sum_{n} w_n \Bigl[ y_n \log \hat{p}_n + (1 - y_n)\log(1 - \hat{p}_n) \Bigr]
$$

where $w_n$ is an optional per-frame confidence weight. Optional **label smoothing** on top of soft targets:

$$
y_n \leftarrow (1 - \varepsilon)\, y_n + \varepsilon/2, \quad \varepsilon \approx 0.05
$$

Avoid relying on class reweighting (`pos_weight`) — the Gaussian label spreads positive mass over several frames, reducing imbalance naturally.

**Evaluation metrics** should include:

- Event-level detection rate and latency (relative to $t_2$)
- False trigger rate on single-tap and near-miss samples
- Probability calibration: $\hat{p}_n$ vs. $y_n$

### 7.4 Online Inference State

At runtime, maintain:

- GRU hidden state $\mathbf{h}_n$
- Causal conv buffer (past $R$ samples)
- Preprocessing EMA state

Each new IMU sample triggers one forward step; no re-processing of history required.

---

## 8. Detection Logic (Hysteresis)

Soft labels are for training only. At inference, convert $\hat{p}_n$ to a hard trigger with hysteresis:

$$
\text{trigger} = \begin{cases}
\text{True} & \text{if } \hat{p}_n > \theta_{\text{on}} \text{ for } K \text{ consecutive frames} \\
\text{False} & \text{if } \hat{p}_n < \theta_{\text{off}}
\end{cases}
$$

with $\theta_{\text{on}} = 0.7$, $\theta_{\text{off}} = 0.3$, $K = 2$, and a **refractory period** $T_{\text{ref}} = 500$ ms after a trigger.

$T_{\text{ref}} > T_{\max}$ ensures one physical double-tap cannot produce multiple triggers (Section 4 constraint).

---

## 9. Model-to-Pipeline Mapping

| Model element | Synthetic data | Recorded data | Network | Training | Inference |
|---------------|----------------|---------------|---------|----------|-----------|
| $h(t; t_k, \mathbf{p}_k)$ | `physics.py` simulator | Real chassis response | Causal CNN | — | — |
| Second tap $t_2$ | Synthesis metadata | `label_imu.py` trigger frames → Gaussian peak | GRU + CNN | BCE vs $y_n$ | Peak pick + hysteresis |
| $\Delta t$ validity | Hard filter in simulator | Estimated $n_1$ → interval filter | — | Exclude out-of-range windows | — |
| Single tap / background | $y_n = 0$ in simulator | Empty `.txt` + random negative windows | — | No false positive target | FP rate metric |
| Causal constraint | Labels from metadata only | Trigger frame → in-window $t_2$ | Left-pad conv + GRU | Per-frame loss | Step-wise GRU state |
| High-pass (Section 6.1) | `IMUSimulator.highpass` | Same filter in `RecordedIMUDataset` | — | — | Causal one-pole IIR |
| Window extraction | Fixed 64-sample synthesis | 64-sample crop around $n_2$ | — | `RecordedIMUDataset` | Streaming step |

### Code references

| Stage | Module |
|-------|--------|
| Synthetic generation | `tap_recognition/physics.py` |
| Gaussian soft labels | `tap_recognition/labels.py` |
| CSV loading | `tap_recognition/recording.py`, `tap_recognition/dataset.py` |
| Interactive annotation | `tools/label_imu.py`, `tools/label_all_imu.py` |
| Training datasets | `tap_recognition/dataset.py` (`IMUDoubleTapDataset`, `RecordedIMUDataset`) |
| Model | `tap_recognition/model.py` |
| Training loop | `train.py` |
| Online inference | `tap_recognition/inference.py` |

---

## 10. Summary

| Component | Model |
|-----------|-------|
| Single tap | Damped oscillator + short impulse |
| Double-tap | Two taps; label Gaussian centered at $t_2$ |
| IMU observation | 6-DoF + gravity + noise + background |
| Training target | $y_n = \exp(-(t_n - t_{\text{peak}})^2 / 2\sigma^2)$ |
| Negative classes | $y_n = 0$ (no second tap) |
| $\Delta t$ constraint | Hard filter: only $[T_{\min}, T_{\max}]$ in training data |
| Recorded annotation | Sidecar `.txt` trigger frames → in-window Gaussian labels |
| Detection | Causal $\hat{p}_n = f_\theta(\mathbf{y}_{0:n})$ |
| Neural net | Causal CNN (local transients) + GRU (inter-tap timing) |
| Inference | Hysteresis + refractory period on $\hat{p}_n$ |

This model motivates:

- The synthetic data generator in `tap_recognition/physics.py`
- Gaussian soft labels in `tap_recognition/labels.py`
- Recorded-data windowing in `tap_recognition/dataset.py` (`RecordedIMUDataset`)
- Interactive annotation in `tools/label_imu.py`
- The network design in `tap_recognition/model.py`
