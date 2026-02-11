# Complete Technical Audit and Refactor Plan
## Coupled Heart–Brain Hopf Oscillator System

**Auditor:** Senior Computational Neuroscientist / ML Systems Engineer  
**Date:** February 11, 2025  
**Subject:** `new_code/Code_update copy.ipynb`

---

# PART 1 — Reconstruct What the Code Is Doing

## 1.1 Full Pipeline Overview

```
Data Loading → Preprocessing → Heart Oscillator Sim → ECG Model → Brain Hopf ODE
                                    ↑                                    ↓
                              FeedbackMLP ←────────────── rcos_phi (brain output)
```

## 1.2 Signal Preprocessing

**ECG:**
- Source: MNE FIF file `transdef_mf2pt2_rest_raw.fif`, channel index 322, samples 2000:4000
- Negated (`-data[0]`) — likely to align R-peak polarity
- Pipeline: detrend → 4th-order Butterworth bandpass (1.5–20 Hz) → z-score normalization
- Output: 1D array, length 2000

**EEG:**
- Source: `scout_id_309.mat` → `Value`, same time window [2000:4000]
- Same preprocessing per channel (0.5–20 Hz bandpass)
- Output: 2D array `(N_channels, 2000)` — 309 cortical regions

**Structural connectivity (SC):**
- Source: `SC_CC120309-27.mat` → `sc`
- Normalized: `(sc / max(sc)) * 0.01`, diagonal zeroed
- Builds `non_zero_indices_per_row` for sparse connectivity lookups

## 1.3 Heart Oscillator Simulation

**Function:** `simulate_coupled_oscillators()`

**Physiology:** Two coupled oscillators representing cardiac dynamics (e.g., sinoatrial node + atrial-ventricular coupling).

**Equations (polar form):**
```
dr₁/dt = α·r₁ - r₁³ + A₁₂·r₂·cos(θ₁₂ + n·(φ₂ - φ₁)) + modulation₁
dr₂/dt = α·r₂ - r₂³ + A₂₁·r₁·cos(θ₂₁ + n·(φ₁ - φ₂)) + modulation₂
dφ₁/dt = ω₁ + A₁₂·(r₂/r₁)·sin(θ₁₂ + n·(φ₂ - φ₁))
dφ₂/dt = ω₂ + A₂₁·(r₁/r₂)·sin(θ₂₁ + n·(φ₁ - φ₂))
```

**Output:** Stack of `(x₁, y₁, x₂, y₂)` = `(r₁cos φ₁, r₁sin φ₁, r₂cos φ₂, r₂sin φ₂)` — shape `(N_steps, 4)`.

**Parameters:** α=1, ω₁=5.01, ω₂=5.1 rad/s, A_init=0.0001, θ_init=π, n=1. Euler integration with dt=0.01.

## 1.4 ECG Prediction Model

**Class:** `HeartModel`

**Architecture:** MLP: 4 → 100 → 100 → 50 → 1, Sigmoid activations.

**Input:** `(batch, 4)` = heart oscillator (x₁,y₁,x₂,y₂) trajectory.

**Output:** Scalar ECG prediction per time step.

**Physiology:** Maps abstract oscillator state to normalized ECG amplitude.

## 1.5 Oscillator Neural Layer

**Class:** `OscillatorLayer`

**Purpose:** Intermediate bank of Hopf oscillators driven by ECG features; output drives the brain.

**Equations:**
```
drᵢ/dt = (μ - rᵢ²)·rᵢ + coupling_r + input_featuresᵢ
dφᵢ/dt = ωᵢ
```

**Coupling:** `coupling_r = strength · Σⱼ Cᵢⱼ · rᵢ · cos(φᵢ - φⱼ)` — radius coupling only; phase coupling is absent.

**Parameters:** N_osc=128, T=2 s, fs=100 → 200 Euler steps. Learnable μ, ω; fixed C (buffer), coupling_strength (buffer).

**Output:** `(r·cos φ, r·sin φ)` concatenated → shape `(batch, N_osc*2)`.

## 1.6 Brain Hopf ODE Network

**Class:** `ODEFuc` (ODE function), `TorchRevHopfNetwork` (wrapper)

**State vector:** `[r, φ, θ, ω, α]` — radii, phases, phase offsets (N×N), frequencies, amplitudes.

**Equations:**
```
phase_diff = φⱼ/ωⱼ - φᵢ/ωᵢ + θᵢⱼ/(ωᵢ·ωⱼ)
D = D_function(t)  (target EEG)
P = Σ αᵢ·rᵢ·cos(φᵢ)
e = D - P

drᵢ/dt = (μ - rᵢ²)·rᵢ + Σⱼ |Scᵢⱼ|·rⱼ·cos(phase_diff) + e·cos(φᵢ) + ecg_inputᵢ
dφᵢ/dt = ωᵢ + Σⱼ |Scᵢⱼ|·(rⱼ/rᵢ)·sin(phase_diff) - (e/rᵢ)·sin(φᵢ)
dθᵢⱼ/dt = η_θ · sin(phase_diff) · |Scᵢⱼ|
dωᵢ/dt = -η_ω · e · sin(φᵢ)
dαᵢ/dt = η_α · e · rᵢ · cos(φᵢ)
```

**ecg_input:** From `ECGToOscillatorMLP(hidden_repr[t_idx])` — brain drive from ECG-derived features.

**Integration:** `torchdiffeq.odeint_adjoint`, RK4, rtol=1e-5, atol=1e-7.

## 1.7 Structural Connectivity Handling

**Function:** `expand_structural_connectivity(Sc_region, osc_per_region)`

- Expands region-level SC (e.g., 68×68) to oscillator-level (e.g., 204×204) with `osc_per_region=3`.
- Intra-region: constant `intra_value=0.0001`.
- Inter-region: random block scaled to preserve row sum from `Sc_region[i,j]`.
- Diagonal set to 0.

**Target region:** `target_idx` (e.g., 4) — EEG channel to fit. Uses `non_zero_indices_per_row[target_idx]` to get connected regions.

## 1.8 EEG Fitting

**Target:** `D_function(t)` = linear interpolation of `eeg_processed[target_idx, ::10]`.

**Prediction:** `P_out = Σ αᵢ·rᵢ·cos(φᵢ)` — weighted sum of oscillators’ real parts.

**Loss:** MSE(P_out, D).

## 1.9 Staged Training Strategy

| Stage | Name | Frozen | Trainable | Purpose |
|-------|------|--------|-----------|---------|
| 0 | Heart pre-training | — | HeartModel | ECG from heart oscillators |
| 1 | Brain pre-training | mlp_model=None | r, φ, θ, ω, α via ODE plasticity | Fit P_out to EEG |
| 2 | ECG→Brain | Brain ODE (η=0) | ECGToOscillatorMLP | Brain drive from ECG |
| 3 | Feedback | — | HeartModel + FeedbackMLP | Heart modulated by brain rcos_φ |

**Critical:** Stage 3 calls `trained_heart_model.apply(reset_weights)` — **all heart weights are reinitialized**, destroying Stage 0.

## 1.10 Feedback Loop

**Flow:** Brain `rcos_φ = Σ rᵢ·cos(φᵢ)` → FeedbackMLP → modulation [2] → `simulate_coupled_oscillators(modulation=...)` → HeartModel → ECG.

**Problem:** `modulation = feedback_output.mean(dim=0).detach()` — gradients do not flow through feedback.

**Problem:** Modulation is **constant** over time (mean over all t); no temporal dependence.

## 1.11 Data Flow Between Components

```
ECG raw → preprocess → ecg_processed
EEG raw → preprocess → eeg_processed
SC → expand_structural_connectivity → Sc_reduced_osc

simulate_coupled_oscillators() → HeartModel → ecg_pred
HeartModel.get_features(heart_osc) → ecg_features [T, 50]
ecg_features → ECGToOscillatorMLP → brain_drive [N]
brain_drive → ODEFuc.ecg_input
ODEFuc → odeint_adjoint → r, φ, θ, ω, α, rcos_φ
rcos_φ → FeedbackMLP → modulation [2]
modulation → simulate_coupled_oscillators → HeartModel
```

## 1.12 Gradient Propagation

- **Stage 1:** `torch.no_grad()` — no gradients; pure Hebbian/plasticity in ODE.
- **Stage 2:** Gradients flow: D_true ← P_out ← r,φ ← odeint_adjoint ← ODEFuc ← ecg_input ← ECGToOscillatorMLP. `hidden_repr` is detached (from `get_features` in no_grad).
- **Stage 3:** Gradients flow: loss ← HeartModel(sim_osc) and FeedbackMLP. But `modulation` is detached and `simulate_coupled_oscillators` is NumPy — **no gradient through feedback path**.

## 1.13 Where torchdiffeq Is Used

- **Single call:** `odeint_adjoint(self.ode_func, y0, t_eval_tensor, method='rk4', ...)` in `TorchRevHopfNetwork.solve()`.
- **Purpose:** Integrate brain Hopf ODE with adjoint method for memory-efficient backprop through time.

## 1.14 Closed-Loop Heart↔Brain Control

**Intended:** Heart → ECG → HeartModel features → ECGToOscillatorMLP → brain drive → brain dynamics → rcos_φ → FeedbackMLP → heart modulation.

**Actual:** Loop is **broken** in implementation:
1. Feedback uses averaged rcos_φ, not time-varying.
2. Heart simulator is NumPy, no gradients.
3. Training disentangles heart and feedback; no joint optimization of the loop.

---

# PART 2 — List Everything That Is Wrong or Fragile

## 2.1 Mathematical / Modeling Design

### M1. OscillatorLayer phase dynamics incorrect
- `dφᵢ/dt = ωᵢ` only — no phase coupling.
- Standard Kuramoto-style: `dφᵢ/dt = ωᵢ + Σ Kᵢⱼ·sin(φⱼ - φᵢ - θᵢⱼ)`.
- **Severity:** High — oscillators never synchronize.

### M2. OscillatorLayer radial coupling formula wrong
- `coupling_r = Σ Cᵢⱼ · rᵢ · cos(φᵢ - φⱼ)` — uses `rᵢ` not `rⱼ` for incoming coupling.
- Correct: `Σⱼ Cᵢⱼ · rⱼ · cos(φᵢ - φⱼ)` (or equivalent).
- **Severity:** High.

### M3. OscillatorLayer input_features directly added to dr/dt
- `drdt = ... + input_features.unsqueeze(-1)` — dimension mismatch risk; input is (batch, N_VNS), r is (batch, N_osc, 1).
- If N_VNS ≠ N_osc, shapes fail. Currently N_VNS=128, N_osc=128 — OK by coincidence.
- **Severity:** Medium — brittle.

### M4. ODEFuc phase_diff formula non-standard
- `phase_diff = φⱼ/ωⱼ - φᵢ/ωᵢ + θᵢⱼ/(ωᵢ·ωⱼ)` — mixes phase/omega and theta in a non-standard way.
- Standard: `phase_diff = φⱼ - φᵢ - θᵢⱼ` or `φⱼ - φᵢ`.
- **Severity:** High — interpretability and correctness unclear.

### M5. Frequency units inconsistent
- `simulate_coupled_oscillators`: ω₁=5.01, ω₂=5.1 rad/s (~0.8 Hz) — too slow for heart.
- Heart rate ~1 Hz → ω ≈ 2π rad/s.
- `get_random_frequencies` returns rad/s; `OscillatorLayer` uses `2*π*freqs_hz` (2–10 Hz).
- **Severity:** High — physiological mismatch.

### M6. Heart model: 4D input vs physiology
- Two oscillators (4D) is a heavy simplification.
- No clear mapping to P-QRS-T or heart rate variability.
- **Severity:** Medium.

### M7. EEG target: single channel
- `D_function` from one EEG channel; brain has many regions.
- P_out = Σ αᵢ·rᵢ·cos(φᵢ) is a single scalar — cannot fit multi-channel EEG.
- **Severity:** Medium.

### M8. Feedback: constant modulation
- `modulation = feedback_output.mean(dim=0)` — one [2] vector for entire T.
- Ignores temporal structure of rcos_φ.
- **Severity:** Critical.

### M9. rcos_φ as brain output
- Σ rᵢ·cos(φᵢ) mixes all oscillators; no lead field or source model.
- **Severity:** Medium — unphysiological.

### M10. Ill-posed error term e = D - P
- e drives both plasticity and direct coupling. For multi-channel, D and P dimensions must match.
- **Severity:** Low (single channel) — but design doesn’t scale.

### M11. Frozen vs trainable confusion in Stage 2
- `eta_omega=0, eta_alpha=0, eta_theta=0` — no plasticity.
- θ, ω, α are still in state but not learning.
- **Severity:** Medium — wasteful.

---

## 2.2 Machine-Learning Implementation

### ML1. Stage 3: reset_weights destroys pre-training
```python
trained_heart_model.apply(reset_weights)
```
- Reinitializes all parameters of HeartModel.
- **Severity:** Critical — Stage 0 work is discarded.

### ML2. Feedback: .detach() breaks gradient flow
```python
modulation = feedback_output.mean(dim=0).detach().cpu().numpy()
```
- No gradients through FeedbackMLP to rcos_φ.
- **Severity:** Critical.

### ML3. NumPy in critical path
- `simulate_coupled_oscillators` is NumPy → no autograd.
- Feedback training cannot backprop through heart dynamics.
- **Severity:** High.

### ML4. TorchRevHopfNetwork recreated every epoch
- In `train_mlp_on_frozen_brain`, `TorchRevHopfNetwork` is built inside the loop.
- Same for Stage 1 — wastes allocation; not necessarily wrong but inefficient.
- **Severity:** Medium.

### ML5. hidden_repr fixed, not updated
- `hidden_repr` from initial heart simulation; never updated when heart model changes.
- In Stage 3, heart is retrained but `hidden_repr` from old heart is still used for brain.
- **Severity:** High.

### ML6. D_function uses NumPy
- `D = torch.tensor(self.D_function(t.item()), ...)` — breaks graph at each call.
- Should use batched t or a precomputed tensor.
- **Severity:** Medium.

### ML7. Unnecessary clamps
- `r = torch.clamp(r, 0.01, 2.0)` in OscillatorLayer can hide instability.
- Many ±1e2 clamps in ODEFuc — band-aids for bad dynamics.
- **Severity:** Medium.

### ML8. Shape/device mismatches
- `ECGToOscillatorMLP` output_dim=N must match ODEFuc N.
- `initial_brain_params['r']` etc. from NumPy; converted to tensor in solve — OK.
- `fill_diagonal_(True)` then "Force diagonal = 0" — comment contradicts code; diagonal is 1 in mask then multiplied by C.
- **Severity:** Medium.

### ML9. No batching
- Single sequence; no batch dimension for multiple subjects/trials.
- **Severity:** Low for research.

### ML10. Nondifferentiable operations
- `t_idx = min(int(t.item() * 100), ...)` — index lookup breaks smooth gradient through t.
- **Severity:** Medium.

### ML11. Mixing NumPy and Torch
- `np.concatenate`, `theta[-1].cpu().numpy()` in training loops.
- Fine for data transfer, but frequent conversions are wasteful.
- **Severity:** Low.

---

## 2.3 Software Engineering / Research Hygiene

### S1. Notebook global state
- `ecg_processed`, `eeg_processed`, `Sw_all`, `non_zero_indices_per_row` are global.
- Cells assume prior execution order.
- **Severity:** High.

### S2. Duplicated logic
- `preprocess_signal` used for ECG and EEG.
- Heart model training loop pattern repeated.
- **Severity:** Medium.

### S3. Magic numbers
- 322, 2000:4000, 0.01, 0.02, 0.1, 1e-5, 25000, 5000, etc.
- **Severity:** Medium.

### S4. Hard-coded hyperparameters
- No config file; everything in code.
- **Severity:** High.

### S5. No configuration system
- Paths, seeds, learning rates, etc. scattered.
- **Severity:** High.

### S6. No reproducibility controls
- `np.random.seed`, `torch.manual_seed` used inconsistently.
- No global seed function.
- **Severity:** High.

### S7. Inconsistent seeding
- `get_random_frequencies(seed=42)`, `expand_structural_connectivity(seed=42)`, `torch.manual_seed(seed+1)` in OscillatorLayer.
- **Severity:** Medium.

### S8. Plotting inside training
- Main block has plotting; fine for notebook but not for scripts.
- **Severity:** Low.

### S9. File I/O mixed with computation
- `loadmat`, `read_raw_fif` in data cell; no abstraction.
- **Severity:** Medium.

### S10. Unclear naming
- `ODEFuc`, `D_function`, `P_out`, `Sw_all`, `Sc`, `C`.
- **Severity:** Medium.

### S11. No entrypoints
- Only `if __name__ == '__main__'`; no CLI.
- **Severity:** Medium.

### S12. No logging
- Only `print`.
- **Severity:** Medium.

### S13. No checkpoints
- No `torch.save`; no resumption.
- **Severity:** High.

### S14. No tests
- No unit tests.
- **Severity:** High.

### S15. No modular structure
- Single notebook; no imports from other modules.
- **Severity:** High.

### S16. OscillatorLayer coupling_mask
- `fill_diagonal_(True)` then "diagonal = 0" — confusing. Mask 1 = keep; C*0 on diagonal gives 0.
- **Severity:** Low.

### S17. Pre-train brain epoch count
- Loop says `range(30)` but print says "Epoch {epoch+1}/100".
- **Severity:** Low.

---

# PART 3 — Concrete Improvements

## Issue: OscillatorLayer phase dynamics (M1)

**Fix:** Add phase coupling:
```python
dphi_dt = self.omega + coupling_strength * torch.sum(
    self.C * torch.sin(phi - phi.transpose(-2, -1)), dim=-1
).unsqueeze(-1)
```

**Justification:** Enables synchronization; standard Kuramoto coupling.

**Required:** Yes for meaningful oscillator behavior.

---

## Issue: OscillatorLayer radial coupling (M2)

**Fix:** Use rⱼ for incoming coupling:
```python
coupling_r = self.coupling_strength * torch.sum(
    self.C * r.transpose(-2, -1) * torch.cos(phi - phi.transpose(-2, -1)), dim=-1
).unsqueeze(-1)
```
(Ensure r from neighbor j, not self.)

**Required:** Yes.

---

## Issue: reset_weights in feedback training (ML1)

**Fix:** Remove `trained_heart_model.apply(reset_weights)`.

**Justification:** Preserve Stage 0 ECG prediction.

**Required:** Critical.

---

## Issue: Feedback gradient flow (ML2)

**Fix:** Implement heart dynamics in PyTorch:
- `HeartOscillatorTorch` module with `forward(t, modulation)`.
- `modulation = feedback_mlp(rcos_phi)` per time step (or batched).
- End-to-end: rcos_φ → FeedbackMLP → modulated heart → HeartModel → ECG loss.

**Required:** Yes for learning feedback.

---

## Issue: Constant modulation (M8)

**Fix:** Use time-varying modulation:
```python
modulation = feedback_mlp(rcos_phi_tensor)  # (T, 2)
# Pass modulation[t] to heart simulator at each step
```
Requires heart simulator to accept `modulation(t)`.

**Required:** Yes.

---

## Issue: phase_diff formula (M4)

**Fix:** Use standard phase difference:
```python
phase_diff = phi[None, :] - phi[:, None] - theta
```
Remove ω from denominator unless there is a specific theory for it.

**Required:** Yes for interpretability.

---

## Issue: Frequency units (M5)

**Fix:** Heart: ω ≈ 2π·1 Hz. Explicitly document Hz vs rad/s; use `2*π*f_hz` everywhere.

**Required:** Yes for physiology.

---

## Issue: hidden_repr staleness (ML5)

**Fix:** Recompute `hidden_repr` from current heart model when it changes, or jointly optimize heart + brain + feedback.

**Required:** Yes for consistency.

---

## Issue: D_function in ODE (ML6)

**Fix:** Precompute `D_tensor = torch.tensor(D_function(t_eval), ...)` and index: `D = D_tensor[t_idx]`.

**Required:** Optional — minor speedup.

---

## Issue: Configuration (S4, S5)

**Fix:** Use YAML/JSON config:
```yaml
data:
  ecg_channel: 322
  t_start: 2000
  t_end: 4000
model:
  n_osc: 128
  fs: 100
```
Load with Hydra or simple custom loader.

**Required:** Yes for research.

---

## Issue: Reproducibility (S6, S7)

**Fix:**
```python
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
```
Call at startup.

**Required:** Yes.

---

## Issue: Checkpointing (S13)

**Fix:**
```python
torch.save({
    'heart_model': heart_model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'epoch': epoch,
}, checkpoint_path)
```

**Required:** Yes for long runs.

---

## Stability: OscillatorLayer

**Change:** Use soft constraints instead of hard clamp:
```python
r = r + drdt * self.dt
r = torch.clamp(r, 1e-6, 10.0)  # only for numerical safety
```
Tune μ and coupling so limit cycle is stable.

**Required:** Optional.

---

# PART 4 — Convert to Proper Python Project

## Proposed Layout

```
project_root/
├── configs/
│   ├── default.yaml          # Hyperparameters, paths
│   └── experiment_001.yaml   # Override for specific run
├── data/
│   ├── __init__.py
│   ├── loaders.py            # load_ecg, load_eeg, load_sc
│   └── paths.py              # Path constants
├── preprocessing/
│   ├── __init__.py
│   └── signals.py            # preprocess_signal, bandpass, etc.
├── dynamics/
│   ├── __init__.py
│   ├── heart_oscillators.py  # simulate_coupled_oscillators (NumPy)
│   ├── heart_oscillators_torch.py  # PyTorch version for feedback
│   └── brain_ode.py          # ODEFuc, TorchRevHopfNetwork
├── models/
│   ├── __init__.py
│   ├── heart.py              # HeartModel
│   ├── ecg_to_brain.py       # ECGToOscillatorMLP
│   ├── oscillator_layer.py   # OscillatorLayer
│   └── feedback.py           # FeedbackMLP
├── training/
│   ├── __init__.py
│   ├── heart.py              # train_heart_model
│   ├── brain.py              # pre_train_brain_model
│   ├── ecg_brain.py          # train_mlp_on_frozen_brain
│   └── feedback.py           # train_feedback_loop
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py            # MSE, correlation, etc.
│   └── plots.py              # plot_losses, plot_predictions
├── utils/
│   ├── __init__.py
│   ├── connectivity.py       # expand_structural_connectivity, get_random_frequencies
│   ├── seeding.py            # set_seed
│   └── checkpoint.py         # save_checkpoint, load_checkpoint
├── scripts/
│   ├── run_full_pipeline.py   # Main entry
│   ├── run_stage_1_only.py   # Debug
│   └── visualize_results.py
├── main.py                   # CLI entrypoint
├── requirements.txt
└── README.md
```

## Module Responsibilities

| Folder | Content | Torch/NumPy |
|--------|---------|-------------|
| `configs/` | YAML files | — |
| `data/` | MNE, loadmat, path handling | NumPy for arrays |
| `preprocessing/` | Filtering, normalization | NumPy |
| `dynamics/` | ODEs, oscillators | Mixed: NumPy for legacy heart, Torch for brain |
| `models/` | NN modules | Torch only |
| `training/` | Training loops | Torch |
| `evaluation/` | Metrics, plots | NumPy for plotting |
| `utils/` | Helpers | Both |

## Config Loading

```python
# configs/default.yaml
data:
  ecg_path: "../transdef_mf2pt2_rest_raw.fif"
  eeg_path: "../scout_id_309.mat"
  sc_path: "../SC_CC120309-27.mat"
  ecg_channel: 322
  t_start: 2000
  t_end: 4000

model:
  heart:
    input_dim: 4
    hidden_dim: 100
    feature_dim: 50
  ecg_to_brain:
    ecg_dim: 50
    n_vns: 128
    output_dim: 9  # N from brain
  oscillator:
    n_osc: 128
    fs: 100
    T: 2.0

training:
  heart_epochs: 25000
  brain_epochs: 30
  mlp_epochs: 100
  feedback_epochs: 5000
  seed: 42
```

## Experiment Launch

```bash
python main.py --config configs/default.yaml --stage all
python main.py --config configs/exp001.yaml --stage feedback --resume checkpoints/run_001.pt
```

---

# PART 5 — How to Make It Runnable as `.py` Scripts

## 5.1 Removing Notebook Globals

- Replace globals with function arguments and return values.
- Use a `PipelineState` dataclass or similar to pass data between stages.

```python
@dataclass
class PipelineState:
    ecg_processed: np.ndarray
    eeg_processed: np.ndarray
    sc_matrix: np.ndarray
    heart_model: nn.Module
    brain_params: dict
    # ...
```

## 5.2 Argument Passing

```python
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/default.yaml')
    parser.add_argument('--stage', choices=['heart', 'brain', 'mlp', 'feedback', 'all'])
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--resume', default=None)
    parser.add_argument('--output_dir', default='results')
    args = parser.parse_args()
    cfg = load_config(args.config)
    # ...
```

## 5.3 CLI Design

```
main.py --config CONFIG --stage STAGE [--device DEVICE] [--resume PATH] [--seed N]
```

## 5.4 Config Files

- YAML with Hydra or OmegaConf.
- Override via CLI: `--model.heart.hidden_dim 200`.

## 5.5 CPU/GPU Switching

```python
device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
```

## 5.6 Reproducibility

```python
def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
```

## 5.7 Training Loop Structure

```python
for epoch in range(n_epochs):
    loss = compute_loss(...)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch + 1) % log_interval == 0:
        logger.info(f"Epoch {epoch+1}, Loss: {loss.item():.6f}")
    if (epoch + 1) % ckpt_interval == 0:
        save_checkpoint(...)
```

## 5.8 Checkpointing

```python
def save_checkpoint(state, path):
    torch.save(state, path)

def load_checkpoint(path):
    return torch.load(path, map_location='cpu')
```

## 5.9 Experiment Resumption

- Save `epoch`, `optimizer.state_dict()`, `model.state_dict()`.
- On `--resume`, load and continue from `epoch+1`.

## 5.10 Result Logging

- Use `logging` or `wandb`/`tensorboard`.
- Log losses, config, git hash.

## 5.11 Plot Generation

- Move to `evaluation/plots.py`.
- Call from script after training: `plot_results(state, output_dir)`.

## 5.12 Output Layout

```
output_dir/
├── checkpoints/
│   ├── heart_epoch_25000.pt
│   ├── brain_epoch_30.pt
│   └── final.pt
├── logs/
│   └── run_20250211_120000.log
├── figures/
│   └── full_feedback_result_idx4.png
└── results/
    └── results_idx4.npz
```

---

# PART 6 — Final Verdict

## Scientifically Promising

1. **Heart–brain coupling idea:** Physiology supports cross-talk (e.g., vagal, baroreflex).
2. **Hopf-based brain model:** Well-established for neural oscillations.
3. **Structural connectivity:** Using SC for coupling is standard.
4. **Staged training:** Sensible for a complex pipeline.
5. **Adjoint ODE:** Appropriate for memory-efficient training.

## Unjustified or Incorrect

1. **OscillatorLayer equations:** Missing phase coupling; radial coupling formula wrong.
2. **Constant feedback modulation:** Removes temporal structure.
3. **reset_weights in feedback:** Clearly wrong.
4. **Detached feedback:** No learning of the feedback path.
5. **Phase_diff formula:** Non-standard, hard to interpret.
6. **Heart frequencies:** 5 rad/s is too slow for heart rate.

## Must Fix Before Publication

1. Remove `reset_weights` in feedback training.
2. Correct OscillatorLayer dynamics (phase and radial coupling).
3. Implement differentiable feedback (PyTorch heart + no detach).
4. Use time-varying modulation.
5. Add reproducibility (seeding, config).
6. Validate heart oscillator frequencies (~1 Hz).
7. Clarify or fix phase_diff in ODEFuc.

## Over-Engineered

1. **N² phase offsets θ:** Possibly excessive; could use fewer parameters.
2. **Two heart oscillators:** Minimal; might be fine for a first model.
3. **128 oscillators in ECG→Brain:** Could start smaller for debugging.

## Should Be Removed

1. `reset_weights` call in feedback training.
2. `.detach()` on modulation.
3. Contradictory comments (e.g., diagonal handling).

## Missing

1. Validation set and proper train/val split.
2. Statistical tests or confidence intervals.
3. Ablation studies (e.g., with/without feedback).
4. Comparison to baselines.
5. Unit tests.
6. Documentation (docstrings, README).
7. Proper citation of the Hopf and connectivity literature.

---

**Summary:** The overall idea is publishable if the listed issues are fixed. The current implementation has critical bugs (reset_weights, broken feedback gradient, incorrect oscillator equations) that prevent it from working as intended. A refactor into a modular project with config, logging, and checkpointing is strongly recommended.
