# Vortices Functional Renormalization Group

An FRG project focusing on 2D bosons which provides a unified method dealing with the BKT physics.

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [What the Project Computes](#what-the-project-computes)
- [Architecture](#architecture)
- [Module Map](#module-map)
- [RGState: Shared Data Backbone](#rgstate-shared-data-backbone)
- [Adding a New Coupling](#adding-a-new-coupling)
- [BCS Track Workflow](#bcs-track-workflow)
- [BEC Track Workflow](#bec-track-workflow)
- [Subsystem Roles](#subsystem-roles)
- [Chemical Potential Root-Finding (`findMu`)](#chemical-potential-root-finding-findmu)
- [Entry Points](#entry-points)
- [Tc Sweep Pipeline](#tc-sweep-pipeline)
- [Testing](#testing)
- [Design Invariants](#design-invariants)

---

## Installation

This project is organized by [uv](https://github.com/astral-sh/uv).

Install uv with our standalone installers:

```bash
# On macOS and Linux.
curl -LsSf https://astral.sh/uv/install.sh | sh
```

```bash
# On Windows.
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Install with homebrew:

```bash
# With Homebrew
brew install uv
```

Sync dependencies:

```bash
uv sync
```

---

## Quick Start

After installation, run one of the main pipelines:

```bash
# 2D BEC-BCS crossover (superfluid stiffness phase diagram)
uv run plotTc.py

# 2D BEC BKT transition
uv run plotTcBEC.py
```

| Command | Track | Output |
|---------|-------|--------|
| `uv run plotTc.py` | BCS–BEC crossover | `Results/bcs-effixed-KToff.pickle` + pcolormesh plot |
| `uv run plotTcBEC.py` | Pure BEC / BKT | `Results/bec.pickle` + pcolormesh plot |
| `uv run scripts/demo_bcs.py` | BCS | Density map over `(μ, T)` |
| `uv run scripts/demo_bec.py` | BEC | RG trajectory plot (`ρ`, `A_v`, `A_l`, `y_k`) |
| `uv run scripts/demo_kt.py` | KT only | Standalone Kosterlitz–Thouless flow |
| `uv run pytest tests/` | — | Regression and unit tests |

**Library imports** (preferred after refactor):

```python
from bcs import BCSAction, BECAction, RGState, bcs_findMu, bec_findMu
```

Root-level modules (`BCSna.py`, `BECna.py`, `parsey.py`, etc.) are backward-compatible shims.

---

## What the Project Computes

Every run answers one of these physics questions via RG integration (lowering the momentum cutoff Λ, RG "time" `l = log(Λ/k)`):

| Question | API | Returns |
|----------|-----|---------|
| Total density at fixed `(ε_B, T, μ)` | `BCSAction(...).FinalNum()` | Scalar `n_tot` |
| Superfluid stiffness fraction `A_{l,k} = ρ_s/ρ_{0,k}` | `.FinalRhoSF()` | Scalar (0 if no superfluid) |
| Chemical potential for target density | `findMu(targetNum, ...)` | `μ` via bisection |
| Phase diagram over `(ε_B, T)` | `plotTc.py` / `plotTcBEC.py` | 2D grid → pickle + plot |

---

## Architecture

The codebase has **two parallel computation tracks** sharing infrastructure in the `bcs/` package:

```
                         ┌─────────────────────────────────────┐
                         │           USER / SCRIPTS            │
                         └─────────────────────────────────────┘
                    ┌────────────┬──────────────┬───────────────┐
                    │            │              │               │
              demo_bcs.py  demo_bec.py   plot_tc.py      plot_tc_bec.py
                    │            │              │               │
                    ▼            ▼              ▼               ▼
              BCSna shim   BECna shim      plotTc.py       plotTcBEC.py
                    │            │         (sweep logic)  (sweep logic)
                    └─────┬──────┴──────────────┴───────────────┘
                          ▼
                    ┌─────────── bcs/ package ───────────┐
                    │  bcs_action.py    bec_action.py  │  ← orchestrators
                    │  fermion.py       quantum.py     │  ← RG subsystems
                    │  thermal.py       kt.py          │
                    │  state.py  keys.py  distributions│  ← infrastructure
                    └──────────────────────────────────┘
```

| Track | Orchestrator | Phases | Physics |
|-------|--------------|--------|---------|
| **BCS** | `BCSAction` | Thermal → optional BEC | Fermion–boson crossover, pairing |
| **BEC** | `BECAction` | Single quantum | Pure bosons, BKT/KT vortex flow |

---

## Module Map

| Root shim | Package module | Role |
|-----------|----------------|------|
| `BCSna.py` | `bcs/bcs_action.py` | BCS-side RG orchestrator |
| `BECna.py` | `bcs/bec_action.py` | BEC-side RG orchestrator |
| `outerBCS.py` | `bcs/fermion.py` | Outer-layer BCS fermion RG |
| `thermBos.py` | `bcs/thermal.py` | Thermal boson RG (pre-condensation) |
| `quantumBos.py` | `bcs/quantum.py` | Quantum BEC boson RG + KT hook |
| `RigidBallRG.py` | `bcs/kt.py` | Kosterlitz–Thouless vortex sector |
| `parsey.py` | `bcs/state.py` | `RGState` container (`parseData` alias) |
| `plotTc.py` | `scripts/plot_tc.py` | BCS Tc sweep driver |
| `plotTcBEC.py` | `scripts/plot_tc_bec.py` | BEC Tc sweep driver |

Supporting modules: `bcs/keys.py` (coupling key enum), `bcs/distributions.py` (`nF`, `nB`), `bcs/mu_root.py` (bisection helpers).

---

## RGState: Shared Data Backbone

All RG integrations share one keyed state object (`bcs/state.py`):

```
RGState
├── _data: dict[str, float]     ← all coupling values (eb, g, rho, ...)
└── _keys_upd: list[str]        ← order for ODE vector ↔ array conversion
```

**Workflow:**

1. **Construction** — Subsystems register keys at init time:
   - `OuterBCSFermion` → `eb`, `ef`, `g`, `h`, `dfac`, `rhoF`
   - `ThermalBoson` → appends `g`, `eb`, `nthrm`
   - `QuantumAction` → appends `g`, `rho`, `avv`, `all` (+ KT keys if active)

2. **Integration** — `scipy.integrate.solve_ivp` sees a flat array:
   - `ydata.ylst()` exports state vector (uses `_keys_upd` order)
   - `ydata.update(arr)` unpacks after each RHS evaluation

3. **Merging derivatives** — Always by key name, never by array index:
   ```python
   dy = dybcs.add_by_key(dyThr)   # NOT positional sum
   ```

**Critical rule:** Two states with identical `_data` but different `_keys_upd` order must produce the same physics when merged.

---

## Adding a New Coupling

When extending the FRG flow with a new coupling variable, **do not start with `keys.py`**. The `Key` enum is a label for orchestrator indexing; the coupling enters the flow when a subsystem registers it in `RGState`.

### Pipeline

```
 1. Physics          Decide which sector owns ∂(newkey)/∂l
        │
        ▼
 2. Register         ydatakeysPrompt + dataAppend(...) in that subsystem
        │
        ▼
 3. Flow equation    dylst(l, dy) → dy.data["newkey"] = ...
        │
        ▼
 4. keys.py          Add Key enum entry (only if orchestrator/events need it)
        │
        ▼
 5. Orchestrator     bcs_action.py / bec_action.py merge, observables, events
        │
        ▼
 6. Tests            pytest regression + state merge if cross-subsystem
```

**Rule of thumb:** `keys.py` labels a drawer; the drawer is created in the subsystem's `dataAppend`.

### Subsystem ownership

| Subsystem | Module | Typical couplings |
|-----------|--------|-------------------|
| Fermion (outer BCS) | `bcs/fermion.py` | `eb`, `ef`, `g`, `h`, `dfac`, `rhoF` |
| Thermal boson | `bcs/thermal.py` | `g`, `eb`, `nthrm` |
| Quantum BEC | `bcs/quantum.py` | `g`, `rho`, `avv`, `all` |
| KT vortex sector | `bcs/kt.py` | `lutK`, `g1`, `g2`, … |

Quantum-sector flow equations are typically derived in `mfPopov.nb` (mean-field + Popov counterterms) and ported to `bcs/quantum.py`.

### BCS-track registration order

Subsystems construct in fixed order inside `BCSAction.__init__`:

```
OuterBCSFermion  →  registers eb, ef, g, h, dfac, rhoF
ThermalBoson     →  appends g, eb, nthrm  (shared keys update in place)
QuantumAction    →  appends g, rho, avv, all  (Phase 2 only, if becShift)
KT               →  appends lutK, g1, …  (when healLength ≤ 2π/k)
```

**Shared keys:** Names like `g` and `eb` already exist when a later subsystem starts. A new *name* is appended to `keysUpd`; an existing name only updates `_data`.

### When `keys.py` is required vs optional

**Add a `Key` enum entry when:**

- `bcs_action.py` or `bec_action.py` calls `key_index(keys, Key.YOUR_KEY)` for events or observables
- Termination functions index into `sol.y[idx, :]`
- Orchestrator merge logic uses `ydata.value(Key.YOUR_KEY)`

**Optional (string keys suffice) when:**

- The coupling is read/written only inside one subsystem via `dy.data["mykey"]` or `self.yval("mykey")`
- Precedent: KT uses `g1`, `g2` as strings; only `LUTK` is in the enum

### Example: how `dfac` was added

```python
# bcs/fermion.py — register + flow (steps 2–3)
ydatakeysPrompt = [..., "dfac", ...]
self.ydata.dataAppend({..., "dfac": 1.0, ...}, self.ydatakeys)

def dylst(self, l, dy):
    ...
    dy.data["dfac"] = self.dDfac()   # only nonzero when isBEC=True

# bcs/keys.py — step 4 (for typed access elsewhere)
DFAC = "dfac"

# bcs/bcs_action.py — step 5 only if orchestrator reads dfac directly
# (dfac is consumed inside fermion propagators; no extra Action glue needed)
```

### Checklist

```
□ Derive ∂ζ/∂l (e.g. in mfPopov.nb for quantum sector)
□ Add "zeta" to owning subsystem's ydatakeysPrompt + dataAppend initial value
□ Implement dy.data["zeta"] = ... in that subsystem's dylst()
□ Add Key.ZETA = "zeta" in keys.py if BCSAction/BECAction needs key_index
□ Update bcs_action.py: dZ / merge corrections, events, FinalNum/FinalRhoSF if needed
□ Add or extend pytest regression case
```

### Common pitfalls

| Pitfall | Consequence | Fix |
|---------|-------------|-----|
| Edit `keys.py` only, no `dataAppend` | Key missing from ODE vector | Register in owning subsystem first |
| Register in wrong subsystem | `add_by_key` strict merge fails | Put registration where `dylst` computes the derivative |
| New key not in `keysUpd` | `ylst()` / `update()` skip it | Pass key list to `dataAppend(..., self.ydatakeys)` |
| Merge by array index | Wrong physics if `keysUpd` orders differ | Always use `add_by_key`, never positional sum |

---

## BCS Track Workflow

`BCSAction` in `bcs/bcs_action.py` runs a **two-stage** RG flow when condensation triggers.

### Initialization

```
INPUTS: eb0, beta (=1/T), mu, cutoff, mf (= fermion mass), h (= pairing scale)

1. bareInt(eb0, mf, cutoff)  →  g₀  (BCS-specific formula + h² correction)
2. RGState() created; subsystems register keys
3. OuterBCSFermion + ThermalBoson constructed
4. solve_ivp(thrEqn, l ∈ [0, 20], events=ThrterminFunc)
```

### Phase 1: Thermal (`thrEqn`)

Each RHS call:

```
1. ydata.update(ylst)
2. bcsFer.dylst(l, dybcs)     ← fermion outer-layer (k-shells, Matsubara loops)
3. thrBos.dylst(l, dyThr)     ← thermal boson (∂g, ∂eb, ∂nthrm)
4. dy = dybcs.add_by_key(dyThr)
5. Apply h-renormalization: dZ corrections to eb, g; nthrm × (h/h₀)²
6. return dy.ylst()
```

**Termination event** (`ThrterminFunc`):

```
max(-m·g,  β·eb / (m·|g|)) + ε  crosses zero
```

When `solThr.status == 1` → `becShift = True` → Phase 2 starts.

### Phase 2: Quantum BEC (`spfEqn`) — optional

```
1. rho_init = -eb / g  (from terminal thermal state)
2. bcsFer.BECcritUpd(True)  — switch fermion propagators to BEC mode
3. QuantumAction constructed (KTSwitch=True)
4. solve_ivp(spfEqn, l ∈ [l_thermal_end, 20], events=BECterminFunc)
```

Phase 2 merges quantum boson + fermion derivatives with BEC-specific corrections (eb and nthrm frozen).

### BCS State Machine

```
                    ┌──────────────┐
                    │  INIT        │
                    └──────┬───────┘
                           ▼
              ┌────────────────────────┐
              │  THERMAL INTEGRATION   │
              │  thrEqn                │
              └────────────┬───────────┘
                           │
              ┌────────────┴────────────┐
              │                         │
     solThr.status ≠ 1          solThr.status == 1
     (no condensation)           (ThrterminFunc fired)
              │                         │
              ▼                         ▼
     becShift=False              BEC INTEGRATION
     use solThr for obs.          spfEqn (+ optional KT)
```

### Observables

```python
FinalNum():
  if becShift:
    ferNum = solBEC[rhoF];  bosNum = solThr[nthrm] + max(rho,0)·(h/h₀)²
  else:
    ferNum = solThr[rhoF];  bosNum = solThr[nthrm]
  return 2·ferNum + 2·bosNum

FinalRhoSF():
  if becShift and solBEC.status == 0:
    return solBEC[all]    # A_{l,k}
  else:
    return 0.0
```

---

## BEC Track Workflow

`BECAction` in `bcs/bec_action.py` is simpler — **single-stage** quantum flow:

```
INPUTS: eb2boson0, beta, mu, m

1. cutoff = √(m · eb)
2. g0 = bareInt(eb + μ, m, cutoff)   ← BEC-specific formula (≠ BCS bareInt)
3. rho_init = max(μ/g0, 0)
4. QuantumAction only (no fermion, no thermal phase)
5. solve_ivp(eqn, l ∈ [0, 20], events=BECterminFunc)

FinalNum()   → sol.y[rho]
FinalRhoSF() → sol.y[all]  (0 if event fired or integration failed)
```

---

## Subsystem Roles

| Module | Class | Per RG step (`dylst`) |
|--------|-------|------------------------|
| `fermion.py` | `OuterBCSFermion` | k-shell integration (particle/hole/boson), self-consistent `ef` diagnostic via Matsubara loops, outputs ∂eb, ∂g, ∂h, ∂ef, ∂dfac, ∂rhoF |
| `thermal.py` | `ThermalBoson` | `ek = Λ² e^{-2l}/(2m)`, Bose occupation `nB(ek+eb,β)` drives ∂g, ∂eb, ∂nthrm |
| `quantum.py` | `QuantumAction` | Quasi-particle pole `E_k`, condensate/vertex flows ∂ρ, ∂g, ∂all, ∂avv; activates KT when healing length ξ ≤ 2π/k |
| `kt.py` | `KT` | Vortex coupling hierarchy: ∂lutK, ∂g₁, ∂g₂, … (nonlinear KT equations) |
| `distributions.py` | `nF`, `nB` | Fermi/Bose Matsubara distribution functions |

**KT activation** (inside `QuantumAction`):

```
healLength = 1 / √(2m·g·ρ·avv)
if healLength <= 2π/k:
    ktStart = True  →  KT.eqRHS_ydata contributes to derivative
```

---

## Chemical Potential Root-Finding (`findMu`)

Both tracks fix density by bisection. Each evaluation runs a **full RG integration**.

```python
# BCS
from bcs import bcs_findMu as findMu
mu = findMu(targetNum, eb, beta, cutoff, mass)

# BEC
from bcs import bec_findMu as findMu
mu = findMu(targetNum, ebBos, beta, mass)
```

**Warm-start:** In Tc sweeps (`plotTc.py`), `mu_guess` from the previous β point narrows the bisection bracket. Module-level hint caches (`_bcs_mu_hint`, `_bec_mu_hint`) also persist across calls with the same `(eb, cutoff, mass, targetNum)`.

---

## Entry Points

### Demo scripts

```bash
# BCS density map over (μ, T)
uv run scripts/demo_bcs.py

# BEC RG trajectory curves
uv run scripts/demo_bec.py

# Standalone KT flow (no full Action wrapper)
uv run scripts/demo_kt.py
```

### Single-point library usage

```python
from bcs import BCSAction, BECAction

# BCS: density and superfluid stiffness at one point
action = BCSAction(eb0=1.2, beta=200.0, mu=-0.22, cutoff=3.0, mf=1.0)
n_tot = action.FinalNum()
rho_sf = action.FinalRhoSF()
bec_branch = action.becShift   # True if thermal phase hit condensation event

# BEC: condensate density at one point
bec = BECAction(eb2boson0=3.0, beta=1000.0, mu=0.3, m=1.0)
n = bec.FinalNum()
rho_sf = bec.FinalRhoSF()
```

---

## Tc Sweep Pipeline

Production phase-diagram runs parallelize over binding energy `eb` rows:

```
PARAMETER GRIDS (defined at import in plotTc.py / plotTcBEC.py)
  BCS: eblst × betalst  (~101 × ~10000 points)
  BEC: eblst × betalst  (~250 × ~15000 points)

SWEEP (12-worker multiprocessing Pool)
  for each eb row:
    mu_hint = None
    for each beta:
      μ = findMu(..., mu_guess=mu_hint)    ← warm-start across T
      mu_hint = μ
      ρ_sf = Action(...).FinalRhoSF()
    return row

OUTPUT
  rhogrid → Results/*.pickle → pcolormesh
  BCS axes: log(k_F a)  vs  k_B T / E_F
  BEC axes: 1/(k_n a)²  vs  k_B T m / k_n²
```

BCS sweep validates density: if `|√(FinalNum · 2π) - kF| > 0.05`, the point is discarded (returns 0).

---

## Testing

```bash
uv run pytest tests/
```

| Test file | Covers |
|-----------|--------|
| `test_regression_bcs.py` | Golden `FinalNum`, `FinalRhoSF`, `solThr`/`solBEC` status for BCS cases |
| `test_regression_bec.py` | Same for BEC track |
| `test_state_merge.py` | `add_by_key` ignores `keysUpd` order |
| `test_findmu_warmstart.py` | Bisection hint correctness |
| `test_distributions.py` | `nF`/`nB` numerical limits |

Regression tests lock numerical outputs after refactors — run them after any physics or integration change.

---

## Design Invariants

These constraints emerged from the `bcs/` package refactor and must be preserved:

1. **Separate `bareInt` formulas** — BCS (`bcs_action.py`) and BEC (`bec_action.py`) use different bare coupling expressions. Do not unify them.

2. **Key-based derivative merge** — Use `add_by_key()`, never assume `keysUpd` order matches between derivative buffers.

3. **No RHS/`upd(l)` caching** — k-scale derived quantities go stale if `upd(l)` is skipped across RHS steps, even when `l` is unchanged.

4. **Root shims for compatibility** — Prefer `from bcs import ...` in new code; root modules re-export from `bcs/`.
