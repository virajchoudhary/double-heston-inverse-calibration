# Deliverable A — repository audit

Prerequisite for the unified redesign. Per the specification: *"Do not implement anything
until you have written down the exact current model convention and reproduced the existing
baseline."* Nothing here is implemented; everything below is verified against source or
measured by running it.

## A.0 Headline findings, including two that contradict the brief's assumptions

1. **The Feller condition, the slow/fast ordering and the correlation disk are enforced by
   the repository's own pricing engine**, not merely by my `decode`. `price_double_heston_call`
   *refuses* a Feller-violating vector. Hard-enforcing them in `decode` is therefore
   correct — they are part of the implemented model class. (§A.4)
2. **`PARAM_BOX` does silently clip, and worse than a plain box clip.** Because
   `sigma = eta * sqrt(2 kappa theta)`, clipping `theta` or `kappa` silently moves `sigma`
   too — even when `sigma` was inside its own range. Demonstrated on a vector the exact
   engine prices without complaint. (§A.5)
3. The two variance factors are **exactly permutation-symmetric** (max relative price
   difference 1.07e-14). The label symmetry is real and `kappa_slow < kappa_fast` is the
   correct resolution. (§A.6)
4. All ten claimed characteristics of the old architecture are **CONFIRMED**. (§A.7)
5. The stored parameter-recovery benchmark **reproduces exactly**, to all six printed
   digits, from a regenerated test split. (§A.8)

## A.1 File map

Repository engine and contracts (not authored by me):

| path | role |
|---|---|
| `src/double_heston.py` | exact Fourier pricer, Gauss-Laguerre, 64 default nodes |
| `src/constants.py` | `PARAMETER_NAMES` — the canonical ordering |
| `src/constraints.py` | Feller gap, correlation disk, slow/fast ordering |
| `data/.../final_parameter_panel.csv` | sealed 10,000-vector parameter panel |

Authored for this study:

| path | role |
|---|---|
| `src/mentor_dh_pinn/collocation_domain.py` | CIR-derived (S, v_slow, v_fast, tau) domain |
| `src/mentor_dh_pinn/model_v3/v4/v5.py` | forward pricing PINNs; V5 is the trained surrogate |
| `src/mentor_dh_pinn/calibrate.py` | `PARAM_BOX`, `encode`/`decode`, `FourierCalibrator` |
| `src/mentor_dh_pinn/dual_pinn.py` | `ShortEndNet`, `LongEndNet`, `combine`, `DualPINN` |
| `src/mentor_dh_pinn/dual_pinn_data.py` | synthetic geometry, parameter draw, noise model |
| `src/mentor_dh_pinn/polish.py` | stage two: prior-regularised polish, `blend` |
| `src/mentor_dh_pinn/nifty_panel.py` | real NSE surface construction |
| `scripts/mentor_dh_pinn/train_dual_pinn.py` | inverse training loop |
| `scripts/mentor_dh_pinn/nifty_model_comparison.py` | real-market comparison |

## A.2 Model convention

`double_heston_characteristic_function` returns the characteristic function of `log(S_T)`:

```
exponent = i u [log S + (r - q) tau]  +  psi_slow(u)  +  psi_fast(u)
```

The two factors combine **additively in log-characteristic space**, i.e. they are
independent — `E[dW_slow dW_fast] = 0`. Each `psi` is the Little-Heston-Trap form in
`heston_log_characteristic_exponent`, which raises `FloatingPointError` on a degenerate
denominator rather than returning a wrong number.

Forward and discounting: `F = S exp((r - q) tau)`, discount `exp(-r tau)`. Verified: pricing
with `spot=F, r=0, q=0` and multiplying by the discount factor reproduces the direct form to
2.4e-15 relative.

## A.3 Parameter convention

`src/constants.py` fixes the canonical order, and it is what every module uses:

```
0 kappa_slow  1 theta_slow  2 sigma_slow  3 rho_slow  4 v0_slow
5 kappa_fast  6 theta_fast  7 sigma_fast  8 rho_fast  9 v0_fast
```

## A.4 What the ENGINE requires (measured, not assumed)

`src/constraints.py`, reached through every pricing entry point:

| constraint | engine behaviour on violation |
|---|---|
| `kappa_slow < kappa_fast` | `ValueError` |
| `2 kappa theta - sigma^2 > 0`, per factor | `ValueError: "...Feller gap must be strictly positive"` |
| `-1 < rho < 1`, each | `ValueError` |
| `rho_slow^2 + rho_fast^2 < 1` | `ValueError` |

Measured: `p = [1.10, 0.090, 0.90, ...]` has `2 k th - s^2 = -0.612`; the engine **refuses to
price it**.

**Consequence for the redesign.** The brief warns against silently preserving Feller "unless
it is actually part of your intended model class". It *is*: this engine cannot price outside
it. A network permitted to emit Feller-violating vectors would emit vectors that cannot be
priced, scored, or refined. Feller, ordering and the disk must stay hard — but must be
**documented as model-class facts**, which is what this section does.

## A.5 What MY decode adds on top — and this is the real defect

`calibrate.py` constants: `PARAM_BOX` (8 finite ranges), `V0_BOX = (1e-3, 1.2)`,
`ETA_MAX = 0.995`, `RHO_RADIUS = 0.97`.

Reachability test, pushing `z` to ±40:

```
z = -40  kappa_slow 0.15   theta_slow 0.014  sigma_slow 2.7e-19  v0 0.001
z = +40  kappa_slow 2.80   theta_slow 0.240  sigma_slow 1.154    v0 1.200
```

Round-trip on a market-plausible vector **outside** the box:

| parameter | target | `decode(encode(target))` | |
|---|---:|---:|---|
| kappa_slow | 1.1000 | 1.1000 | |
| **theta_slow** | **0.4000** | **0.2400** | silently clipped |
| **sigma_slow** | **0.2800** | **0.2169** | silently moved, though in range |
| rho_slow | −0.5500 | −0.5500 | |
| v0_slow | 0.9000 | 0.9000 | |
| **kappa_fast** | **14.000** | **11.000** | silently clipped |
| **theta_fast** | **0.3000** | **0.1900** | silently clipped |
| **sigma_fast** | **0.6000** | **0.4233** | silently moved, though in range |

The exact engine prices the **target** vector without complaint (0.22600850). So the model
class permits it; only `decode` forbids it, and it reports nothing.

`sigma` is the subtle part: it is not stored, it is *reconstructed* as
`eta * sqrt(2 kappa theta)`. Clipping `theta` or `kappa` therefore drags `sigma` with it.
A box-membership check on `sigma` alone would not have caught this.

Two further silent restrictions:

* `RHO_RADIUS = 0.97` is **stricter than the engine's 1.0** — `decode` forbids a legal band
  `0.97 <= sqrt(rho_s^2+rho_f^2) < 1`, for no stated reason.
* At `z = -40`, `sigma` underflows to 2.7e-19 — a degenerate corner reachable by the optimiser.

## A.6 Factor permutation symmetry — tested, not assumed

Swapping the slow and fast five-blocks and pricing with `enforce_ordering=False`:

| K | tau | original | swapped | rel. diff |
|---|---|---|---|---|
| 0.85 | 0.082 | 0.155876958764 | 0.155876958764 | 0.00e+00 |
| 1.00 | 0.493 | 0.114085730221 | 0.114085730221 | 0.00e+00 |
| 1.15 | 0.082 | 0.005197839827 | 0.005197839827 | 1.07e-14 |

**Maximum relative difference 1.07e-14** — the factors are exactly exchangeable. The
repository already anticipates this: `enforce_ordering=False` exists "only for factor-symmetry
diagnostics". `kappa_slow < kappa_fast` is the correct and sufficient resolution, and
`decode` implements it by construction (`kappa_fast = kappa_slow + (hi - kappa_slow) sigmoid`).

## A.7 Claimed characteristics — all verified in source

| claim | verdict | evidence |
|---|---|---|
| fixed 45-price input | CONFIRMED | 5 x 9 = 45 |
| 5 expiries 30/60/90/180/365 d | CONFIRMED | `ALL_DAYS = (30,60,90,180,365)` |
| 9 strikes per expiry | CONFIRMED | `linspace(0.85, 1.15, 9)` |
| spot fixed to 1 in training | CONFIRMED | `SPOT = 1.0` |
| r = 0.05 | CONFIRMED | `RATE = 0.05` |
| q = 0.01 | CONFIRMED | `CARRY = 0.01` |
| separate short/long networks | CONFIRMED | short (30,60,90), long (180,365) |
| ~3 rounds of information exchange | CONFIRMED | checkpoint `cross=True, sweeps=3` |
| output centres a regularised exact fit | CONFIRMED | `polish.py` |
| global scalar lambda | CONFIRMED | `polish_choice.json`: `lam=3.0`, start `ensemble` |
| iid multiplicative lognormal noise <= 1% | CONFIRMED | `NOISE_MAX=0.01`, `level~U(0,0.01)`, `clean*exp(shock - level^2/2)` |
| V5 surrogate used in training | CONFIRMED | `Repricer(outputs/mentor_dh_pinn_v5/checkpoint_v5.pt)` |
| exact engine used in final fit/eval | CONFIRMED | `polish.exact_surface`, `FourierCalibrator` |

## A.8 Exact losses

Inverse training (`train_dual_pinn.py:133`), `lambda_reprice = 30.0`:

```
L = mean_j ((p_hat_j - p*_j)/span_j)^2  +  30 * mean_i ((C_hat_i(p_hat) - C_clean_i)/S)^2
```

The network input is the **noisy** surface; both targets are the **clean** truth. There is
**no PDE-residual term in the inverse networks** — the physics enters through the V5 surrogate
(itself PDE-trained) and through the exact engine at stage two. This is worth stating plainly,
because "PINN" in the inverse stage means *amortised inverse model with a physics repricer*,
not PDE collocation. The brief reaches the same conclusion independently.

Stage two (`polish.py`):

```
argmin_z ||(C(decode(z)) - C_obs)/S||^2 + lam * eps^2 * ||z - z_net||^2
eps = max(noise_level, 1e-4) * RMS(C_obs / S)
```

## A.9 Data generation

`draw_truths`: 60% of structural vectors resampled from the sealed panel, 40% fresh in-box
draws filtered by `_valid`. `v0_slow`, `v0_fast` log-uniform on `(0.01, 0.60)` independently.
Splits: train 200,000 / seed 11, validation 25,000 / seed 22, test 25,000 / seed 33.

## A.10 Benchmark reproduction

The test split was regenerated from scratch (seed 33, sealed panel, 235 s) and both stored
checkpoints re-evaluated:

| arm | reproduced median | stored median | reproduced mean | stored mean | within 0.10 |
|---|---:|---:|---:|---:|---:|
| independent | 0.140435 | 0.140435 | 0.154398 | 0.154398 | 0.22864 = 0.22864 |
| aware | 0.130666 | 0.130666 | 0.144562 | 0.144562 | 0.28796 = 0.28796 |
| ensemble | 0.133133 | 0.133133 | 0.146571 | 0.146571 | 0.27456 = 0.27456 |

**Exact to all printed digits.** The pipeline is deterministic and the stored numbers are real.

## A.11 What cannot currently be reproduced, and why

The session scratchpad holding the working clone was wiped between sessions. Lost:

* `outputs/mentor_dh_pinn_v5/checkpoint_v5.pt` — the differentiable surrogate. **Training
  cannot be re-run** without retraining V5 first. Evaluation is unaffected (it uses the exact
  engine), which is why A.10 succeeded.
* `dual_train.npz` (144 MB) and `dual_validation.npz` — regenerable, seeds 11 and 22, about
  30 min and 4 min respectively.
* `run_inverse_study.py` (multi-start `STARTS`) and `run_model_comparison.py`
  (`fit_heston`, `heston_call`). **The classical baselines and all repricing benchmarks
  cannot be re-run until these are rewritten.** The reported numbers stand on their stored
  CSVs, which survive, but I cannot re-derive them from source today.

This is stated rather than worked around: the parameter-recovery benchmark is reproduced from
scratch; the repricing benchmarks are currently only *archived*, not reproducible.

## A.12 Audit conclusions carried into the redesign

| finding | consequence |
|---|---|
| Feller / ordering / disk are engine-level | keep hard; document as model-class facts; do **not** relax |
| `PARAM_BOX` silently clips, and drags `sigma` | remove; replace with unbounded transforms plus a loud OOD report |
| `RHO_RADIUS = 0.97` stricter than engine | relax to the engine's own bound |
| factors exactly exchangeable (1.07e-14) | keep ordered `kappa` parameterisation |
| no PDE term in the inverse loss | the exact engine must become the teacher, per the brief |
| repricing benchmarks not reproducible | rewrite the two lost baseline scripts before claiming any comparison |
