# Mentor Baseline Protocol: Double Heston Forward PINN V1

## Scientific question

Given one known, valid Double Heston parameter vector and synthetic canonical
CALL prices, can a 7-input forward network learn the price field while
explicitly satisfying four losses?

```text
L_total = lambda_PDE L_PDE + lambda_B L_B + lambda_T L_T + lambda_data L_data
lambda_PDE = lambda_B = lambda_T = lambda_data = 1
```

The lambda values are a neutral, untuned baseline. Test metrics cannot affect
training, checkpoint selection, architecture, or lambda selection.

## Parameter source and current variance states

The source is the first eligible stored TRAIN record, in stored file order,
from `data/final_r2_clean_10000/surfaces.jsonl`, after verifying SHA256
`148b579a4f6ce572e34796e872479c4c016c89bbcd20438c2bb62d6b6960f1f6`
and all canonical validation rules.

Canonical order:

```text
[kappa_slow, theta_slow, sigma_slow, rho_slow, v0_slow,
 kappa_fast, theta_fast, sigma_fast, rho_fast, v0_fast]
```

For every synthetic pricing sample the eight structural parameters remain
fixed. The sampled current states `v_slow` and `v_fast` replace
`v0_slow` and `v0_fast` in the canonical pricing call. A fixed source-record
`v0` is never used while claiming arbitrary variance-state conditioning.

## Inputs, output, and architecture

Inputs are `[S, v_slow, v_fast, tau, K, r, q]`; the output is one raw
European CALL price. The network has five 128-neuron tanh hidden layers and
67,201 trainable parameters. Inputs are affinely normalized over the complete
data/terminal/boundary loss domain. The output is not normalized or bounded,
so the physical operator differentiates the raw price field in float64.

## Time convention and PDE

`tau = T - t`. The residual is the tau-forward form

```text
R = C_tau
    - (r-q) S C_S
    - 0.5 (v_slow+v_fast) S^2 C_SS
    - kappa_slow (theta_slow-v_slow) C_vslow
    - kappa_fast (theta_fast-v_fast) C_vfast
    - rho_slow sigma_slow v_slow S C_Svslow
    - rho_fast sigma_fast v_fast S C_Svfast
    - 0.5 sigma_slow^2 v_slow C_vslowvslow
    - 0.5 sigma_fast^2 v_fast C_vfastvfast
    + r C.
```

A solution has `R=0`. The positive `C_tau` sign follows from
`tau=T-t`; copying a backward `C_t` equation without this sign change would
be wrong. The independent-factor model has spot/variance mixed derivatives
but no `v_slow/v_fast` cross derivative.

## Four losses and separate point classes

- `L_PDE = mean((R / max(|C|, 1))^2)` on interior collocation points.
- `L_T = MSE(C(S,0), max(S-K,0))` on explicit `tau=0` points.
- `L_B = L_low + L_high`, with `C(S_low,tau)=0` and
  `C(S_high,tau)=S_high exp(-q tau)-K exp(-r tau)`.
- `L_data = mean(((C-C_ref)/S)^2)` on training contracts priced by
  `src/double_heston.py` with 64 Gauss-Laguerre nodes.

Data, PDE, terminal, low-boundary, and high-boundary batches are sampled
separately from deterministic seed streams. `S_low=1e-4`; `S_high=2.0`,
outside the interior spot range `[0.70,1.30]`. Boundary strikes remain in the
global contract strike range.

## Data and split isolation

The baseline uses seeded deterministic pseudo-random sampling with independent
split streams. Intended counts are 4096/1024/1024. Split IDs and hashes are
created before training and are disjoint. The trainer requests only train and
validation indices. The selected checkpoint minimizes validation normalized
price RMSE. Test evaluation is an explicit later operation guarded by an
atomic claim written before test access. An existing claim or metrics artifact
fails closed with no repeat override.

## Optimizer and numerical health

AdamW uses learning rate `1e-3`, weight decay `1e-6`, batch size 256,
256 PDE points, 128 terminal points, and 128 points for each stock boundary.
Maximum epochs are 1000 with patience 100. Every epoch records all four losses,
low/high boundary subcomponents, validation errors, PDE/terminal/boundary
diagnostics, gradient norm, finite-gradient status, and duration. Non-finite
losses or gradients fail immediately.
The CLI has explicit `--device cpu|cuda` routing. Collocation coordinates are
sampled deterministically on CPU and recreated as float64 autograd leaves on
the selected device; the Kaggle P100 notebook passes `--device cuda`.

## Evaluation and figures

The one-shot synthetic evaluation reports raw and normalized pricing errors,
relative-error mean/p95 under a disclosed epsilon, PDE residual RMS/max,
terminal RMSE/max, boundary RMSEs, and inference runtime. Six 300-dpi grayscale
figures cover price vs strike, price vs maturity, absolute error vs strike,
four training losses, validation error, and PDE residual diagnostics.

Sparse reference markers in slice figures are held-out deterministic
figure-grid samples; dashed curves are dense canonical evaluations from the
same pricing source, and solid curves are PINN predictions.

## Interpretation boundary

This experiment can demonstrate only synthetic fixed-parameter forward
pricing behavior. It cannot establish inverse parameter identification,
global recovery, uniqueness, real-market validity, model superiority, or a
representation freeze. The next scientific step, only after mentor review, is
a validation-only lambda-weight study with the test split still sealed.
