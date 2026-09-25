# Isolated literature-exact experiment: contract audit

The active pricer and regular price-PDE PINN use the **same separable four-shock SDE**:

```
dS/S = (r-q)dt + sqrt(v_s)dW_s,S + sqrt(v_f)dW_f,S
dv_i = kappa_i(theta_i-v_i)dt + sigma_i sqrt(v_i)dW_i,v
Corr(dW_i,S,dW_i,v)=rho_i; all cross-factor shock correlations zero.
```

In factor-paired shock order, the correlation matrix has two independent 2x2
blocks, with eigenvalues 1 +/- rho_s and 1 +/- rho_f. Individual |rho_i|<1 is
sufficient. The instantaneous log-price/variance cross coefficient is
rho_i*sigma_i*v_i, exactly as in `regular_pinn_torch.residual`. There is no
v_s/v_f cross derivative. Both characteristic exponents add independently.

`src/constraints.py` additionally requires rho_s^2+rho_f^2<1. The existing
`docs/DOUBLE_HESTON_ENGINE.md` explicitly attributes this to the inherited
handoff, not to the separable SDE. Such a disk is necessary for a **different**
three-shock construction with one return Brownian correlated at constant rho_i
with two mutually independent variance Brownian shocks. That construction would
have rho_i*sigma_i*sqrt((v_s+v_f)*v_i) price/variance cross terms and is NOT the
implemented affine pricer/PDE. We do not relabel one construction as the other.

The [Christoffersen–Heston–Jacobs paper](https://pure.au.dk/ws/files/17142435/rp09_34.pdf),
section 3, equations (3)–(5), explicitly uses four shocks with the separable
correlations described above. The [Chang–Wang–Zhang article](https://onlinelibrary.wiley.com/doi/10.1155/2021/6634779),
Table 1, distinguishes its ordinary Double Heston comparator from its fractional
extension. The supplied numbers belong to the ordinary comparator; we do not
implement or claim to replicate their fractional model or their empirical MSE.

Thus the numerical CF and PINN support the supplied ordinary-Double-Heston
correlations without projection. `literature_exact.py` is an isolated adapter
validating pairwise bounds, strict Feller and slow-first order. No canonical
validator, frozen experiment or parameter decoder is weakened or patched.
The original published vector is reordered only: paper factor 2 is our slow
factor, paper factor 1 our fast factor.

A radius-.95 projected vector is saved in the bank as
`PROJECT_VALID_LITERATURE_ANALOG`, for canonical-engine regression diagnostics
only. It is not called published, not pooled with the main bank, and not
selected after observing results. Projection is mathematically unnecessary
for the four-shock literature experiment, but lets us cross-check canonical
production entry points on their admissible subset.

Other audit findings: old baseline code has only five SH starts and discards
failed starts; the new isolated benchmark records a global search and twelve
local starts. Old NIFTY panel code uses settlement prices and interpolates
onto a fixed grid; this is not reused for market prices. The newer close-based
`nifty_fixed_crisis.prepare` and its anchor-only parity fit are reused unchanged.
The unused `smoke_test_pricing_function` is explicitly not Heston and is never
called. Old inverse-prior, exact-polish and Archive-2 trainers are not reused.

All previously examined 61 dates remain development. Older NIFTY work also
used March/April 2026 and earlier corpus dates; the old corpus reserves dates
through August 3. To avoid ambiguous reuse, validation is reserved August 4–21,
2026, final August 24–September 16, 2026, beyond the local archive cutoff.
If new official files cannot be obtained, the real-market leg remains blocked;
previously seen dates are not substituted or called fresh. This protects local
research exposure; unknown exposure by external team members cannot be certified.
