# Canonical Double Heston Engine

Status date: 06 August 2026

## Provenance and scope

`src/double_heston.py` is an independent canonical reimplementation from the mathematical contract preserved in the handoff and primary literature. The unavailable teammate source, helpers, tests, exact bounds, and original fixtures were not recovered. Equivalence to that unavailable implementation is not claimed.

The handoff supplied the parameter order, model constraints, total-variance rule, Little-Heston-Trap requirement, 64-node Gauss-Laguerre setting, put-call parity rule, and variance-state propagation equation. The Python implementation, APIs, numerical checks, provisional bounds, regression fixture, tests, and calibration tooling are newly implemented here.

## Model and parameter order

The separable two-factor risk-neutral specification uses two square-root variance factors:

```text
dS/S = (r-q) dt + sqrt(v_slow) dW_slow,S + sqrt(v_fast) dW_fast,S
dv_i = kappa_i (theta_i-v_i) dt + sigma_i sqrt(v_i) dW_i,v
```

The instantaneous return variance is `v_slow + v_fast`. Each return shock is correlated with its corresponding variance shock by `rho_i`; the factor transforms are otherwise separable.

The exact vector order is:

1. `kappa_slow`
2. `theta_slow`
3. `sigma_slow`
4. `rho_slow`
5. `v0_slow`
6. `kappa_fast`
7. `theta_fast`
8. `sigma_fast`
9. `rho_fast`
10. `v0_fast`

## Characteristic function

For one factor and complex Fourier argument `u`, define

```text
b = kappa - rho*sigma*i*u
d = sqrt(b^2 + sigma^2*(u^2+i*u))
g = (b-d)/(b+d)
```

with the square-root branch normalized to `Re(d) >= 0`. The Little-Heston-Trap factor exponent is

```text
C = (kappa*theta/sigma^2) *
    ((b-d)*T - 2*log((1-g*exp(-d*T))/(1-g)))
D = ((b-d)/sigma^2) * (1-exp(-d*T))/(1-g*exp(-d*T))
L_i = C_i + D_i*v0_i
```

The Double Heston characteristic function of `log(S_T)` is

```text
phi(u) = exp(i*u*(log(S)+(r-q)*T) + L_slow(u) + L_fast(u))
```

Thus the two factor transforms multiply, while spot and deterministic carry enter once. This construction follows the factorwise affine form in Christoffersen, Heston, and Jacobs and uses the stable factor representation from Albrecher et al., [The Little Heston Trap](https://www.ma.imperial.ac.uk/~ajacquie/IC_Num_Methods/IC_Num_Methods_Docs/Literature/HestonTrap.pdf). The original Heston derivation is [Heston (1993)](https://doi.org/10.1093/rfs/6.2.327), and the two-factor affine construction is documented by [Christoffersen, Heston, and Jacobs (2009)](https://pure.au.dk/ws/files/17142435/rp09_34.pdf).

## European option integration

The engine evaluates

```text
P2 = 1/2 + (1/pi) integral Re[e^(-iu log K) phi(u)/(iu)] du
P1 = 1/2 + (1/pi) integral Re[e^(-iu log K) phi(u-i)/(iu phi(-i))] du
C  = S exp(-qT) P1 - K exp(-rT) P2
P  = C - S exp(-qT) + K exp(-rT)
```

from zero to infinity. The unweighted integrals are transformed for Gauss-Laguerre quadrature using the `exp(x)` compensation. The node count is configurable from 8 through 128 and defaults to 64. Puts are always obtained from put-call parity after the call calculation.

## Constraints

Normal pricing enforces every repository rule:

- positive `kappa`, `theta`, `sigma`, and `v0` for both factors;
- `kappa_slow < kappa_fast`;
- strictly positive Feller gaps for both factors;
- each correlation strictly inside `(-1, 1)`;
- `rho_slow^2 + rho_fast^2 < 1`.

The joint correlation disk is preserved because it is part of the validated handoff contract. It is stricter than the separate pairwise-correlation condition in the canonical four-shock Double Heston paper and therefore should be treated as a repository convention, not a universal literature requirement. The pricing formula remains the documented separable two-factor affine construction.

## Numerical safeguards and failure behavior

- The stable reciprocal-`g`, `exp(-dT)` Little-Heston-Trap form is used.
- Complex `log1p` and `expm1` reduce cancellation in the log ratio and short-time numerator.
- The square-root branch is normalized explicitly.
- Inputs, characteristic values, exponents, and output prices must be finite.
- Spot, strike, maturity, factor states, and positive-only parameters are rejected at invalid endpoints.
- Malformed vector lengths, array ranks, mismatched shapes, invalid option types, and unsupported node counts raise clear exceptions.
- Prices are not clipped or silently replaced. The adapter checks discounted lower and upper bounds with a `1e-10` diagnostic tolerance only; it does not alter values.
- `enforce_ordering=False` is available only for the explicit factor-symmetry diagnostic. Public adapter calls enforce ordering.

## Independent numerical reference and benchmark

`src/double_heston_reference.py` independently codes the affine transform and uses adaptive `scipy.integrate.quad` integration rather than production Gauss-Laguerre nodes. It imports the shared parameter contract and constraint validator but does not import or call the production engine. It is a slow validation reference and is not connected to the ANN generation pipeline.

The frozen 36-case benchmark compares production 64-node and 96-node prices separately with the reference. All 72 production/reference comparisons passed the predeclared combined tolerance. The largest absolute differences were `8.100187187665142e-13` at 64 nodes and `5.6985527407960035e-12` at 96 nodes. Reference integration warnings and unreliable results were zero, and every method passed no-arbitrage and paired parity checks.

This agreement supports the controlled numerical implementation but does not prove universal correctness, reproduce the unavailable teammate source, or validate real-market performance. See [Independent pricing benchmark](INDEPENDENT_PRICING_BENCHMARK.md).

## Variance-state propagation

For elapsed calendar days `delta_days`, each factor state propagates as

```text
theta_i + (v0_i-theta_i) * exp(-kappa_i*delta_days/365)
```

## Limitations

- There is no source-level or numerical equivalence claim relative to the unavailable teammate engine.
- The regression fixture is generated by this implementation and detects later code drift; it is not independent market truth.
- The repository correlation disk and the literature’s separable shock construction have different provenance, as noted above.
- The benchmark covers controlled fixtures and selected near-boundary cases, not every extreme allowed parameter combination.
- Synthetic recovery does not establish unique identification or performance on real NIFTY surfaces.
- The freeze decision is `NEEDS_BOUNDS_REVIEW`; provisional parameter ranges require financial/domain review before full ANN research generation.
