"""Unified physics-informed set calibrator for Double Heston.

One model, one call:

    arbitrary set of option quotes
      -> permutation-invariant quote encoder
      -> ten parameter-query tokens, each cross-attending to the quotes
      -> location mu_z and a full 10x10 covariance Sigma_z
      -> unrolled damped Gauss-Newton refinement against the EXACT Fourier pricer
      -> final parameters, uncertainty, identifiability, OOD status

Design notes, each answering a measured defect of the previous architecture:

* **Quotes are a set, not a 45-vector.** Strike and maturity are token *features*, so one
  expiry, five expiries, or ninety irregular quotes are all valid inputs and no market has
  to be interpolated onto a training lattice.
* **Ten parameter tokens, not one pooled regression.** Measured relative sensitivity
  |d log C / d log p| spans 30-300x across the ten parameters; a single pooled head lets the
  loud ones dominate. Each parameter gets its own cross-attention path, so theta and kappa
  can attend to long maturities without competing with v0 for the same pooled vector. The
  maturity specialisation is therefore learned and soft, not a hard partition.
* **Uncertainty is an output, not an afterthought.** Double Heston is practically
  non-identifiable on many surfaces; a point estimate cannot say so. The full covariance
  also replaces the global scalar ridge: directions the data pins get a stiff prior,
  directions it does not get a loose one, per surface.
* **The exact engine is inside the model.** `torch_pricer` matches the production NumPy
  engine to 3e-15 and is differentiable, so refinement is part of `forward()` and the
  reported latency is the latency of the answer actually used.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

from .params_v2 import decode
from .torch_pricer import price_call

N_PARAMS = 10
PARAM_NAMES = ("kappa_slow", "theta_slow", "sigma_slow", "rho_slow", "v0_slow",
               "kappa_fast", "theta_fast", "sigma_fast", "rho_fast", "v0_fast")


def fourier_features(x: torch.Tensor, n_bands: int, scale: float) -> torch.Tensor:
    """Continuous positional encoding, so maturity and moneyness are real coordinates."""
    freqs = scale * (2.0 ** torch.arange(n_bands, dtype=x.dtype, device=x.device))
    a = x.unsqueeze(-1) * freqs                       # (..., n_bands)
    return torch.cat([torch.sin(a), torch.cos(a)], dim=-1)   # (..., 2 n_bands)


def quote_features(spot, strike, tau, rate, carry, price, mask):
    """Scale-free per-quote tokens. Nothing here assumes a grid."""
    eps = 1e-12
    fwd = spot * torch.exp((rate - carry) * tau)
    disc = torch.exp(-rate * tau)
    x = torch.log(torch.clamp(strike, min=eps) / torch.clamp(fwd, min=eps))   # log-fwd-moneyness
    lt = torch.log(torch.clamp(tau, min=1e-6))
    # normalise by the option's own scale so a 3-year and a 7-day quote are comparable
    norm_price = torch.clamp(price, min=eps) / torch.clamp(fwd * disc, min=eps)
    intrinsic = torch.clamp(1.0 - torch.exp(x), min=0.0)
    time_value = torch.clamp(norm_price - intrinsic, min=eps)
    base = torch.stack([x, lt, torch.log(norm_price), torch.log(time_value),
                        norm_price, intrinsic, rate, carry,
                        torch.sqrt(torch.clamp(tau, min=0.0)),
                        x / torch.sqrt(torch.clamp(tau, min=1e-6))], dim=-1)
    ff = torch.cat([fourier_features(x, N_BANDS, 2.0),
                    fourier_features(lt, N_BANDS, 1.0)], dim=-1)
    feat = torch.cat([base, ff], dim=-1)
    return feat * mask.unsqueeze(-1)


N_BANDS = 6
FEATURE_DIM = 10 + 2 * (2 * N_BANDS)      # 10 base + 12 moneyness + 12 maturity


class SetBlock(nn.Module):
    """Pre-norm self-attention over quotes, with key padding mask."""

    def __init__(self, d, heads, ff_mult=4, dropout=0.0):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.att = nn.MultiheadAttention(d, heads, batch_first=True, dropout=dropout)
        self.ff = nn.Sequential(nn.Linear(d, ff_mult * d), nn.GELU(), nn.Linear(ff_mult * d, d))

    def forward(self, h, pad):
        y = self.n1(h)
        a, _ = self.att(y, y, y, key_padding_mask=pad, need_weights=False)
        h = h + a
        return h + self.ff(self.n2(h))


class ParameterBlock(nn.Module):
    """One communication round: tokens read the quotes, then read each other."""

    def __init__(self, d, heads, ff_mult=4):
        super().__init__()
        self.nq, self.np_, self.nf = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.cross = nn.MultiheadAttention(d, heads, batch_first=True)
        self.self_ = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d, ff_mult * d), nn.GELU(), nn.Linear(ff_mult * d, d))

    def forward(self, p, x, pad):
        a, w = self.cross(self.nq(p), x, x, key_padding_mask=pad, need_weights=True,
                          average_attn_weights=True)
        p = p + a
        b, _ = self.self_(self.np_(p), self.np_(p), self.np_(p), need_weights=False)
        p = p + b
        return p + self.ff(self.nf(p)), w


class UnifiedCalibrator(nn.Module):
    def __init__(self, d_model=128, heads=4, enc_blocks=4, rounds=3,
                 share_rounds=True, refine_steps=3, node_count=48):
        super().__init__()
        self.rounds, self.refine_steps, self.node_count = rounds, refine_steps, node_count
        self.embed = nn.Sequential(nn.Linear(FEATURE_DIM, d_model), nn.GELU(),
                                   nn.Linear(d_model, d_model))
        self.enc = nn.ModuleList([SetBlock(d_model, heads) for _ in range(enc_blocks)])
        self.tokens = nn.Parameter(torch.randn(N_PARAMS, d_model) * 0.02)
        blocks = [ParameterBlock(d_model, heads)]
        if not share_rounds:
            blocks += [ParameterBlock(d_model, heads) for _ in range(rounds - 1)]
        self.pblocks = nn.ModuleList(blocks)
        self.share_rounds = share_rounds
        self.norm = nn.LayerNorm(d_model)
        self.head_mu = nn.Linear(d_model, 1)                       # one scalar per token
        self.head_diag = nn.Linear(d_model, 1)                     # log-diagonal of L
        nn.init.zeros_(self.head_diag.weight); nn.init.constant_(self.head_diag.bias, 0.55)
        self.head_off = nn.Linear(d_model * 2, 1)                  # strictly-lower entries
        nn.init.zeros_(self.head_off.weight); nn.init.zeros_(self.head_off.bias)
        self.noise_embed = nn.Sequential(nn.Linear(1, d_model), nn.GELU(),
                                         nn.Linear(d_model, d_model))
        self.double()

    # ---------------------------------------------------------------- encoder
    def encode(self, batch):
        mask = batch["mask"]
        pad = mask < 0.5
        f = quote_features(batch["spot"], batch["strike"], batch["tau"],
                           batch["rate"], batch["carry"], batch["price"], mask)
        h = self.embed(f)
        h = h + self.noise_embed(torch.log(torch.clamp(
            batch["noise_level"], min=1e-6)).unsqueeze(-1)).unsqueeze(1)
        for blk in self.enc:
            h = blk(h, pad)
        return h, pad

    def tokens_forward(self, h, pad, batch_size):
        p = self.tokens.unsqueeze(0).expand(batch_size, -1, -1)
        attn = []
        for r in range(self.rounds):
            blk = self.pblocks[0] if self.share_rounds else self.pblocks[r]
            p, w = blk(p, h, pad)
            attn.append(w)
        return self.norm(p), attn

    def gaussian_head(self, p):
        """mu_z and a positive-definite Sigma_z = L L^T + eps I, L lower-triangular."""
        B = p.shape[0]
        mu = self.head_mu(p).squeeze(-1)                                  # (B, 10)
        diag = torch.nn.functional.softplus(self.head_diag(p).squeeze(-1)) + 1e-4
        i, j = torch.tril_indices(N_PARAMS, N_PARAMS, offset=-1)
        pair = torch.cat([p[:, i, :], p[:, j, :]], dim=-1)                # (B, 45, 2d)
        off = self.head_off(pair).squeeze(-1)
        L = torch.zeros(B, N_PARAMS, N_PARAMS, dtype=p.dtype, device=p.device)
        L[:, torch.arange(N_PARAMS), torch.arange(N_PARAMS)] = diag
        L[:, i, j] = off
        return mu, L

    # ------------------------------------------------------- physics refinement
    def _prices(self, z, batch):
        return price_call(torch.stack(decode(z), dim=-1), batch["spot"], batch["strike"],
                          batch["tau"], batch["rate"], batch["carry"],
                          node_count=self.node_count)

    def refinement_objective(self, z, mu, L, batch, prior_weight=1.0):
        """Per-surface sum of squared quote-noise residuals plus the encoder prior.

        A failed active quote makes the whole surface objective infinite. No quote is
        silently removed. Only fields in this calibration batch enter the objective.
        """
        if not math.isfinite(prior_weight) or prior_weight < 0:
            raise ValueError("prior_weight must be finite and nonnegative")
        active = batch["mask"] > 0.5
        pred = self._prices(z, batch)
        residual = torch.where(active, (pred - batch["price"]) / quote_scale(batch), 0.0)
        value = residual.square().sum(-1)
        if prior_weight:
            delta = torch.linalg.solve_triangular(L, (z - mu).unsqueeze(-1), upper=False)
            value = value + prior_weight * delta.square().sum((-2, -1))
        good = ((~active) | torch.isfinite(pred)).all(-1) & torch.isfinite(value)
        return torch.where(good, value, torch.full_like(value, float("inf")))

    def refine(self, z0, L, batch, steps=None, create_graph=False, prior_weight=1.0):
        """Damped Gauss-Newton with per-surface objective acceptance/backtracking.

        The default encoder prior is unchanged. An explicitly requested zero prior is
        useful for noiseless synthetic calibration, where exact reconstruction is the
        objective. The returned history contains the actual post-acceptance mean of
        ``refinement_objective``, including the prior, for every attempted step.
        """
        if not math.isfinite(prior_weight) or prior_weight < 0:
            raise ValueError("prior_weight must be finite and nonnegative")
        steps = self.refine_steps if steps is None else steps
        mu = z0
        z = z0
        active = batch["mask"] > 0.5
        w = active.to(z.dtype) / quote_scale(batch)
        eye = torch.eye(N_PARAMS, dtype=z.dtype, device=z.device)
        if prior_weight:
            Li = torch.linalg.solve_triangular(L, eye.expand_as(L), upper=False)
            Sinv = prior_weight * (Li.transpose(-1, -2) @ Li)
        else:
            Sinv = torch.zeros_like(L)
        hist = []
        damping = torch.full((len(z),), 1e-6, dtype=z.dtype, device=z.device)
        max_step = 1.5
        for _ in range(max(steps, 0)):
            f = lambda zz: self._prices(zz, batch)
            J = _batched_jacobian(f, z, create_graph)
            pred = f(z)
            good = ((~active) | (torch.isfinite(pred) & torch.isfinite(J).all(-1))).all(-1)
            # Neutralisation is solely for the batched solve: a bad surface is frozen,
            # never accepted using a smaller subset of its calibration quotes.
            usable = active & good.unsqueeze(-1)
            r = torch.where(usable, pred - batch["price"], 0.0) * w
            Jw = torch.where(usable.unsqueeze(-1), J, 0.0) * w.unsqueeze(-1)
            H = Jw.transpose(-1, -2) @ Jw + Sinv
            scale = torch.diagonal(H, dim1=-2, dim2=-1).mean(-1).detach().clamp(min=1.0)
            A = H + (damping * scale)[:, None, None] * eye
            g = Jw.transpose(-1, -2) @ r.unsqueeze(-1) + Sinv @ (z - mu).unsqueeze(-1)
            good = good & torch.isfinite(A).all((-2, -1)) & torch.isfinite(g).all((-2, -1))
            A = torch.where(good[:, None, None], A, eye)
            g = torch.where(good[:, None, None], g, 0.0)
            step, info = torch.linalg.solve_ex(A, g)
            step = step.squeeze(-1)
            good = good & (info == 0) & torch.isfinite(step).all(-1)
            step = torch.where(good[:, None], step, 0.0)
            nrm = step.norm(dim=-1, keepdim=True).clamp(min=1e-30)
            step = step * torch.clamp(max_step / nrm, max=1.0)
            with torch.no_grad():
                current = self.refinement_objective(z, mu, L, batch, prior_weight)
            accepted = torch.zeros_like(good)
            z_next = z
            for backtrack in range(10):
                candidate = z - (0.5 ** backtrack) * step
                # Acceptance is discrete; keep the accepted candidate's autograd path,
                # without retaining ten discarded pricer graphs for the line search.
                with torch.no_grad():
                    trial = self.refinement_objective(candidate, mu, L, batch, prior_weight)
                    take = good & ~accepted & torch.isfinite(trial) & (trial <= current)
                z_next = torch.where(take[:, None], candidate, z_next)
                current = torch.where(take, trial, current)
                accepted = accepted | take
                if bool((accepted | ~good).all()):
                    break
            z = z_next
            damping = torch.where(accepted, damping * 0.3, damping * 10).clamp(1e-10, 1e12)
            hist.append(float(current.mean().detach()))
        return z, hist

    def local_identifiability(self, z, batch, latent_scale=None, threshold=1e-6):
        """Local practical sensitivity of the calibration quotes, without an encoder prior.

        SVD avoids squaring the Jacobian's condition number. ``threshold`` applies to
        relative information (s/s_max)^2; it is a declared diagnostic threshold, not a
        proof of global uniqueness. Supply training-only latent standard deviations for
        comparable scaled directions. Basis vectors are in those scaled coordinates.
        """
        if not 0 < threshold < 1:
            raise ValueError("threshold must be between zero and one")
        scale = torch.ones(N_PARAMS, dtype=z.dtype, device=z.device) if latent_scale is None \
            else torch.as_tensor(latent_scale, dtype=z.dtype, device=z.device)
        if scale.shape != (N_PARAMS,) or not bool((torch.isfinite(scale) & (scale > 0)).all()):
            raise ValueError("latent_scale must contain ten finite positive values")
        J = _batched_jacobian(lambda zz: self._prices(zz, batch), z, False)
        pred = self._prices(z, batch)
        weights = 1.0 / quote_scale(batch)
        rows = []
        for i in range(len(z)):
            active = batch["mask"][i] > 0.5
            A = J[i, active] * weights[i, active, None] * scale
            if not bool(torch.isfinite(A).all() & torch.isfinite(pred[i, active]).all()):
                rows.append({"status": "numerically_unsafe", "rank": None})
                continue
            # Full right basis is needed for fewer than ten quotes; avoid forming an
            # unused quotes-by-quotes left matrix for dense market surfaces.
            _, singular, vh = torch.linalg.svd(A, full_matrices=A.shape[0] < N_PARAMS)
            singular = torch.nn.functional.pad(singular, (0, N_PARAMS - len(singular)))
            relative = (singular / singular[0].clamp(min=1e-300)).square()
            rank = int((relative > threshold).sum())
            rows.append({"status": "local_practical_sensitivity", "rank": rank,
                         "relative_information_threshold": threshold,
                         "singular_values": singular.detach().cpu().tolist(),
                         "relative_information": relative.detach().cpu().tolist(),
                         "identified_basis": vh[:rank].detach().cpu().tolist(),
                         "weak_basis": vh[rank:].detach().cpu().tolist(),
                         "latent_scale": scale.detach().cpu().tolist(),
                         "coordinates": "z = fitted_z + latent_scale * displacement",
                         "interpretation": "Local sensitivity only; weak directions need profile/set analysis."})
        return rows

    def forward(self, batch, refine_steps=None, create_graph=False, prior_weight=1.0):
        h, pad = self.encode(batch)
        p, attn = self.tokens_forward(h, pad, batch["mask"].shape[0])
        mu, L = self.gaussian_head(p)
        z, hist = self.refine(mu, L, batch, steps=refine_steps, create_graph=create_graph,
                              prior_weight=prior_weight)
        return {"mu_z": mu, "L": L, "z": z,
                "params_pre": torch.stack(decode(mu), dim=-1),
                "params": torch.stack(decode(z), dim=-1),
                "attn": attn, "residual_history": hist, "objective_history": hist,
                "covariance_scope": "L describes the encoder Gaussian around mu_z, before refinement; "
                                    "it is not a posterior covariance around refined z.",
                "history_definition": "post-acceptance mean per-surface squared-noise-residual plus prior"}


def quote_scale(batch):
    """Explicit price-unit quote sigma takes precedence over legacy relative-price noise."""
    active = batch["mask"] > 0.5
    if not bool(active.any(-1).all()):
        raise ValueError("each calibration surface must have an active quote")
    if "quote_sigma" in batch:
        scale = batch["quote_sigma"]
        if scale.shape != batch["price"].shape:
            raise ValueError("quote_sigma must have the same shape as price")
    else:
        noise = batch["noise_level"]
        if not bool((torch.isfinite(noise) & (noise >= 0)).all()):
            raise ValueError("noise_level must be finite and nonnegative")
        scale = (noise.unsqueeze(-1) * batch["price"].clamp(min=0.0)).clamp(min=1e-6) \
            + 1e-6 * batch["spot"].clamp(min=1e-12)
    if not bool(((~active) | (torch.isfinite(scale) & (scale > 0))).all()):
        raise ValueError("active quote_sigma values must be finite and positive")
    return torch.where(active, scale, torch.ones_like(scale))


def _batched_jacobian(f, z, create_graph):
    """d f / d z for a batch, forward-mode: N >> 10 so ten tangents beat N adjoints."""
    B, P = z.shape
    cols = []
    for k in range(P):
        tangent = torch.zeros_like(z)
        tangent[:, k] = 1.0
        _, jv = torch.func.jvp(f, (z,), (tangent,))
        cols.append(jv)
    J = torch.stack(cols, dim=-1)
    return J if create_graph else J.detach()
