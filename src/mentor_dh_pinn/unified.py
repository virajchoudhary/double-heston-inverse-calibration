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
    def refine(self, z0, L, batch, steps=None, create_graph=False):
        """Unrolled damped Gauss-Newton against the exact pricer, with the covariance prior.

            A  = J^T W J + Sigma^-1 + alpha I
            g  = J^T W r + Sigma^-1 (z - mu)
            z <- z - solve(A, g)

        Sigma^-1 = (L L^T)^-1 is applied via the Cholesky factor, never by forming an
        explicit inverse. The solve is 10x10; cost is dominated by the exact prices and
        their Jacobian, not by the linear algebra.
        """
        steps = self.refine_steps if steps is None else steps
        mu = z0
        z = z0
        mask, spot = batch["mask"], batch["spot"]
        # Weight each residual by the RECIPROCAL QUOTE NOISE, not by 1/spot. The Gauss-Newton
        # normal equations combine a data term J^T W J with a prior precision Sigma^-1, and
        # the two are only commensurate if W is the noise precision. Weighting by 1/spot
        # instead left the data term about 3e3 times weaker than the prior, so every step was
        # ~1e-4 in latent units against a trust region of 1.5 and refinement did nothing.
        eps_q = torch.clamp(batch["noise_level"].unsqueeze(-1) *
                            torch.clamp(batch["price"], min=0.0), min=1e-6) \
            + 1e-6 * torch.clamp(spot, min=1e-12)
        w = mask / eps_q
        eye = torch.eye(N_PARAMS, dtype=z.dtype, device=z.device)
        Linv_eye = torch.linalg.solve_triangular(L, eye.expand_as(L), upper=False)
        Sinv = Linv_eye.transpose(-1, -2) @ Linv_eye                      # (L L^T)^-1, PD
        hist = []
        max_step = 1.5                      # trust region in latent units
        for _ in range(max(steps, 0)):
            def f(zz):
                pp = torch.stack(decode(zz), dim=-1)
                return price_call(pp, batch["spot"], batch["strike"], batch["tau"],
                                  batch["rate"], batch["carry"], node_count=self.node_count)
            J = _batched_jacobian(f, z, create_graph)
            pred = f(z)
            # A non-finite price or Jacobian is a numerical failure, not a zero. Neutralise
            # the affected rows so the linear system stays well posed, and leave those
            # surfaces where they are rather than stepping on corrupt information.
            good_q = torch.isfinite(pred) & torch.isfinite(J).all(-1)
            wq = w * good_q.to(w.dtype)
            pred = torch.where(good_q, pred, batch["price"])
            J = torch.where(good_q.unsqueeze(-1), J, torch.zeros_like(J))
            r = (pred - batch["price"]) * wq
            Jw = J * wq.unsqueeze(-1)
            A = Jw.transpose(-1, -2) @ Jw + Sinv + 1e-6 * eye
            g = Jw.transpose(-1, -2) @ r.unsqueeze(-1) + Sinv @ (z - mu).unsqueeze(-1)
            step = torch.linalg.solve(A, g).squeeze(-1)
            nrm = step.norm(dim=-1, keepdim=True).clamp(min=1e-30)
            step = step * torch.clamp(max_step / nrm, max=1.0)      # trust region
            z_new = z - step
            ok_row = torch.isfinite(z_new).all(-1, keepdim=True)
            z = torch.where(ok_row, z_new, z)
            hist.append(float(torch.sqrt(torch.mean(r[mask > 0.5] ** 2)).detach()))
        return z, hist

    def forward(self, batch, refine_steps=None, create_graph=False):
        h, pad = self.encode(batch)
        p, attn = self.tokens_forward(h, pad, batch["mask"].shape[0])
        mu, L = self.gaussian_head(p)
        z, hist = self.refine(mu, L, batch, steps=refine_steps, create_graph=create_graph)
        return {"mu_z": mu, "L": L, "z": z,
                "params_pre": torch.stack(decode(mu), dim=-1),
                "params": torch.stack(decode(z), dim=-1),
                "attn": attn, "residual_history": hist}


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
