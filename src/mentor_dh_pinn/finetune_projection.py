"""Semi-supervised projection fine-tuning for the unified Double Heston calibrator.

The diagnosis this fixes
------------------------
The encoder was trained with a pure RECOVERY objective: given a surface produced by Double
Heston, return the parameters that produced it. Real surfaces are not produced by Double
Heston, so on real data there is no p* to recover -- only a PROJECTION, the parameter vector
minimising fit error under a misspecified model. Measured consequence on real ADANIPOWER
quotes: the projection optimum sits a median 2.60 sd from the predicted mu_z, and 10.10 sd
in v0_total, with 43% of coordinates beyond 3 sd. The network is confidently wrong.

The objective triad
-------------------
    L = w_anchor * L_recovery(synthetic)      keeps the physical prior anchored
      + w_proj   * L_projection(real)         the objective real data actually needs
      + w_sc     * L_negative_ELBO(real)      explicit likelihood and Gaussian-prior KL

The projection and self-consistency terms use UNLABELLED market quotes. They are possible
only because the exact Fourier engine is differentiable: fit error is computable without
knowing any p*.

Numerical realities of this pricer, all measured, all handled below
-------------------------------------------------------------------
* The characteristic function genuinely overflows to NaN for extreme parameters -- the
  reference NumPy engine raises there. Non-finite prices on observed quotes reject the
  batch. Padding is excluded, and rejected losses never update the running loss scales.
* A finite loss can still produce non-finite gradients through the Fourier integrals, so
  gradients are checked before every optimiser step.
* Deep-OTM prices reach ~1e-8 of spot. Residuals are vega-weighted, which is the
  first-order equivalent of working in implied-volatility space and keeps worthless quotes
  from dominating.
* float64 throughout: the P1 - P2 difference cancels catastrophically at long maturity.
"""

from __future__ import annotations

import math
from typing import Callable

import torch
from torch import Tensor, nn

try:                                     # the class is Lightning-shaped either way
    import pytorch_lightning as pl
    _Base = pl.LightningModule
    _HAS_LIGHTNING = True
except ImportError:                      # runnable without the dependency
    _HAS_LIGHTNING = False

    class _Base(nn.Module):              # minimal shim with the same surface
        def __init__(self):
            super().__init__()
            self._logs: dict[str, float] = {}

        def log(self, name, value, **kw):
            self._logs[name] = float(value.detach() if torch.is_tensor(value) else value)

        def log_dict(self, d, **kw):
            for k, v in d.items():
                self.log(k, v)

        def clip_gradients(self, optimizer, gradient_clip_val=None,
                           gradient_clip_algorithm="norm"):
            if gradient_clip_val:
                torch.nn.utils.clip_grad_norm_(self.parameters(), gradient_clip_val)


_LOG_2PI = math.log(2.0 * math.pi)


# ----------------------------------------------------------------------- losses
def gaussian_nll_latent(z_true: Tensor, mu: Tensor, L: Tensor) -> Tensor:
    """-log N(z* | mu, L L^T) via the Cholesky factor. Sigma is never inverted."""
    d = (z_true - mu).unsqueeze(-1)
    sol = torch.linalg.solve_triangular(L, d, upper=False)
    quad = (sol ** 2).sum((-2, -1))
    logdet = 2.0 * torch.log(torch.diagonal(L, dim1=-2, dim2=-1)).sum(-1)
    return 0.5 * (quad + logdet + z_true.shape[-1] * _LOG_2PI)


def log_gauss_latent(z: Tensor, mu: Tensor, L: Tensor) -> Tensor:
    return -gaussian_nll_latent(z, mu, L)


class ProjectionFineTuner(_Base):
    """Fine-tunes a pretrained set-transformer calibrator on the projection objective."""

    def __init__(
        self,
        encoder: nn.Module,
        bijection: Callable[[Tensor], list[Tensor]],
        exact_fourier_pricer: Callable[..., Tensor],
        z_mean: Tensor,
        z_sd: Tensor,
        *,
        w_anchor: float = 1.0,
        w_projection: float = 1.0,
        w_self_consistency: float = 0.1,
        n_consistency_draws: int = 4,
        lr: float = 5e-5,
        warmup_steps: int = 150,
        max_steps: int = 3000,
        grad_clip: float = 1.0,
        unpriceable_penalty: float = 10.0,
        node_count: int = 48,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.bijection = bijection
        self.exact_fourier_pricer = exact_fourier_pricer
        self.hp = dict(w_anchor=w_anchor, w_projection=w_projection,
                       w_self_consistency=w_self_consistency,
                       n_consistency_draws=n_consistency_draws, lr=lr,
                       warmup_steps=warmup_steps, max_steps=max_steps,
                       grad_clip=grad_clip, unpriceable_penalty=unpriceable_penalty,
                       node_count=node_count)
        self.hp["self_consistency_objective"] = "negative_elbo_gaussian_prior"
        if n_consistency_draws < 1 and w_self_consistency != 0:
            raise ValueError("negative ELBO requires at least one Monte Carlo draw")
        if _HAS_LIGHTNING:
            self.save_hyperparameters(ignore=["encoder", "bijection", "exact_fourier_pricer"])
        # Fixed affine standardisation of the latent coordinates, from the TRAINING set.
        # Without it one badly scaled coordinate dominates the objective; that defect cost a
        # full training run earlier in this project.
        self.register_buffer("z_mean", z_mean)
        self.register_buffer("z_sd", z_sd.clamp(min=1e-8))
        # A diagonal Gaussian approximation to the synthetic training prior in latent
        # coordinates. Its first two moments do not specify the true sampling distribution.
        self.register_buffer("prior_mean", z_mean.clone())
        self.register_buffer("prior_sd", z_sd.clamp(min=1e-8).clone())
        self.skipped = 0
        self._loss_ema: dict[str, float] = {}

    # ------------------------------------------------------------------ helpers
    def _std(self, z: Tensor) -> Tensor:
        return (z - self.z_mean) / self.z_sd

    def _encode(self, batch: dict) -> tuple[Tensor, Tensor]:
        """Set-transformer only. Refinement is NOT unrolled here: the point of the fine-tune
        is to move mu_z into the right basin, not to lean harder on the optimiser."""
        h, pad = self.encoder.encode(batch)
        p, _ = self.encoder.tokens_forward(h, pad, batch["mask"].shape[0])
        return self.encoder.gaussian_head(p)

    def _price(self, z: Tensor, batch: dict) -> Tensor:
        params = torch.stack(self.bijection(z), dim=-1)
        active = batch["mask"] > 0
        # Padding must not enter a logarithm or Fourier integral with invalid geometry.
        geometry = [torch.where(active, batch[key], batch[key].new_full((), fill))
                    for key, fill in (("spot", 1.), ("strike", 1.), ("tau", .1),
                                      ("rate", 0.), ("carry", 0.))]
        return self.exact_fourier_pricer(
            params, *geometry, node_count=self.hp["node_count"])

    # ------------------------------------------------------- 1. anchor (synthetic)
    def recovery_loss(self, batch: dict) -> tuple[Tensor, dict]:
        """Supervised Gaussian NLL against p*, plus a standardised Huber on the location.

        This is the only thing stopping the fine-tune from walking to a parameter region
        that fits the real corpus and nothing else. Its validation value is monitored so
        catastrophic forgetting is visible rather than inferred.
        """
        mu, L = self._encode(batch)
        nll = gaussian_nll_latent(batch["z_true"], mu, L).mean()
        hub = torch.nn.functional.huber_loss(self._std(mu), self._std(batch["z_true"]))
        return nll + hub, {"anchor_nll": nll.detach(), "anchor_huber": hub.detach()}

    # ------------------------------------------- 2-4. projection (real, unlabelled)
    def projection_loss(self, batch: dict, mu: Tensor) -> tuple[Tensor, dict]:
        """Vega-weighted fit error between exact model prices and observed market quotes.

            L = mean_i [ (C_model,i(p_hat) - C_real,i) / vega_i ]^2

        Vega comes from the market quote and is DETACHED: it is a weight, not a quantity to
        differentiate. Dividing by it makes the residual a first-order implied-volatility
        error without putting a non-differentiable root-find in the graph.
        """
        active = batch["mask"] > 0
        px = self._price(mu, batch)
        frac_bad = ((~torch.isfinite(px)) & active).sum() / active.sum().clamp(min=1)
        price, vega = batch["price"][active], batch["vega"][active].detach()
        if (not active.any() or not torch.isfinite(px[active]).all()
                or not torch.isfinite(price).all() or not torch.isfinite(vega).all()
                or (vega <= 0).any()):
            loss = mu.new_full((), float("inf"))
        else:
            resid = (px[active] - price) / vega.clamp(min=1e-8)
            loss = resid.square().mean()
        return loss, {"proj_iv_rmse": torch.sqrt(loss.detach()),
                      "proj_unpriceable": frac_bad.detach()}

    # ----------------------------------------- 5. self-consistency (real, unlabelled)
    def self_consistency_loss(self, batch: dict, mu: Tensor, L: Tensor) -> tuple[Tensor, dict]:
        """Negative ELBO: E_q[-log p(quotes|z)] + KL(q || Gaussian prior).

        The public name is retained for existing callers; this is no longer variance of
        log evidence. Gaussian KL is analytic, including the entropy term -log det L.
        Only the likelihood expectation uses reparameterised Monte Carlo draws. Fixed
        likelihood normalisers are omitted. The likelihood sums observed quotes per
        surface, then surfaces are averaged; quote_sigma defines its noise assumption.

        On a flat likelihood the optimum covariance equals the prior covariance: a
        narrower posterior widens and a wider one shrinks. Requiring a negative width
        gradient everywhere would be incorrect, especially on identified directions.
        The diagonal prior and Gaussian posterior are approximations. Combining this
        loss with anchor/projection losses and EMA weights is not exact Bayesian inference.
        """
        K = self.hp["n_consistency_draws"]
        if K < 1:
            raise ValueError("negative ELBO requires at least one Monte Carlo draw")
        active = batch["mask"] > 0
        sig = batch["quote_sigma"].detach()
        diag = torch.diagonal(L, dim1=-2, dim2=-1)
        invalid = mu.new_full((), float("inf"))
        if (not active.any(-1).all() or not torch.isfinite(mu).all()
                or not torch.isfinite(L).all() or (diag <= 0).any()
                or not torch.isfinite(sig[active]).all() or (sig[active] <= 0).any()
                or not torch.isfinite(batch["price"][active]).all()):
            return invalid, {"sc_invalid": mu.new_ones(())}
        # D is diagonal prior covariance; tr(D^-1 L L^T) is a squared row-scaled norm.
        kl = 0.5 * (((L / self.prior_sd.unsqueeze(-1)).square().sum((-2, -1)))
                    + ((mu - self.prior_mean) / self.prior_sd).square().sum(-1)
                    - mu.shape[-1] + 2 * self.prior_sd.log().sum()
                    - 2 * diag.log().sum(-1))
        nll = []
        for _ in range(K):
            eps = torch.randn_like(mu)
            z_k = mu + (L @ eps.unsqueeze(-1)).squeeze(-1)
            px = self._price(z_k, batch)
            if not torch.isfinite(px[active]).all():
                return invalid, {"sc_invalid": mu.new_ones(())}
            # Index before division: NaNs in padded targets/noise never enter arithmetic.
            r = torch.zeros_like(px).masked_scatter(
                active, (px[active] - batch["price"][active]) / sig[active].clamp(min=1e-8))
            nll.append(0.5 * r.square().sum(-1))
        expected_nll = torch.stack(nll).mean(0)
        loss = (kl + expected_nll).mean()
        if not torch.isfinite(loss):
            return invalid, {"sc_invalid": mu.new_ones(())}
        return loss, {"sc_kl": kl.detach().mean(),
                      "sc_expected_nll": expected_nll.detach().mean(),
                      "sc_invalid": mu.new_zeros(())}

    # ------------------------------------------------------------------- steps
    def _balance(self, key: str, value: Tensor) -> Tensor:
        """Divide each objective by a running estimate of its own scale.

        Use an EMA of absolute magnitude because a density NLL can be negative. Rejected
        non-finite objectives must not poison the scale for every subsequent training step.
        """
        if not torch.isfinite(value):
            return value
        x = abs(float(value.detach()))
        prev = self._loss_ema.get(key)
        self._loss_ema[key] = x if prev is None else 0.98 * prev + 0.02 * x
        return value / max(self._loss_ema[key], 1e-12)

    def compute_losses(self, syn: dict, real: dict) -> tuple[Tensor, dict]:
        zero = self.z_mean.new_zeros(())
        anchor = proj = sc = zero
        la, lp, ls = {}, {}, {}
        if self.hp["w_anchor"] != 0:
            anchor, la = self.recovery_loss(syn)
        if self.hp["w_projection"] != 0 or self.hp["w_self_consistency"] != 0:
            mu, L = self._encode(real)
            if self.hp["w_projection"] != 0:
                proj, lp = self.projection_loss(real, mu)
            if self.hp["w_self_consistency"] != 0:
                sc, ls = self.self_consistency_loss(real, mu, L)
        terms = (("anchor", anchor, self.hp["w_anchor"]),
                 ("proj", proj, self.hp["w_projection"]),
                 ("sc", sc, self.hp["w_self_consistency"]))
        if any(not torch.isfinite(value) for _, value, weight in terms if weight != 0):
            loss = zero.new_full((), float("inf"))
        else:
            loss = sum((weight * self._balance(key, value)
                        for key, value, weight in terms if weight != 0), zero)
        return loss, {"anchor": anchor.detach(), "projection": proj.detach(),
                      "self_consistency": sc.detach(), **la, **lp, **ls}

    def training_step(self, batch: dict, batch_idx: int):
        loss, logs = self.compute_losses(batch["syn"], batch["real"])
        if not torch.isfinite(loss):
            # Skip rather than propagate: a NaN here writes NaN into the weights and the
            # model never recovers. This exact failure cost six epochs earlier.
            self.skipped += 1
            self.log("train/skipped", float(self.skipped))
            return None
        self.log_dict({f"train/{k}": v for k, v in logs.items()})
        self.log("train/loss", loss.detach(), prog_bar=True)
        return loss

    @torch.no_grad()
    def validation_step(self, batch: dict, batch_idx: int) -> dict:
        out = {}
        if "syn" in batch:
            syn = batch["syn"]; mu, L = self._encode(syn)
            sd = torch.sqrt(torch.diagonal(L @ L.transpose(-1, -2), dim1=-2, dim2=-1))
            res = (syn["z_true"] - mu).abs()
            out["val/syn_z_mae"] = (self._std(mu) - self._std(syn["z_true"])).abs().mean()
            out["val/syn_cov90"] = (res <= 1.6448536 * sd).to(mu.dtype).mean()
            # forgetting monitor: if this climbs while projection falls, the fine-tune is
            # trading away the physical prior that makes the model general
            out["val/syn_anchor_nll"] = gaussian_nll_latent(syn["z_true"], mu, L).mean()
        real = batch["real"]; mu, L = self._encode(real)
        proj, lp = self.projection_loss(real, mu)
        out["val/real_projection"] = proj
        out["val/real_iv_rmse"] = lp["proj_iv_rmse"]
        self.log_dict(out)
        return out

    # ------------------------------------------------------------- optimisation
    def on_before_optimizer_step(self, optimizer) -> None:
        """A finite loss can still yield non-finite gradients through the Fourier integrals."""
        total = torch.nn.utils.clip_grad_norm_(self.parameters(), float("inf"))
        if not torch.isfinite(total):
            optimizer.zero_grad(set_to_none=True)
            self.skipped += 1
            self.log("train/nonfinite_grad", float(self.skipped))
            return
        self.log("train/grad_norm_preclip", total)

    def configure_gradient_clipping(self, optimizer, gradient_clip_val=None,
                                    gradient_clip_algorithm=None) -> None:
        self.clip_gradients(optimizer,
                            gradient_clip_val=gradient_clip_val or self.hp["grad_clip"],
                            gradient_clip_algorithm=gradient_clip_algorithm or "norm")

    def lr_lambda(self, step: int) -> float:
        """Linear warmup then cosine decay.

        Conservative for three reasons: this fine-tunes an already competent encoder, so
        large steps destroy more than they build; gradients arrive through a Gauss-Laguerre
        quadrature whose curvature varies by orders of magnitude across the parameter box;
        and the real corpus, while far larger than the evaluation sets, is still small enough
        that overfitting is the default outcome.
        """
        w, t = self.hp["warmup_steps"], max(self.hp["max_steps"], 1)
        if step < w:
            return (step + 1) / max(w, 1)
        prog = min((step - w) / max(t - w, 1), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * prog))

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.hp["lr"], weight_decay=1e-6)
        return {"optimizer": opt,
                "lr_scheduler": {"scheduler": torch.optim.lr_scheduler.LambdaLR(
                    opt, self.lr_lambda), "interval": "step"}}
