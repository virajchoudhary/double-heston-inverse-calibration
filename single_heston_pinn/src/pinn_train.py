#!/usr/bin/env python3
"""Train the parameter-conditioned Heston PINN and score it against the exact model."""

from __future__ import annotations

import json
import math
import time
from functools import partial
from dataclasses import dataclass, asdict, replace

import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

import pinn_heston_core as C


@dataclass
class TrainConfig:
    collocation_points: int = 18000     # the study's 14k-20k budget
    anchor_points: int = 18000
    width: int = 160
    depth: int = 5
    steps: int = 16000
    learning_rate: float = 2.0e-3
    final_learning_rate: float = 2.0e-5
    warmup: int = 400
    weight_pde: float = 1.0
    weight_anchor: float = 1.0
    weight_calendar: float = 0.05
    weight_butterfly: float = 0.05
    huber_delta: float = 0.10
    resample_every: int = 0             # 0 = fixed collocation set, as specified
    seed: int = 0
    label: str = "physics_and_anchor"


def _mx(a):
    return mx.array(np.asarray(a, dtype=np.float32))


def make_collocation_tensors(points):
    return {k: _mx(points[k]) for k in
            ("x", "variance", "maturity", "kappa", "theta", "sigma", "rho")}


def build_loss(model, norm, cfg, use_anchor):
    delta = cfg.huber_delta

    def huber(r):
        a = mx.abs(r)
        return mx.where(a <= delta, 0.5 * r * r, delta * (a - 0.5 * delta)) / (0.5 * delta * delta)

    def loss_fn(col, anchor_feats, anchor_target):
        residual, diag = C.heston_residual(
            model, col["x"], col["variance"], col["maturity"],
            col["kappa"], col["theta"], col["sigma"], col["rho"], norm)
        weight = C.price_relevance_weight(col["x"], diag)
        norm_w = mx.sum(weight)
        parts = {
            "pde": mx.sum(weight * huber(residual)) / norm_w,
            "calendar": mx.sum(weight * C.calendar_penalty(diag)) / norm_w,
            "butterfly": mx.sum(weight * C.butterfly_penalty(col["x"], diag)) / norm_w,
        }
        total = (cfg.weight_pde * parts["pde"]
                 + cfg.weight_calendar * parts["calendar"]
                 + cfg.weight_butterfly * parts["butterfly"])
        if use_anchor:
            g = model(anchor_feats)
            parts["anchor"] = mx.mean((g - anchor_target) ** 2)
            total = total + cfg.weight_anchor * parts["anchor"]
        return total

    return loss_fn


def train(cfg: TrainConfig, collocation, anchor, box, verbose=True, model=None):
    mx.random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    norm = C.Normaliser.from_box(box)
    if model is None:
        model = C.ImpliedVarianceNet(width=cfg.width, depth=cfg.depth)
    mx.eval(model.parameters())

    col = make_collocation_tensors(collocation)
    use_anchor = cfg.weight_anchor > 0 and anchor is not None
    if use_anchor:
        keep = anchor["usable"]
        a_feats = C.build_features(
            _mx(anchor["x"][keep]), _mx(anchor["variance"][keep]), _mx(anchor["maturity"][keep]),
            _mx(anchor["kappa"][keep]), _mx(anchor["theta"][keep]),
            _mx(anchor["sigma"][keep]), _mx(anchor["rho"][keep]), norm)
        a_target = _mx(anchor["g_target"][keep])
        mx.eval(a_feats, a_target)
    else:
        a_feats = mx.zeros((1, C.FEATURES)); a_target = mx.zeros((1,))

    loss_fn = build_loss(model, norm, cfg, use_anchor)
    schedule = optim.join_schedules(
        [optim.linear_schedule(cfg.learning_rate * 0.02, cfg.learning_rate, cfg.warmup),
         optim.cosine_decay(cfg.learning_rate, max(1, cfg.steps - cfg.warmup), cfg.final_learning_rate)],
        [cfg.warmup])
    opt = optim.Adam(learning_rate=schedule)
    value_and_grad = nn.value_and_grad(model, loss_fn)

    state = [model.state, opt.state, mx.random.state]

    @partial(mx.compile, inputs=state, outputs=state)
    def step(col_x, col_v, col_t, col_k, col_th, col_s, col_r, af, at):
        packed = {"x": col_x, "variance": col_v, "maturity": col_t,
                  "kappa": col_k, "theta": col_th, "sigma": col_s, "rho": col_r}
        loss, grads = value_and_grad(packed, af, at)
        opt.update(model, grads)
        return loss

    history = []
    start = time.time()
    for i in range(cfg.steps):
        loss = step(col["x"], col["variance"], col["maturity"], col["kappa"],
                    col["theta"], col["sigma"], col["rho"], a_feats, a_target)
        mx.eval(state, loss)
        if i % 250 == 0 or i == cfg.steps - 1:
            history.append({"step": i, "loss": float(loss), "seconds": time.time() - start})
            if verbose and (i % 2000 == 0 or i == cfg.steps - 1):
                print("    step %6d  loss %.6e  (%.0fs)" % (i, float(loss), time.time() - start), flush=True)
    return model, history


# --------------------------------------------------------------------------
# Scoring against the exact model
# --------------------------------------------------------------------------

def evaluate(model, box, n=40000, seed=999, traded_z=3.0, traded_min_days=7.0):
    """Implied-volatility and price error of the network against exact Heston."""
    norm = C.Normaliser.from_box(box)
    points = C.sample_anchor(n, box, seed=seed)
    target = C.anchor_targets(points, g_limit=C.G_LIMIT)
    keep = target["usable"]
    g_hat = np.asarray(model(C.build_features(
        _mx(points["x"]), _mx(points["variance"]), _mx(points["maturity"]),
        _mx(points["kappa"]), _mx(points["theta"]), _mx(points["sigma"]),
        _mx(points["rho"]), norm)), dtype=np.float64)
    backbone = points["maturity"] * target["vbar"]
    w_hat = backbone * np.exp(2.0 * g_hat)
    iv_hat = np.sqrt(w_hat / points["maturity"])
    iv_true = np.sqrt(target["total_variance"] / points["maturity"])
    price_hat = C.black76_normalised(points["x"], w_hat)
    price_true = target["price"]

    traded = keep & (np.abs(points["z"]) <= traded_z) & (points["maturity"] * 365.0 >= traded_min_days)

    def block(mask):
        if mask.sum() == 0:
            return {"points": 0}
        div = iv_hat[mask] - iv_true[mask]
        dp = price_hat[mask] - price_true[mask]
        return {
            "points": int(mask.sum()),
            "iv_rmse": float(np.sqrt(np.mean(div ** 2))),
            "iv_mae": float(np.mean(np.abs(div))),
            "iv_max": float(np.max(np.abs(div))),
            "iv_p99": float(np.percentile(np.abs(div), 99)),
            "price_rmse_per_strike": float(np.sqrt(np.mean(dp ** 2))),
            "price_max_per_strike": float(np.max(np.abs(dp))),
        }

    return {"traded_region": block(traded), "full_box": block(keep),
            "usable_fraction": float(keep.mean())}


def pde_residual_score(model, box, collocation):
    norm = C.Normaliser.from_box(box)
    col = make_collocation_tensors(collocation)
    residual, diag = C.heston_residual(
        model, col["x"], col["variance"], col["maturity"],
        col["kappa"], col["theta"], col["sigma"], col["rho"], norm)
    r = np.asarray(residual, dtype=np.float64)
    om = np.asarray(C.price_relevance_weight(col["x"], diag), dtype=np.float64)
    cal = np.asarray(C.calendar_penalty(diag), dtype=np.float64)
    bfly = np.asarray(C.butterfly_penalty(col["x"], diag), dtype=np.float64)
    core = om > 0.25
    return {
        "collocation_points": int(r.size),
        "residual_rms_vega_weighted": float(np.sqrt(np.sum(om * r ** 2) / np.sum(om))),
        "residual_rms_price_relevant_core": float(np.sqrt(np.mean(r[core] ** 2))) if core.any() else float("nan"),
        "price_relevant_core_points": int(core.sum()),
        "residual_rms_unweighted": float(np.sqrt(np.mean(r ** 2))),
        "residual_median_abs": float(np.median(np.abs(r))),
        "residual_p99_abs": float(np.percentile(np.abs(r), 99)),
        "calendar_violation_fraction": float((cal > 0).mean()),
        "butterfly_violation_fraction": float((bfly > 0).mean()),
    }
