"""A good price fit or a lucky seed must not approve failed parameter recovery."""
import importlib.util
from pathlib import Path
import sys

folder = Path(__file__).resolve().parents[1] / "scripts/mentor_dh_pinn"
sys.path.insert(0, str(folder))
spec = importlib.util.spec_from_file_location("sweep_test_module", folder / "run_literature_parameter_sweep.py")
sweep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sweep)


def test_all_seeds_must_pass_even_with_perfect_prices():
    gates = {"max_repriced_iv_rmse": .005, "max_fresh_scaled_pde_rmse": .01,
             "max_sampled_negative_convexity_points": 0}
    good = {"all_ten_pass": True, "holdout": {"repriced_iv_rmse": 0.}}
    bad = {"all_ten_pass": False, "holdout": {"repriced_iv_rmse": 0.}}
    diag = [{"scaled_pde_rmse": 0., "negative_convexity_points": 0}]
    unseen = [{"repriced_iv_rmse": 0.}]
    assert sweep.promotion_allowed([good], diag, unseen, gates)
    assert not sweep.promotion_allowed([good, bad], diag, unseen, gates)
    assert not sweep.promotion_allowed([], diag, unseen, gates)
    assert not sweep.promotion_allowed([good], diag, [{"repriced_iv_rmse": .006}], gates)


def test_ranking_cannot_prefer_nice_price_to_parameter_recovery():
    def row(name, ok, error, price):
        return {"case": name, "truth": [1.] * 10, "absolute_error": [error] * 10,
                "all_ten_pass": ok, "passed_parameters": 10 if ok else 0,
                "holdout": {"repriced_iv_rmse": price}}
    ranked = sweep.ranking([row("wrong_parameters", False, 1., 0.),
                            row("recovered", True, .01, .001)])
    assert ranked[0]["case"] == "recovered"
    assert abs(ranked[0]["median_worst_normalized_error"] - .2) < 1e-12
