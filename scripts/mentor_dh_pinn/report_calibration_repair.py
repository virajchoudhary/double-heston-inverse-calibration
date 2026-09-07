#!/usr/bin/env python3
"""Build a shareable report, scientific figures and a lightweight results notebook."""
from __future__ import annotations
import json
import math
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.mentor_dh_pinn.params_v2 import CANONICAL
from src.mentor_dh_pinn.torch_pricer import price_call
from scripts.mentor_dh_pinn.evaluate_repair_validation import digest

OUT=ROOT/"outputs/calibration_repair"
PRIMARY="seed_43/repaired"
COLORS={"encoder":"#5068a9","refined":"#df8530","hybrid_polish":"#248779"}
LABELS={"encoder":"Network only","refined":"Network + 3 refinement steps","hybrid_polish":"Network + exact optimisation"}


def main():
    folders=[OUT/f"assessment_{seed}" for seed in (905101,905102,905103)]
    for folder in folders:
        if not (folder/"summary.json").exists():
            raise RuntimeError(f"Assessment not complete: {folder}")
    market=json.loads((OUT/"market_precision_checked/summary.json").read_text())
    metrics=pd.concat([pd.read_csv(f/"recovery_metrics.csv") for f in folders],ignore_index=True)
    params=pd.concat([pd.read_csv(f/"parameter_recovery.csv") for f in folders],ignore_index=True)
    truths=pd.concat([pd.read_csv(f/"synthetic_truth.csv") for f in folders],ignore_index=True)
    if metrics.duplicated(["case","geometry","noise","model","arm"]).any():
        raise AssertionError("Duplicate assessment record")
    if len(truths)!=24 or metrics.case.nunique()!=24 or len(metrics)!=864:
        raise AssertionError("Final assessment incomplete")
    # Verify frozen run provenance before rebuilding any presentation of its results.
    expected={}
    for seed in (17,29,43):
        run=json.loads((OUT/f"seed_{seed}/repaired_run.json").read_text())
        for section in ("input_sha256","code_sha256"):
            for path,sha in run[section].items():
                if path in expected and expected[path]!=sha:
                    raise AssertionError(f"Training provenance disagreement: {path}")
                expected[path]=sha
    for path,sha in expected.items():
        if digest(ROOT/path)!=sha:
            raise AssertionError(f"Frozen training input/source changed: {path}")
    selection=json.loads((OUT/"selection.json").read_text())
    if digest(ROOT/selection["primary_checkpoint"])!=selection["checkpoint_sha256"]:
        raise AssertionError("Selected checkpoint changed")
    keys=["case","geometry","noise","model","arm","parameter"]
    if len(params)!=8640 or params.duplicated(keys).any():
        raise AssertionError("Missing or duplicate parameter records")
    original=pd.read_csv(OUT/"market_validation/holdout_predictions.csv")
    checked=pd.read_csv(OUT/"market_precision_checked/holdout_predictions.csv")
    quote_keys=["model","label","quote_index"]
    if len(checked)!=6428 or checked.duplicated(quote_keys).any() or not original[quote_keys].equals(checked[quote_keys]):
        raise AssertionError("Missing, reordered or duplicate market holds")
    if not np.allclose(original.observed_price,checked.observed_price,rtol=1e-14,atol=0):
        raise AssertionError("Observed prices changed during numerical re-evaluation")
    for path,sha in market["precision_check"]["input_sha256"].items():
        if digest(ROOT/path)!=sha:raise AssertionError("Original validation evidence changed")
    (OUT/"artifact_integrity.json").write_text(json.dumps({
        "frozen_training_inputs_and_sources_verified":len(expected),
        "selected_checkpoint_hash_verified":True,"synthetic_truths":24,"assessment_records":864,
        "parameter_records":8640,"market_quote_records":6428,"duplicate_records":0,
        "original_market_evidence_hashes_preserved":True,
        "observed_quote_copy_check":"rtol 1e-14 for CSV float serialization; original files byte-unchanged"},indent=2))
    metrics.to_csv(OUT/"all_recovery_metrics.csv",index=False)
    params.to_csv(OUT/"all_parameter_recovery.csv",index=False)
    figure_dir=OUT/"figures";figure_dir.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False,
                         "figure.facecolor":"white","axes.titleweight":"bold"})
    explanations=[]
    def save(fig,name,caption):
        fig.savefig(figure_dir/name,dpi=160,bbox_inches="tight")
        plt.close(fig)
        explanations.append((name,caption))

    fig,axs=plt.subplots(1,3,figsize=(14,4))
    selected=[]
    for seed,color in zip((17,29,43),("#5068a9","#df8530","#248779")):
        directory=OUT/f"seed_{seed}"
        first=json.loads((directory/"repaired_initial.json").read_text())
        h=pd.DataFrame([{"step":0,**first}]+json.loads((directory/"repaired_history.json").read_text()))
        ck=torch.load(directory/"repaired.pt",map_location="cpu",weights_only=False)
        selected.append({"seed":seed,"step":ck["step"],**ck["metrics"]})
        for ax,col,title in zip(axs,("real_iv_rmse","syn_z_mae","syn_cov90"),
                                ("Development quote error","Synthetic latent MAE","Synthetic 90% coverage")):
            ax.plot(h.step,h[col],"o-",ms=4,label=f"seed {seed}",color=color)
            ax.set(xlabel="Training step",title=title);ax.grid(alpha=.15)
        axs[0].scatter([ck["step"]],[ck["metrics"]["real_iv_rmse"]],s=85,
                       marker="*",edgecolor="black",color=color,zorder=5)
    axs[0].set_ylabel("Pooled price error / calibration-derived vega")
    axs[1].axhline(first["syn_z_mae"]*1.05,color="#a43d46",ls="--",label="5% worsening limit")
    axs[2].axhspan(.85,.95,color="#248779",alpha=.08);axs[2].axhline(.9,color="gray",ls=":")
    axs[2].set_ylim(.84,.96);axs[0].legend();fig.tight_layout()
    save(fig,"training_diagnostics.png","All three seeds improve the development quote metric. Synthetic recovery and marginal coverage remain near their starting levels; better pricing has not solved ten-parameter recovery. Stars mark validation-selected checkpoints.")

    primary=metrics[metrics.model==PRIMARY]
    groups=[("rich",0.),("rich",.01),("single_expiry",0.),("single_expiry",.01)]
    fig,axs=plt.subplots(1,2,figsize=(13,4.8));rng=np.random.default_rng(71)
    for arm,offset in zip(COLORS,(-.22,0,.22)):
        for j,(geo,noise) in enumerate(groups):
            g=primary[(primary.arm==arm)&(primary.geometry==geo)&(primary.noise==noise)]
            xx=j+offset+rng.uniform(-.055,.055,len(g))
            for ax,key in zip(axs,("price_rmse_spot","z_max_error")):
                yy=np.maximum(g[key].to_numpy(),1e-16)
                ax.scatter(xx,yy,s=22,alpha=.65,color=COLORS[arm],label=LABELS[arm] if j==0 else None)
    for ax,title in zip(axs,("Held-out price reconstruction","Largest scaled latent parameter error")):
        ax.set_yscale("log");ax.set_title(title);ax.grid(axis="y",alpha=.15)
        ax.set_xticks(range(4),["Rich\nclean","Rich\n1% noise","One expiry\nclean","One expiry\n1% noise"])
    axs[0].axhline(1e-6,color="#a43d46",ls="--",label="Price target")
    axs[1].axhline(.01,color="#a43d46",ls="--",label="Parameter target")
    axs[0].set_ylabel("RMSE / spot");axs[1].set_ylabel("Max |z estimate − z true| / training SD")
    axs[0].legend(fontsize=8);axs[1].legend(fontsize=8);fig.tight_layout()
    save(fig,"price_and_parameter_recovery.png","Each dot is a separate case (tiny horizontal offsets only separate markers). Low price error does not imply low parameter error. Clean multi-expiry surfaces give the strongest recovery evidence. Red lines are the fixed targets; plot values below 1e-16 use that display floor only. Two noisy-rich hybrid prices fail the quadrature check, so their displayed 128-node errors are numerically unresolved.")

    # The first predetermined case is shown, without choosing the most attractive fit.
    case="905101_00";truth=truths.set_index("case").loc[case,list(CANONICAL)].to_numpy(float)
    own=params[(params.case==case)&(params.model==PRIMARY)&(params.geometry=="rich")&(params.noise==0)]
    fig,axs=plt.subplots(2,3,figsize=(13,7))
    for ax,days in zip(axs.flat,(30.,60.,90.,180.,365.,730.)):
        strike=np.linspace(.8,1.2,15);tau=np.full(15,days/365.)
        geo=[torch.ones(15,dtype=torch.float64),torch.tensor(strike),torch.tensor(tau),
             torch.full((15,),.05,dtype=torch.float64),torch.full((15,),.01,dtype=torch.float64)]
        with torch.no_grad():target=price_call(torch.tensor(truth),*geo,node_count=128).numpy()
        ax.scatter(strike,target,color="black",s=18,label="Synthetic reference",zorder=4)
        for arm in COLORS:
            p=own[own.arm==arm].set_index("parameter").loc[list(CANONICAL),"estimated"].to_numpy()
            with torch.no_grad():pred=price_call(torch.tensor(p),*geo,node_count=128).numpy()
            ax.plot(strike,pred,color=COLORS[arm],label=LABELS[arm],lw=1.5,
                    ls="--" if arm=="hybrid_polish" else "-")
        ax.set(title=f"{days:g} days",xlabel="Strike / spot",ylabel="Call price / spot");ax.grid(alpha=.15)
    axs[0,0].legend(fontsize=8);fig.tight_layout()
    save(fig,"example_price_surface.png","The first prespecified synthetic case is shown across six maturities. Black points are exact-engine synthetic prices, not NSE observations. Lines use the estimated parameters. An overlapping line demonstrates price reconstruction for this case, not universal parameter identification.")

    model_summary=market["models"]
    fig,axs=plt.subplots(1,2,figsize=(13,4.7))
    model_labels=["Base","Seed 17","Seed 29","Seed 43"]
    actual=[m["iv_rmse_all_quotes"] for m in model_summary]
    firstorder=[m["first_order_vega_rmse_all_quotes"] for m in model_summary]
    xx=np.arange(4)
    axs[0].bar(xx-.18,[np.nan if v is None else v for v in actual],.36,label="Actual IV inversion",color="#248779")
    axs[0].bar(xx+.18,[np.nan if v is None else v for v in firstorder],.36,label="Price / vega approximation",color="#5068a9")
    axs[0].set_xticks(xx,model_labels);axs[0].set_title("All 1,607 development holdout quotes")
    axs[0].set_ylabel("RMSE in volatility units");axs[0].legend(fontsize=8)
    symbols=list(model_summary[0]["by_symbol"])
    for i in (0,3):
        vals=[model_summary[i]["by_symbol"][s]["iv_rmse_all_quotes"] for s in symbols]
        axs[1].plot(range(len(symbols)),[np.nan if v is None else v for v in vals],"o-",label=model_labels[i])
    axs[1].set_xticks(range(len(symbols)),symbols,rotation=70,ha="right")
    axs[1].set_title("Per-symbol actual IV error");axs[1].legend(fontsize=8)
    for ax in axs:ax.grid(axis="y",alpha=.15)
    fig.tight_layout()
    save(fig,"market_development_iv.png","These are development-validation results, including dates used for checkpoint selection. Actual IV error is recomputed by inversion, separately from the price/vega approximation. Seven near-zero model prices across all four models use independently checked high-precision inversion; none are clipped or dropped. The symbol breakdown exposes gains or regressions hidden by the pooled average; it is not a fresh market test.")

    aggregate=[]
    for (model,arm,geometry,noise),g in metrics.groupby(["model","arm","geometry","noise"]):
        aggregate.append({"model":model,"arm":arm,"geometry":geometry,"noise":noise,"cases":len(g),
            "price_gate_passes":int(g.price_gate.sum()),"parameter_gate_passes":int(g.parameter_gate.sum()),
            "joint_gate_passes":int((g.price_gate&g.parameter_gate).sum()),
            "pooled_price_rmse":float(np.sqrt(np.mean(g.price_rmse_spot**2))),
            "successful_only_median_max_scaled_parameter_error":float(g.loc[~g.failed_parameter_case,"z_max_error"].median()),
            "failed_parameter_cases":int(g.failed_parameter_case.sum()),
            "quadrature_unresolved_cases":int((g.predicted_quadrature_max_difference>1e-6).sum()),
            "median_seconds":float(g.seconds.median()),
            "invalid_iv_pairs":int(g.invalid_iv_pairs.sum()),
            "price_bound_violations":int(g.price_bound_violations.sum()),
            "maximum_predicted_quadrature_difference":float(g.predicted_quadrature_max_difference.max())})
    (OUT/"assessment_summary.json").write_text(json.dumps(aggregate,indent=2))
    (OUT/"figure_interpretations.json").write_text(json.dumps(explanations,indent=2))
    fmt=lambda x:"unavailable" if x is None or not math.isfinite(x) else f"{x:.6g}"
    lines=["# Double Heston calibration repair: measured results","",
        "The repair improves development calibration. Perfect recovery of all ten parameters is not established across the complete assessment.","",
        "## Training and real-data development validation","",
        "Three independent fine-tunes used the same base checkpoint and a 600-step budget. Selection used development validation only, before the independent synthetic assessment. Seed 43, step 500 is the selected primary.","",
        "| Model | Selected step | Selection price/vega RMSE | Actual IV RMSE, precision checked | Unresolved IV quotes |",
        "|---|---:|---:|---:|---:|"]
    for j,m in enumerate(model_summary):
        score=json.loads((OUT/"seed_17/repaired_initial.json").read_text())["real_iv_rmse"] if j==0 else selected[j-1]["real_iv_rmse"]
        lines.append(f"| {model_labels[j]} | {m.get('checkpoint_step') or 'base'} | {fmt(score)} | {fmt(m['iv_rmse_all_quotes'])} | {m['invalid_iv_quotes']} |")
    lines += ["","All 203 validation surfaces and all 1,607 quote holds are included per model. These are development results, not forward prediction or a new unexamined market test. Prices are normalised undiscounted calls; IV uses fractional annual volatility (0.01 is one volatility point).", "",
        "The selected primary improves actual pooled IV RMSE by about 1.20%, not the roughly 22% improvement in the selection price/vega approximation. Nine symbols improve; JSWENERGY and NIFTY worsen (NIFTY: 0.048854 → 0.050243). The approximation cannot be substituted for actual IV accuracy, and this is not evidence of universal improvement. Seed 29 has a lower actual IV RMSE here, but the prespecified selection remains seed 43.","",
        "The original 128-node evaluator returned three tiny negative call prices (one per repaired model), making each original all-quote IV metric unavailable. A uniform time-value threshold of 1e-10, applied to every model, triggered independent Carr–Madan inversion at 40 and 60 decimal precision for seven quotes, including the base model. All seven passed precision and tail checks. Original prices and failures are preserved in market_validation; numerical re-evaluation is in market_precision_checked. No observed quote, fitted parameter, checkpoint selection or assessment gate was changed. This numerical guard was introduced after tracing the failures, not claimed as preregistered.","",
        "## Independent synthetic recovery","",
        "24 independent parameter sets (seeds 905101/905102/905103), each evaluated with six expiries and one expiry, clean and 1% noisy: 96 conditions. All three repaired networks and the base network were assessed; exact-optimiser polish was applied only to the primary selected before the assessment. Every third strike was withheld. Calibration code receives no true parameters or quote holds.","",
        "| Primary arm | Geometry | Noise | Price target passed | All-parameter target passed | Both passed | Pooled price RMSE / spot |",
        "|---|---|---:|---:|---:|---:|---:|"]
    for r in aggregate:
        if r["model"]==PRIMARY:
            lines.append(f"| {LABELS[r['arm']]} | {r['geometry']} | {r['noise']:.0%} | {r['price_gate_passes']}/24 | {r['parameter_gate_passes']}/24 | {r['joint_gate_passes']}/24 | {fmt(r['pooled_price_rmse'])} |")
    lines += ["","Price target: heldout RMSE ≤1e-6 of spot, convergent 96/128-node prices within 1e-6, no bound violations. Parameter target: every latent coordinate within 0.01 training standard deviations of its generating value. These thresholds were fixed in the protocol. They were not loosened after seeing results.","",
        "The synthetic richness ladder is 30/60/90/180/365/730 days with 15 strikes from 0.8 to 1.2 of spot. This is a controlled synthetic design, not a claim that all such maturities are available for Indian stock contracts. For the noisy runs, strict clean-recovery gates are diagnostic stress targets, not realistic guarantees of exact recovery from noisy observations.","",
        "A low price error with a failed parameter target is a failure of parameter recovery. Optimiser success flags are recorded separately in polish_runs.json; a budget-exhausted fit remains identifiable as such. The exact optimiser is part of the hybrid's reported runtime. It is not credited to the network alone.","",
        "Selected optimiser termination was successful in 60/96 hybrid conditions. Two noisy-rich hybrid fits have unresolved quadrature: 905103_00 (96/128-node difference 1.72e-4) and 905103_07 (7.77e-5). Both fail the price gate. Their finite 128-node errors remain in the pooled table above, which must not be mistaken for numerically certified accuracy.","",
        "### All four networks, same three-step refinement","",
        "| Model | Rich clean RMSE / spot | Rich noisy | Single clean | Single noisy |",
        "|---|---:|---:|---:|---:|"]
    for model in ("unified_v6/unified","seed_17/repaired","seed_29/repaired",PRIMARY):
        values=[next(r["pooled_price_rmse"] for r in aggregate if r["model"]==model and r["arm"]=="refined" and r["geometry"]==geo and r["noise"]==noise) for geo,noise in groups]
        lines.append(f"| {model} | "+" | ".join(fmt(v) for v in values)+" |")
    lines += ["","The primary checkpoint was chosen on real-data development validation, not this table. It does not beat the base in every synthetic condition: rich-clean and single-expiry noisy pricing regress slightly. These regressions are retained; no alternative seed was selected after inspecting them.","",
        "### Example parameters: first prespecified clean rich case 905101_00","",
        "| Parameter | Generating value | Network only | Three-step refined | Hybrid exact optimisation |",
        "|---|---:|---:|---:|---:|"]
    for j,key in enumerate(CANONICAL):
        values=[float(own[(own.arm==arm)&(own.parameter==key)].estimated.iloc[0]) for arm in COLORS]
        lines.append(f"| {key} | {fmt(float(truth[j]))} | "+" | ".join(fmt(v) for v in values)+" |")
    lines += ["","These are one synthetic case's values, not a universal or market parameter set. Every estimate and its generating label, including failed recovery cases, is in all_parameter_recovery.csv. Market estimates are in market_validation/surface_parameters_and_metrics.csv with no invented truth labels.","",
        "## Changes and verification","",
        "- Correct entropy-bearing negative ELBO replaced the unsupported variance-loss collapse argument. Analytic tests verify appropriate expansion and contraction.",
        "- Refinement uses the declared quote scale and per-surface objective acceptance. Invalid quotes are rejected, not removed from losses.",
        "- Nonfinite objectives cannot corrupt moving loss scales. Checkpoints retain seed, optimiser state, RNG state, metrics and provenance.",
        "- Calendar embargo covers all symbols in the new corpus; quote holds are removed before encoder feature construction. NIFTY carry and validation weight/noise fits use calibration strikes.",
        "- Local sensitivity uses SVD and reports weak directions; encoder uncertainty is explicitly distinguished from refined estimates.","",
        "The final scoped verification log is verification_final.txt: 46 tests passed, with 18 PyTorch deprecation warnings. This is the repair/unified test set, not a claim that every historical repository test was executed. Source hashes, all checkpoint selections, data exclusions and runtime versions are saved beside the results.","",
        "## What remains unresolved","",
        "Ten unique true market parameters cannot be verified from real quotes because no such labels are observed. Noisy/short-tenor surfaces can support very different parameters with similar prices. The neural parameter MAE remains substantial; the approximate Gaussian posterior does not fully describe curved or multimodal ambiguity. The new market split is embargoed, but prior human/test inspection and historical checkpoint exposure cannot be undone.","",
        "Fast/slow ordering removes factor-label swapping; it does not remove all compensating parameter combinations. IV derived from the same option prices changes the weighting of errors, but is not an independent observation that can by itself guarantee identification. Clean rich recovery succeeded for 21/24 hybrid cases, while all 24 clean single-expiry cases failed the parameter gate despite passing the price gate. Unregularised noisy polish can overfit: it worsens pooled single-expiry price error and yields much larger parameter errors. Do not deploy it indiscriminately for noisy market quotes.","",
        "Strict quote-level liquidity leaves 3–7 NIFTY validation expiries, correcting the earlier 10+ claim under aggregate liquidity. Exact market IV and first-order price/vega metrics differ; neither should be silently substituted for the other. No new p-value or market superiority claim is made.","",
        "No further model/hyperparameter selection was made from this assessment. A future improvement cycle must preserve these results as development evidence and reserve new data for assessment.","",
        "The next defensible steps are noise-aware regularisation of weak parameter directions, selecting future models with actual IV metrics when IV accuracy is the objective, and evaluating on new untouched dates/parameter draws. Adding price-derived IV as another target or increasing optimisation steps alone does not provide missing identifying information.","",
        "## Figures and interpretation",""]
    for name,caption in explanations:
        lines += [f"![{name}](figures/{name})","",caption,""]
    lines += ["## Continue the project","",
        "Start with docs/CALIBRATION_REPAIR_HANDOFF.md and docs/CALIBRATION_REPAIR_PROTOCOL.md. The executable paths are src/mentor_dh_pinn/unified.py, finetune_projection.py and precise_calibration.py. Run scripts/mentor_dh_pinn/evaluate_repair_validation.py for the new development evaluation, not the historical evaluator scripts.","",
        "Methodological reference for the entropy/KL objective: [Blei, Kucukelbir and McAuliffe (2017)](https://www.cs.columbia.edu/~blei/papers/BleiKucukelbirMcAuliffe2017.pdf). Calibration flat valleys are discussed for the single Heston case in [Cui, del Baño Rollin and Germano (2017)](https://discovery.ucl.ac.uk/1552816/1/Germano_full%20and%20fast%20calibration_Heston_EurJOperRes_263_625_2017_accepted.pdf); this is context, not a proof of Double Heston non-identifiability. Independent damped inversion follows [Carr and Madan (1999), equations 5–6](https://wwwf.imperial.ac.uk/~ajacquie/IC_Num_Methods/IC_Num_Methods_Docs/Literature/CarrMadan.pdf)."]
    (OUT/"RESULTS.md").write_text("\n".join(lines))
    cells=[]
    def markdown(text):cells.append({"id":f"cell-{len(cells):03d}","cell_type":"markdown","metadata":{},"source":text.splitlines(True)})
    def code(text):cells.append({"id":f"cell-{len(cells):03d}","cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":text.splitlines(True)})
    markdown("# Double Heston calibration repair\n\nThis notebook displays completed results. It does not retrain or tune on assessment data. Read RESULTS.md for limits and gate counts.")
    code("from pathlib import Path\nimport pandas as pd\nfrom IPython.display import display, Image, Markdown\nBASE = Path.cwd()\nif not (BASE / 'RESULTS.md').exists():\n    BASE = BASE / 'outputs' / 'calibration_repair'\nassert (BASE / 'RESULTS.md').exists(), 'Open from repository root or this notebook folder.'\ndisplay(Markdown((BASE / 'RESULTS.md').read_text().split('## Figures')[0]))\n")
    for name,caption in explanations:
        code(f"display(Image(filename=str(BASE / 'figures' / '{name}')))\n")
        markdown(caption)
    code("parameters = pd.read_csv(BASE / 'all_parameter_recovery.csv')\ndisplay(parameters.query(\"model == 'seed_43/repaired' and geometry == 'rich' and noise == 0\").head(30))\nmarket_parameters = pd.read_csv(BASE / 'market_validation' / 'surface_parameters_and_metrics.csv')\ndisplay(market_parameters.head())\n")
    markdown("The first table has synthetic generating labels; the market parameter table contains fitted estimates only. They must not be presented as observed true market parameters.")
    notebook={"cells":cells,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"}},"nbformat":4,"nbformat_minor":5}
    (OUT/"Double_Heston_Repair_Results.ipynb").write_text(json.dumps(notebook,indent=2))
    print(OUT/"RESULTS.md")


if __name__=="__main__":main()
