"""Draw the verified, selected C3 network; no training or data changes."""
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from mentor_dh_pinn.regular_pinn_torch import TorchRegularVariancePINN

selection = json.loads((ROOT / "experiments/nifty_multifactor_v4/artifacts/pinn_selection.json").read_text())
assert selection["chosen"] == "C3"
net = TorchRegularVariancePINN(factors=2, width=selection["width"], depth=selection["depth"])
assert [(layer.in_features, layer.out_features) for layer in net.hidden] == [(24, 256)] + [(256, 256)] * 4
assert net.head.out_features == 1
assert sum(p.numel() for p in net.parameters()) == 269825

fig, ax = plt.subplots(figsize=(15, 8.8), facecolor="white")
ax.set(xlim=(-.65, 12.65), ylim=(-2.95, 4.95), aspect="equal")
ax.axis("off")
ink, blue, muted = "#172b46", "#336aa5", "#586b7e"
ax.text(6, 4.6, "Current Double Heston PINN — neural architecture", ha="center", fontsize=23, color=ink, weight="bold")
ax.text(6, 4.17, "Selected C3 model • one network shown • fully connected feedforward layers", ha="center", fontsize=14, color=muted)

xs = [0, 2, 4, 6, 8, 10, 12]
ys = [2.9, 2.25, 1.6, .45, -.2]
layers = [ys] * 6 + [[1.35]]
for i in range(6):
    for y1 in layers[i]:
        for y2 in layers[i + 1]:
            ax.plot([xs[i], xs[i + 1]], [y1, y2], color=blue, alpha=.16, lw=.8, zorder=1)
for i, x in enumerate(xs):
    for y in layers[i]:
        ax.add_patch(Circle((x, y), .19, facecolor="#e7f0fa" if i < 6 else "#d7f0e7", edgecolor=blue, lw=1.5, zorder=3))
    if i < 6:
        ax.text(x, 1.04, "⋮", ha="center", va="center", fontsize=27, color=ink,
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 0}, zorder=4)
    heading = "Input" if i == 0 else (f"Hidden {i}" if i < 6 else "Output head")
    count = "24 features" if i == 0 else ("256 neurons" if i < 6 else "1 neuron")
    activation = "Engineered inputs" if i == 0 else ("tanh" if i < 6 else "Linear")
    ax.text(x, 3.65, heading, ha="center", fontsize=14, color=ink, weight="bold")
    ax.text(x, 3.29, count, ha="center", fontsize=12, color=muted)
    ax.text(x, -.66, activation, ha="center", fontsize=12, color=ink)

ax.text(6, -1.13, "Circles are representative; dots indicate omitted neurons. Each adjacent layer is fully connected.", ha="center", fontsize=12, color=muted)

labels = ["Bounded correction\nδ = 1.8 tanh(head)", "Implied volatility\nIV = √v̄ × exp(δ)", "Black price transformation\nCall price C"]
centers = [1.65, 6, 10.35]
for x, label in zip(centers, labels):
    ax.add_patch(FancyBboxPatch((x - 1.82, -2.18), 3.64, .72, boxstyle="round,pad=0.08", facecolor="#f0f4f8", edgecolor="#bdcbd7"))
    ax.text(x, -1.82, label, ha="center", va="center", fontsize=13, color=ink)
for left, right in zip(centers[:-1], centers[1:]):
    ax.annotate("", (right - 1.94, -1.82), (left + 1.94, -1.82), arrowprops={"arrowstyle": "->", "color": blue, "lw": 1.5})
ax.text(6, -2.67, "Training loss: price + IV + Double Heston PDE + convexity  |  Final prediction: mean price of two networks (seeds 17, 43)", ha="center", fontsize=11.7, color=ink)
out = Path(__file__).resolve().parent
fig.savefig(out / "current_pinn_architecture.png", dpi=200, bbox_inches="tight", facecolor="white")
fig.savefig(out / "current_pinn_architecture.svg", bbox_inches="tight", facecolor="white")
print("Verified C3: 24 → 256 → 256 → 256 → 256 → 256 → 1; 269,825 trainable weights and biases.")
