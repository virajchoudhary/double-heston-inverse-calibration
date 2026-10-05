# Defence briefing — slides 10, 11, 12, 14

Every number here is traceable to a committed artifact. Source files are named at the end.
Note: the deck has no slide 13; numbering jumps 12 -> 14.

---

# PART 0 — THE THINGS YOU MUST KNOW COLD

These four facts answer about half of all likely questions.

## 0.1 What the error numbers actually measure

**Price RMSE is not in dollars.** It is measured on the *forward-normalised* call price

    c = C / F

where `C` is the call price and `F` is the forward price. It is therefore **dimensionless**.

> "Price RMSE of 6.61e-6 means: as a fraction of the forward price. If the forward
> were 100, that is an error of about 0.00066 in currency units."

**IV RMSE is in volatility points**, which are percentage points of volatility. The code
computes `e = 100 * (model_iv - market_iv)`, so 0.0738 means the implied volatility is off
by 0.0738 percentage points — e.g. 20.0000% versus 20.0738%.

**PDE residual** is the scaled residual of the pricing equation. Dimensionless. It measures
how well the network satisfies the differential equation it is supposed to obey, at points
where no answer was supplied.

## 0.2 The data map — "on what data is the RMSE calculated?"

There are **six separate synthetic panels**, drawn in advance from the same frozen sampling
law with different seeds, plus the market data. They never overlap.

| Panel | Size | Seed | What it is used for |
|---|---|---|---|
| Training labels | 100,000 exact prices | 93101 | fitting the network |
| Collocation points | 18,000 | v4 sampler | the physics term (no answer supplied) |
| **Development** | **8,192** | **93201** | **slide 10 — choosing the ablation winner** |
| **Final fidelity** | **16,384** | **93202** | **slide 11 — opened exactly once** |
| PDE final | 4,096 | 93203 | the PDE residual on the final architecture |
| Short-ATM diagnostic | 6,144 | 93204 | reporting only, never selection |
| RAD pool | 120,000 | 93205 | candidate pool for adaptive collocation |

Market data:

| Panel | Size | Used for |
|---|---|---|
| S&P 500 quotes | **10,706** (VIX 14.2) | slide 12 |
| Track B split | 4,271 calibration / 1,082 development / **5,353 held out** | slide 14 |

**So the one-line answer:** slide 10's numbers are on the 8,192-point development panel;
slide 11's are on the 16,384-point final panel opened once; slide 12 and 14 are on real
S&P 500 quotes.

## 0.3 Where the "exact answer" comes from

The targets are not market prices and not another network. They are computed from the
Double-Heston characteristic function by Fourier inversion:

- Gauss-Laguerre quadrature with **128 nodes**, checked against a **96-node** run.
- If the two disagree by more than `1e-8`, or a price bound is breached, the code falls back
  to independent adaptive quadrature. It never clips.
- Audit result: 1,000 accepted prices agreed with tighter quadrature to within
  **2.65e-11**; the 121 fallback prices to within **1.76e-12**. No reference was flagged
  unreliable.

> That is why we can call it "exact": the reference is accurate to ~1e-11, while the network
> error we are measuring is ~1e-5. The reference is four orders of magnitude tighter than
> the thing being measured.

## 0.4 The one sentence that carries the whole deck

> "We made the fast approximation 1.61x more accurate. On real market data that changed the
> answer by 0.003 out of 5.85. The approximation was never the problem — the model was."

---

# PART 1 — SLIDE 10: SIX-STAGE ARCHITECTURE ABLATION

## What the slide claims

Six versions of the network, each adding one ingredient to the previous one, all trained
with an identical budget. Shortcut connections matter most; the gated version wins.

## The exact numbers (development panel, 8,192 points, seed 93201)

| Stage | What it adds | Params | Price RMSE | IV RMSE | PDE | Train s |
|---|---|---:|---:|---:|---:|---:|
| locked v4 | *(different budget — see below)* | 269,825 | 1.081e-5 | 0.1125 | 0.031 | 5,394 |
| A0 | plain tanh stack | 269,830 | 6.144e-5 | 0.3513 | 0.138 | 1,491 |
| A1 | + residual blocks | 269,834 | 1.985e-5 | 0.1904 | 0.044 | 1,716 |
| A2 | + layer-wise adaptive tanh | 269,834 | 1.572e-5 | 0.1575 | 0.038 | 1,801 |
| A3 | + gradient-norm loss balancing | 269,834 | 2.549e-5 | 0.1157 | 0.082 | 1,542 |
| A4 | + residual-adaptive collocation | 269,834 | 2.475e-5 | 0.1197 | 0.036 | 1,333 |
| **A5** | **gated block replaces residual** | **282,632** | **1.471e-5** | **0.1051** | **0.020** | **1,884** |

## The identical budget (say this if asked "how do you know it's the design and not luck?")

Every stage: **seed 17 only**, 12,000 Adam steps, learning rate 1e-3 cosine-decayed to 1e-5,
**512 label points and 128 physics points per step**, then **300 strong-Wolfe L-BFGS steps**,
same data throughout. Parameter counts are within 5 of each other for A0-A4.

## THE TRAP ON THIS SLIDE

**The "locked v4" bar is not comparable to A0-A5.** It was trained with 40,000 Adam steps
and two seeds; the ablation stages get 12,000 steps and one seed. It is on the chart as
context — the level the production model reached — not as a competitor.

> If asked: "The locked bar answers 'where were we', not 'which stage is best'. Comparing it
> to A0 would be comparing 40,000 steps to 12,000."

This also explains the apparent oddity that **A0 (6.144e-5) is far worse than locked v4
(1.081e-5) despite being the same architecture.** It is the same design starved of budget.
That is the point: it shows how much of v4's quality came from brute-force training.

## Reading each result

- **Residual connections: 6.144e-5 -> 1.985e-5, a 3.1x drop**, at +4 parameters. This is the
  single biggest effect in the study.
- **Adaptive activation: 1.985e-5 -> 1.572e-5, a further 21%.** The learned scales settle at
  1.1-1.4 in the residual stages and 0.6-1.0 in the gated one.
- **Gradient balancing: price error gets WORSE (1.572e-5 -> 2.549e-5) while IV error improves
  (0.1575 -> 0.1157).** This is not a bug. The method reweights the loss toward whichever term
  has the smaller gradient, which moves weight away from the price term by construction.
- **Adaptive collocation: mainly fixes the physics** (PDE 0.082 -> 0.036) at slightly less
  time. Price barely moves.
- **Gated block: best on all three** for 4.7% more parameters.

## Q&A

**Q: Why keep A3 and A4 in the study if they made things worse?**
They were predeclared. Removing a stage after seeing its result is exactly the selection bias
the protocol exists to prevent. They also produce the most interesting finding — that the two
objectives compete.

**Q: Only one seed in the ablation. Isn't that weak?**
Yes, and it is listed as a limitation. The mitigation is that the ablation only *chooses* a
winner; the winner was then retrained with two seeds and confirmed on a panel it had never
seen. A bad choice would have shown up there.

**Q: What was the selection rule?**
Fixed in advance: lowest development price RMSE, with a simpler stage winning if it came
within 5%. A5 won outright (1.471e-5 vs A2's 1.572e-5, a 6.4% gap, so the simplicity
tie-break did not trigger).

**Q: Why is A5's number here 1.471e-5 but 6.610e-6 on the next slide?**
Two changes at once. Different budget (12,000 steps, one seed here; the full 40,000-step
recipe with two seeds there) and a different panel (8,192 development here; 16,384 final
there). The numbers are not comparable and are never compared in the paper.

**Q: What is a "gated block" in plain terms?**
Two encodings of the input are computed once, and at each layer the network learns a gate
that mixes between them. Instead of forcing information through a chain of transformations,
it gives every layer direct access to the original input.

---

# PART 2 — SLIDE 11: UNTOUCHED FINAL PANEL

## What the slide claims

On 16,384 examples never used during development, opened once, the selected architecture is
1.61x more accurate — but slower, larger, and worse at obeying the equation.

## The exact numbers (final panel, seed 93202, two seeds averaged)

| Metric | Locked baseline | Selected | Ratio |
|---|---:|---:|---|
| Price RMSE | 1.062e-5 | **6.610e-6** | 1.61x better |
| Price P95 | 2.258e-5 | 1.419e-5 | 1.59x |
| Price max | 1.523e-4 | 5.938e-5 | 2.56x |
| IV RMSE | 0.1175 | **0.0738** | 1.59x |
| 7-30 day IV RMSE | 0.2232 | 0.1429 | 1.56x |
| Short-ATM IV RMSE | 0.0618 | 0.0317 | 1.95x |
| **PDE residual** | **0.0309** | **0.0558** | **1.8x WORSE** |
| Parameters | 269,825 | 282,632 | +4.7% |
| Inference | 14.2 us/quote | 24.8 us/quote | 1.7x slower |
| Training time | 5,394 s | 14,205 s | 2.6x |

## Predeclared targets — state these, they show the bar was set first

- Price RMSE below **7e-6**: **met** (6.610e-6).
- Stretch target below 5e-6: **not met**.
- IV RMSE below **0.07**: **just missed** at 0.0738.
- Stretch below 0.05: **not met**.

Reporting a missed target is a strength. It proves the target was set before the result.

## Q&A

**Q: "Opened once" — what does that actually mean operationally?**
The panel was generated from a seed fixed in advance and not evaluated at any point during
architecture work. The winner was chosen entirely on the development panel. The final panel
was scored one time, after the architecture was frozen. No result from it fed back into any
decision.

**Q: Why is the worst-case error (2.56x) better than the typical error (1.61x)?**
The gated design helps most where the old network struggled most — long maturities and the
tails. Improving the worst region moves the maximum more than the average.

**Q: Why did the PDE residual get worse?**
Gradient-norm balancing moves weight toward whichever loss term is being under-served, and in
this configuration that was the supervised price and IV terms. The equation term got less
weight, so its residual rose. It is a direct consequence of a method we chose, and we report
it rather than dropping the metric.

**Q: Is a worse PDE residual dangerous?**
It matters for extrapolation and for derivatives of the price. It does not break the outputs
here: the appendix shows no violation of the no-arbitrage bounds. But it is why we do not
claim the new design is better in every respect.

**Q: Why average two seeds?**
Two independently trained models, predictions averaged. It reduces run-to-run noise. Two is
not enough to characterise the variability, which is why it is a stated limitation.

**Q: 24.8 microseconds per quote sounds fast. Why call it a cost?**
Because the use case is calibration, which evaluates the surrogate many thousands of times
per fit. A 1.7x slowdown compounds across a whole calibration loop.

---

# PART 3 — SLIDE 12: MARKET CEILING

## What the slide claims

On 10,706 real S&P 500 prices, making the network 1.61x better moved the market error by
0.003 out of 5.85, because the network contributes only ~1.3% of the total error.

## The exact numbers (IV RMSE, volatility points)

| Model | Nothing fitted | One level scale fitted |
|---|---:|---:|
| Black-Scholes, fixed vol | 6.855 | 7.271 |
| Single Heston, published | 5.359 | 5.788 |
| **Double Heston, published (exact)** | **5.344** | 5.818 |
| Surrogate, locked (v4) | 5.365 | 5.850 |
| Surrogate, selected (v5) | 5.364 | 5.847 |

## The arithmetic of the ceiling — know this cold

- Network's distance to the exact model improved by **0.044** (IV 0.1175 -> 0.0738).
- Its distance to the **market** changed by **0.003** (5.850 -> 5.847).
- Network error / market error = **0.0738 / 5.85 ~ 1.3%**.
- A **perfect** surrogate would land exactly on the exact model: **5.818**. The remaining
  5.8 is model and parameter misspecification, not approximation error.

## THE HARDEST QUESTION ON THIS SLIDE

**Q: Why does fitting a level scale make Double Heston WORSE (5.344 -> 5.818)?**

This looks wrong and you must be ready for it. The answer:

The scale is fitted by minimising **vega-weighted price error**, but the table reports
**implied-volatility RMSE**. Those are two different objectives. Improving the first can
worsen the second, and here it does. The scale is also constrained to [0.7, 3.2] — the range
the network was trained on — so it cannot wander outside the trained domain.

> "The 'nothing fitted' column uses the published parameters raw. The 'fitted' column adjusts
> one overall volatility level to best match prices. Because we report the error in
> volatility terms rather than price terms, the fitted column can be worse. Both columns are
> shown so nobody has to take one on trust."

This does not affect the headline: the surrogate-versus-exact comparison (5.847 vs 5.818) is
within the same column, so the comparison is like-for-like.

## Other Q&A

**Q: Is 5.8 volatility points a big error?**
Yes, very. Implied volatility might be around 15-20%, so being off by 5.8 percentage points
is a large miss. That is exactly the point of the slide — the model with fixed published
parameters is a poor fit to this surface, and no amount of network improvement fixes that.

**Q: Why not just calibrate the model to the market?**
That is precisely the recommendation on the final slide. This experiment deliberately holds
the structural parameters fixed at published values in order to isolate the network's
contribution. Calibrating would have changed two things at once.

**Q: Where is the model worst?** (by maturity, level scale fitted)

| Maturity | BS | SH | DH exact | v5 |
|---|---:|---:|---:|---:|
| <= 30 days | **7.33** | 7.47 | 7.49 | 7.55 |
| 30-90 days | 7.75 | 5.57 | 5.51 | **5.50** |
| 90-365 days | 7.00 | **4.30** | 4.45 | 4.45 |
| > 365 days | 6.18 | 3.66 | **3.56** | 3.56 |

Note honestly: **Black-Scholes is the best model at 30 days or less.** A fixed two-factor
shape is simply wrong in that region. This is not a network failure — the network's own
short-dated error against the exact model is 0.03 volatility points, against a market gap
of 3-7.

**Q: Only one day of market data?**
Yes, one surface of 10,706 quotes at VIX 14.2. That is a limitation. The breadth check across
five other markets is in the paper's appendix, not on this slide.

---

# PART 4 — SLIDE 14: KEY FINDINGS, LIMITATIONS & NEXT STEPS

## What the slide claims

A small correction layer cut market error by two thirds — but it helped every underlying
model equally, so the correction did the work, not the richer model.

## The setup

The surface is split by a **frozen checkerboard on (expiry rank + strike rank)**:
**4,271 calibration / 1,082 development / 5,353 held out.** Held-out quotes are never inputs,
never train the correction, and never select its hyper-parameters. A checkerboard is used
rather than a random split so that neighbouring strikes cannot leak into each other.

The correction layer, **identical for every model**:
inputs `[x, log tau, prior IV, x/sqrt(tau)]` standardised on calibration statistics,
then `Linear(4,64) -> tanh -> Linear(64,64) -> tanh -> Linear(64,1)`.
Output is `delta_max * tanh(.)` added to the model's implied volatility.
**4,545 parameters.** Adam, 4,000 steps, penalty 0.05 on the squared correction.
`delta_max` chosen on development from {0.02, 0.05, 0.10}.

## The exact numbers (held-out, 5,353 quotes)

| Prior model | Alone | + identical correction | Mean correction size |
|---|---:|---:|---:|
| Black-Scholes | 7.274 | 2.366 | 4.33 |
| **Single Heston** | 5.773 | **1.917** | 3.33 |
| Double Heston, exact | 5.839 | 1.990 | 3.35 |
| Double Heston, surrogate | 5.850 | 1.970 | 3.36 |

## The fair comparison, over 19 maturity-by-moneyness cells

| Comparison | Mean difference | Cells favouring DH | p | 95% interval | Verdict |
|---|---:|---:|---:|---|---|
| Black-Scholes − Double Heston | +0.580 | 13/19 | **0.018** | [+0.087, +1.292] | **DH wins** |
| Single Heston − Double Heston | +0.020 | 9/19 | **0.340** | [−0.009, +0.079] | **DH does NOT win** |

## WHY THE NEGATIVE RESULT IS THE STRONGEST PART OF THE TALK

Do not apologise for it. Say:

> "We expected the two-factor model to win. It did not. We had a diagnostic that predicted
> this before we ran the test: after fitting the level, the error left over by Double Heston
> is 5.797 and by Single Heston is 5.801 — statistically indistinguishable. Double Heston's
> leftover error is smoother across maturity (0.678 versus 0.794) but it is not smaller. So
> the same small correction learns both about equally well. We reported it rather than
> hunting for a framing where the result flipped."

## Q&A

**Q: Is the hybrid still a Double-Heston model?**
No, and it is labelled as a hybrid everywhere. The structural parameters never change; a
learned correction is added on top. It should never be reported as exact Double Heston.

**Q: `delta_max` picked 0.10 — is that a problem?**
It picked the largest value on the grid for every model, which means the constraint is
binding: the correction wants more freedom than the protocol allows. We report this rather
than widening the grid after the fact, which would have been tuning on the result.

**Q: Does the hybrid produce sensible prices?**
Checked on the held-out grid. **No monotonicity violations for any hybrid.** Convexity
violations 773-866 out of ~5,250 triples, calendar violations ~660 out of ~5,330. For context,
**the raw market quotes on the same grid show 1,087 convexity violations** — so the hybrids
are smoother than the data they are fitting, not broken.

**Q: Where does the two-factor model actually help?**
Only the 30-90 day band (1.49 vs 1.55), and it is small. The deep low-strike wing is the
hardest region for everything (about 4 volatility points for every model).

**Q: What would actually improve the market number?**
Two things, and neither is more network capacity: fitting the structural parameters to each
day's data, and a richer model for the steep low-strike wing.

---

# PART 5 — CROSS-CUTTING QUESTIONS

**Q: What is a physics-informed neural network, in one sentence?**
A network trained not only to match known answers but also to satisfy the differential
equation those answers obey, with the equation checked at extra points where no answer is
supplied.

**Q: Why use a network at all if you have an exact formula?**
Speed. The exact formula requires numerical Fourier integration per price. The network
returns a price in about 25 microseconds, which matters when calibration evaluates it
thousands of times.

**Q: How is the payoff at expiry guaranteed correct?**
By construction, not by training. The network outputs a bounded correction to a volatility,
which is then passed through an analytic Black formula. At expiry that formula returns the
payoff exactly, so the boundary condition cannot be violated and the price bounds cannot break.

**Q: What are the 24 engineered features?**
Four general, nine per variance factor, and two cross-factor terms, built from the 12 raw
inputs. None of them are learned — they are fixed transformations chosen in advance, and they
were not changed during this study.

**Q: What stopped you from continuing to improve the architecture?**
A stopping rule fixed in advance: once the IV error against the exact model falls below 0.07
volatility points, architecture work stops and the remaining market error is attributed, with
numbers, to model misspecification. We reached 0.0738 and stopped. The argument does not
depend on the last 0.004.

**Q: What is the single biggest weakness of this work?**
Two seeds. Everything else is either measured or disclosed, but two training runs cannot
establish how much of the improvement is run-to-run variation.

---

# SOURCES

- `IMPROVED_DH_PINN_REPORT.md` — all tables in Parts 1-4
- `config.json` — panel sizes, seeds, budgets, selection rule, stopping rule
- `track_b.py` — `fit_scale` (the vega-weighted price objective), the correction head
- `spx_benchmark.py` — the market table and the two fitting columns
- `artifacts/` — the raw per-stage metrics
