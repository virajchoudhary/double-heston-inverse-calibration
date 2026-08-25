# Literature inventory

Scope: citation-ready sources used by `paper/references.bib` and
`paper/sections/02_motivation_and_related_work.tex`. Metadata was audited from the
repository first, then verified on 2026-08-25 through Crossref DOI/bibliographic
queries or the arXiv API. Claims below are limited to verified metadata, returned
abstracts where available, and titles when no abstract was returned. The project's
own Model2 remains a constraint- and differentiable-repricing-informed inverse model,
not a PINN.

## Summary

- Selected references: 24.
- Formal venue/DOI records: 15.
- arXiv/preprint records: 9.
- Records dated 2022--2026: 8.
- Coverage: Heston/multifactor stochastic volatility; affine transforms; empirical
  model comparison; parameter estimation; neural amortization/calibration;
  approximation theory; robust Bayesian calibration; PINN foundations/failure
  modes; physics-informed option/volatility-surface computation.
- Not covered exhaustively: local-stochastic volatility calibration, rough-volatility
  theory, market microstructure, Bayesian uncertainty quantification, and all
  multifactor Heston variants. These are future bibliography work, not negative
  findings about the literature.

## Reference records

### `heston1993closed`

- Verified metadata: Heston, Steven L. (1993), Review of Financial Studies 6(2),
  327--343, DOI `10.1093/rfs/6.2.327`.
- Stable identifier: <https://doi.org/10.1093/rfs/6.2.327>.
- Category: foundational stochastic volatility.
- Problem solved: closed-form option valuation under stochastic variance, with
  applications to bond and currency options.
- Model/data/task: single-factor Heston; analytical/forward pricing.
- Parameter truth: model parameters are inputs to the formula, not recovered truth.
- Architecture/PDE residual: none.
- Metrics/noise/OOD: none inferred from available metadata.
- Limitation: foundational single-factor formulation rather than a two-factor
  inverse-calibration benchmark.
- Relationship: supplies the one-factor limit and naming basis for this paper's
  canonical ten-parameter Double Heston extension.

### `christoffersen2009shape`

- Verified metadata: Christoffersen, Peter; Heston, Steven; Jacobs, Kris (2009),
  Management Science 55(12), 1914--1932, DOI `10.1287/mnsc.1090.1065`.
- Stable identifier: <https://doi.org/10.1287/mnsc.1090.1065>.
- Category: multifactor stochastic volatility.
- Problem solved: explains index-option smirk level/slope variation that a
  single-factor model does not fully capture.
- Model/data/task: two-factor stochastic volatility; real index-option data;
  forward model fit plus empirical comparison.
- Parameter truth: unavailable in market data; fit is relative.
- Architecture/PDE residual: none.
- Metrics/noise/OOD: reported in-sample/out-of-sample fit improvement over a Heston
  benchmark in the publisher abstract; no parameter-truth recovery metric.
- Limitation: empirical fit does not establish unique recovery of latent factors.
- Relationship: directly motivates richer factor structure while distinguishing
  improved fit from identification.

### `duffie2000transform`

- Verified metadata: Duffie, Darrell; Pan, Jun; Singleton, Kenneth (2000),
  Econometrica 68(6), 1343--1376, DOI `10.1111/1468-0262.00164`.
- Stable identifier: <https://doi.org/10.1111/1468-0262.00164>.
- Category: affine transform pricing.
- Problem solved: transform analysis and asset pricing for affine jump-diffusions.
- Model/data/task: analytical forward pricing framework.
- Parameter truth: not applicable.
- Architecture/PDE residual: none.
- Metrics/noise/OOD: none inferred from metadata.
- Limitation: provides pricing machinery, not an identifiability solution.
- Relationship: supports the additive characteristic-function construction used by
  the canonical production Double Heston pricer.

### `bakshi1997empirical`

- Verified metadata: Bakshi, Gurdip; Cao, Charles; Chen, Zhiwu (1997), Journal of
  Finance 52(5), 2003--2049, DOI `10.1111/j.1540-6261.1997.tb02749.x`.
- Stable identifier: <https://doi.org/10.1111/j.1540-6261.1997.tb02749.x>.
- Category: empirical option-model comparison.
- Problem solved: compares empirical performance of alternative option-pricing
  models.
- Model/data/task: multiple option-pricing models; real-market evaluation; forward
  fit/comparison.
- Parameter truth: unavailable; performance is relative.
- Architecture/PDE residual: none.
- Metrics/noise/OOD: pricing-performance comparison implied by title; detailed metric
  fields were not asserted here because no abstract was retrieved.
- Limitation: relative price-fit evidence cannot identify true latent parameters.
- Relationship: motivates comparing BS/Heston/DH while retaining separate recovery
  diagnostics.

### `chernov2003alternative`

- Verified metadata: Chernov, Mikhail; Gallant, A. Ronald; Ghysels, Eric; Tauchen,
  George (2003), Journal of Econometrics 116(1--2), 225--257, DOI
  `10.1016/S0304-4076(03)00108-8`.
- Stable identifier: <https://doi.org/10.1016/S0304-4076(03)00108-8>.
- Category: stochastic-volatility model selection/estimation.
- Problem solved: evaluates alternative models for stock-price dynamics.
- Model/data/task: alternative continuous-time dynamics; econometric estimation;
  real return data is implied by the estimation context but was not additionally
  asserted.
- Parameter truth: unavailable in empirical estimation.
- Architecture/PDE residual: none.
- Metrics/noise/OOD: model-comparison focus follows from title; specific scores were
  not asserted without the paper text.
- Limitation: model ranking need not imply point-identification of every parameter.
- Relationship: supports treating model richness and parameter recoverability as
  distinct questions.

### `aitsahalia2007maximum`

- Verified metadata: A{\"i}t-Sahalia, Yacine; Kimmel, Robert (2007), Journal of
  Financial Economics 83(2), 413--452, DOI `10.1016/j.jfineco.2005.10.006`.
- Stable identifier: <https://doi.org/10.1016/j.jfineco.2005.10.006>.
- Category: stochastic-volatility parameter estimation/inverse problem.
- Problem solved: maximum likelihood estimation of stochastic-volatility models.
- Model/data/task: stochastic-volatility diffusions; inverse statistical estimation.
- Parameter truth: unavailable for empirical dynamics.
- Architecture/PDE residual: none.
- Metrics/noise/OOD: likelihood-based inference is indicated by title; numerical
  noise/OOD results were not asserted.
- Limitation: likelihood optimization does not by itself prove practical
  identifiability under finite noisy option panels.
- Relationship: contrasts classical statistical estimation with this project's
  tolerance-conditioned known-truth recovery tests.

### `bollerslev2002estimating`

- Verified metadata: Bollerslev, Tim; Zhou, Hao (2002), Journal of Econometrics
  109(1), 35--65, DOI `10.1016/S0304-4076(01)00141-5`.
- Stable identifier: <https://doi.org/10.1016/S0304-4076(01)00141-5>.
- Category: integrated-variance/moment estimation.
- Problem solved: estimates stochastic-volatility diffusion parameters using
  conditional moments of integrated volatility.
- Model/data/task: stochastic-volatility diffusion; moment-based inverse estimation.
- Parameter truth: unavailable empirically.
- Architecture/PDE residual: none.
- Metrics/noise/OOD: estimator properties are indicated by title; no external
  numerical values were asserted.
- Limitation: informative integrated-variance summaries do not automatically identify
  every two-factor component.
- Relationship: repository G2 development work used this result as motivation while
  explicitly finding that one persistence scalar cannot identify both kappas.

### `albrecher2007little`

- Repository-linked metadata: Albrecher, Hansj{\"o}rg; Mayer, Philipp; Schoutens,
  Wim; Tistaert, Jurgen (2007), Wilmott Magazine, 83--92.
- Stable identifier: repository-referenced PDF at Imperial College London:
  <https://www.ma.imperial.ac.uk/~ajacquie/IC_Num_Methods/IC_Num_Methods_Docs/Literature/HestonTrap.pdf>.
- Verification status: authors/title/year/pages/source are recorded from the
  canonical engine documentation's existing literature link; no Crossref DOI record
  was found in lightweight checks.
- Category: numerical implementation of Heston.
- Problem solved: numerically stable representation for Heston characteristic-
  function pricing.
- Model/data/task: single-factor Heston forward pricing.
- Parameter truth: not applicable.
- Architecture/PDE residual: none.
- Metrics/noise/OOD: none asserted.
- Limitation: magazine article without a discovered Crossref DOI; used narrowly for
  the stable trap formulation.
- Relationship: justifies the production engine's Little-Heston-Trap form before any
  inverse experiment.

### `liu2019neural`

- Verified metadata: Liu, Shuaiqiang; Borovykh, Anastasia; Grzelak, Lech A.;
  Oosterlee, Cornelis W. (2019), arXiv:1904.10523.
- Stable identifier: <https://arxiv.org/abs/1904.10523>.
- Category: neural financial-model calibration.
- Problem solved: CaNN framework treats model calibration as learning/training with
  an offline pricing pass and online parameter search.
- Model/data/task: high-dimensional stochastic-volatility models; inverse calibration;
  numerical experiments described in abstract.
- Parameter truth: synthetic experiments have prescribed settings, but the abstract
  does not provide per-case truth metrics; no stronger claim is made.
- Architecture: ANN solver plus adapted parallel global optimization.
- PDE residual: none stated.
- Metrics/noise/OOD: abstract reports efficient/accurate calibration and avoidance of
  local minima; no controlled noise/OOD protocol is claimed.
- Limitation: preprint; architecture and evaluation differ from frozen R2 protocol.
- Relationship: establishes amortized/learned pricing maps as a route around repeated
  numerical calibration, which Model1 and Model2 adapt in different ways.

### `itkin2019deep`

- Verified metadata: Itkin, Andrey (2019), arXiv:1906.03507.
- Stable identifier: <https://arxiv.org/abs/1906.03507>.
- Category: pitfalls in neural calibration / robustness discipline.
- Problem solved: identifies pitfalls in deep-learning calibration and proposes
  performance/accuracy improvements.
- Model/data/task: option-model calibration; methodological critique and remedies.
- Parameter truth: not established generally by a critique paper.
- Architecture/PDE residual: neural calibration methods discussed; no universal PDE
  residual.
- Metrics/noise/OOD: addresses accuracy/performance pitfalls and no-arbitrage use of
  trained networks according to abstract; no standardized OOD benchmark.
- Limitation: preprint; recommendations are not equivalent to a controlled recovery
  gate.
- Relationship: supports this paper's separation of fit quality, arbitrage behavior,
  and parameter accuracy.

### `bayer2019deep`

- Verified metadata: Bayer, Christian; Horvath, Blanka; Muguruza, Aitor; Stemper,
  Benjamin; Tomas, Mehdi (2019), arXiv:1908.08806.
- Stable identifier: <https://arxiv.org/abs/1908.08806>.
- Category: deep calibration of rough stochastic volatility.
- Problem solved: learns a fast parameter-to-price map, then performs calibration in
  a second step.
- Model/data/task: rough Bergomi demonstration in abstract; forward-map learning plus
  inverse calibration.
- Parameter truth: numerical calibration targets exist, but abstract reports no
  known-truth recovery table.
- Architecture: neural network approximator of prices/implied volatilities.
- PDE residual: none.
- Metrics/noise/OOD: compares sampling/training approaches and reports practical-use
  accuracy in abstract; no controlled observation-noise/OOD claim.
- Limitation: rough-volatility setting differs from canonical two-factor Heston.
- Relationship: motivates learning forward maps while preserving traditional inverse
  search, unlike direct supervised inverse mapping.

### `horvath2021deep`

- Verified metadata: Horvath, Blanka; Muguruza, Aitor; Tomas, Mehdi (2021),
  Quantitative Finance 21(1), 11--27, DOI `10.1080/14697688.2020.1817974`.
- Stable identifier: <https://doi.org/10.1080/14697688.2020.1817974>.
- Category: neural pricing/calibration in volatility models.
- Problem solved: neural-network perspective on pricing and calibration in rough
  volatility models.
- Model/data/task: learned forward map followed by calibration; formal journal
  version of the deep-learning-volatility line.
- Parameter truth: calibrated estimates are relative; known-truth recovery is not
  asserted here.
- Architecture: DNN pricing map.
- PDE residual: none.
- Metrics/noise/OOD: title supports combined pricing/calibration focus; specific
  noise/OOD rates were not asserted.
- Limitation: primarily rough-volatility calibration, not canonical DH recovery.
- Relationship: closest broad template for speed-motivated surrogate calibration;
  this paper adds fixed-truth recovery and non-identifiability gates.

### `sridi2023applying`

- Verified metadata: Sridi, Abir; Bilokon, Paul (2023), SSRN Electronic Journal,
  DOI `10.2139/ssrn.4572108`; arXiv:2309.07843.
- Stable identifiers: <https://doi.org/10.2139/ssrn.4572108>;
  <https://arxiv.org/abs/2309.07843>.
- Category: differential-learning Heston calibration.
- Problem solved: uses differential machine learning to accelerate Heston calibration.
- Model/data/task: vanilla European puts under Heston in abstract; forward surrogate
  then inverse calibration; simulation experiments.
- Parameter truth: sampled Heston settings are implicit, but abstract does not report
  a range-scaled truth-recovery protocol.
- Architecture: differential machine learning with feed-forward comparisons and
  regularization experiments.
- PDE residual: none.
- Metrics/noise/OOD: abstract reports faster computation, regularization/generalization
  experiments, and DML versus ordinary DL comparison; no OOD protocol.
- Limitation: working-paper record and single-factor Heston scope.
- Relationship: recent Heston-specific acceleration baseline; this paper studies two
  variance factors and distinguishes repricing loss from PDE residual.

### `baschetti2024deep`

- Verified metadata: Baschetti, Fabio; Bormetti, Giacomo; Rossi, Pietro (2024),
  Quantitative Finance 24(9), 1263--1285, DOI
  `10.1080/14697688.2024.2332375`.
- Stable identifier: <https://doi.org/10.1080/14697688.2024.2332375>.
- Category: neural calibration design.
- Problem solved: combines grid-based robustness with pointwise calibration using
  random grids.
- Model/data/task: rough Bergomi and Heston in abstract; Monte Carlo and empirical
  experiments; inverse calibration.
- Parameter truth: generated settings exist in Monte Carlo, but the retrieved abstract
  gives no known-truth recovery threshold.
- Architecture: neural network trained on surfaces represented on random grids.
- PDE residual: none.
- Metrics/noise/OOD: abstract emphasizes robustness inherited from grid methods and
  avoiding interpolation/extrapolation; no standardized OOD split.
- Limitation: representation and models differ from frozen R2/DH test.
- Relationship: supports evaluating input-grid design as a scientific choice, exactly
  why R2/R3 was frozen before primary training.

### `biagini2024approximation`

- Verified metadata: Biagini, Francesca; Gonon, Lukas; Walter, Niklas (2024), SIAM
  Journal on Financial Mathematics 15(3), 734--784, DOI `10.1137/23M1606769`.
- Stable identifier: <https://doi.org/10.1137/23M1606769>.
- Category: approximation theory for neural calibration/pricing.
- Problem solved: quantitative error bounds for DNN approximators of option prices as
  functions of parameters/payoffs/initial conditions.
- Model/data/task: Markovian stochastic volatility and rough Bergomi; theoretical
  forward-map approximation.
- Parameter truth: theoretical parameter dependence, not finite-sample recovery.
- Architecture: DNN approximation theory.
- PDE residual: none.
- Metrics/noise/OOD: approximation error and dimension scaling in abstract; no noisy
  quote or OOD experiment.
- Limitation: existence/approximation guarantees do not establish practical
  identifiability.
- Relationship: separates learnability of the forward map from invertibility of the
  inverse map.

### `zhang2025calibrating`

- Verified metadata: Zhang, Chen; Amici, Giovanni; Morandotti, Marco (2025),
  Decisions in Economics and Finance, DOI `10.1007/s10203-025-00558-1`; arXiv
  2407.15536.
- Stable identifiers: <https://doi.org/10.1007/s10203-025-00558-1>;
  <https://arxiv.org/abs/2407.15536>.
- Category: recent Heston differential-network calibration.
- Problem solved: gradient-based deep calibration using learned prices and parameter
  sensitivities.
- Model/data/task: Heston; selected equity markets in abstract; learned forward map +
  gradient-based inverse calibration.
- Parameter truth: market calibration lacks truth; abstract does not give a synthetic
  recovery threshold.
- Architecture: deep differential network learning value and parameter derivatives.
- PDE residual: none.
- Metrics/noise/OOD: abstract reports calibration accuracy/time advantages versus
  nondifferential networks/global optimizers; no OOD protocol.
- Limitation: single-factor Heston and different market protocol.
- Relationship: shows recent derivative-aware amortization; this project instead
  evaluates direct inverse models against a strong traditional frontier.

### `cuchiero2025robust`

- Verified metadata: Cuchiero, Christa; Flonner, Eva; Kurt, Kevin (2025), Journal of
  Computational Finance, DOI `10.21314/jcf.2025.009`; arXiv 2409.06551.
- Stable identifiers: <https://doi.org/10.21314/jcf.2025.009>;
  <https://arxiv.org/abs/2409.06551>.
- Category: robust/neural-SDE calibration.
- Problem solved: Bayesian posterior over neural-SDE calibrations for robust bounds on
  implied volatility surfaces.
- Model/data/task: neural SDEs; historical time series and option data in abstract;
  inverse calibration.
- Parameter truth: unavailable in market data; posterior expresses uncertainty rather
  than recovering known truth.
- Architecture: neural SDE prior/posterior with Langevin-type sampling.
- PDE residual: none stated.
- Metrics/noise/OOD: abstract frames robustness through posterior mixture bounds and
  joint historical/option information; no deterministic OOD split.
- Limitation: neural-SDE class differs structurally from constrained DH.
- Relationship: supports uncertainty-aware interpretation of non-identifiable
  calibrations; this paper retains equivalence-class evidence rather than claiming a
  unique vector.

### `wu2025efficient`

- Verified metadata: Wu, Keyuan; Zhong, Tenghan; Ouyang, Yuxuan (2025),
  arXiv:2510.19126.
- Stable identifier: <https://arxiv.org/abs/2510.19126>.
- Category: recent Fourier-transform calibration acceleration.
- Problem solved: structure-preserving acceleration for characteristic-function
  models with jumps.
- Model/data/task: rough volatility with tempered-stable jumps; VIX options in
  abstract; forward pricing map approximation then global-to-local calibration.
- Parameter truth: market calibration lacks truth.
- Architecture: small neural approximation of market-dependent remainder with GPU
  precomputation.
- PDE residual: none.
- Metrics/noise/OOD: abstract reports speed/accuracy for VIX dynamics; no OOD claim.
- Limitation: volatility derivative/jump model differs from R2 equity-surface task.
- Relationship: confirms current interest in hybrid analytic/neural calibration while
  keeping the analytic transform central, as Model2 does without a PDE term.

### `raissi2019physics`

- Verified metadata: Raissi, Maziar; Perdikaris, Paris; Karniadakis, George E.
  (2019), Journal of Computational Physics 378, 686--707, DOI
  `10.1016/j.jcp.2018.10.045`.
- Stable identifier: <https://doi.org/10.1016/j.jcp.2018.10.045>.
- Category: PINN foundation.
- Problem solved: neural learning for forward and inverse nonlinear PDE problems.
- Model/data/task: PDE-constrained scientific machine learning.
- Parameter truth: depends on problem; no finance-specific claim made.
- Architecture: PINN data/model-equation loss.
- PDE residual: yes.
- Metrics/noise/OOD: title establishes forward/inverse framing; no option-calibration
  metric inferred.
- Limitation: generic methodology does not solve option-surface non-identifiability.
- Relationship: defines the standard that Model3 must meet; Model2 has no such PDE
  residual.

### `karniadakis2021physics`

- Verified metadata: Karniadakis, George Em; Kevrekidis, Ioannis G.; Lu, Lu;
  Perdikaris, Paris; Wang, Sifan; Yang, Liu (2021), Nature Reviews Physics 3(6),
  422--440, DOI `10.1038/s42254-021-00314-5`.
- Stable identifier: <https://doi.org/10.1038/s42254-021-00314-5>.
- Category: physics-informed machine-learning review.
- Problem solved: reviews incorporation of physical structure into ML across
  forward/inverse settings.
- Model/data/task: review across scientific-machine-learning tasks.
- Parameter truth: problem-dependent; no finance claim made.
- Architecture: physics-informed ML frameworks.
- PDE residual: includes PDE-constrained formulations.
- Metrics/noise/OOD: review context only; no benchmark values asserted.
- Limitation: broad survey rather than a calibration protocol.
- Relationship: situates genuine PDE-informed learning separately from soft economic
  constraints or repricing losses.

### `cuomo2022scientific`

- Verified metadata: Cuomo, Salvatore; Di Cola, Vincenzo Schiano; Giampaolo, Fabio;
  Rozza, Gianluigi; Raissi, Maziar; Piccialli, Francesco (2022), Journal of
  Scientific Computing 92(3), article 88, DOI `10.1007/s10915-022-01939-z`.
- Stable identifier: <https://doi.org/10.1007/s10915-022-01939-z>.
- Category: recent PINN review/best practices.
- Problem solved: reviews PINN variants, losses, architectures, optimization, and
  disadvantages.
- Model/data/task: scientific machine-learning review.
- Parameter truth: problem-dependent; no finance claim made.
- Architecture: vanilla PINN and variants.
- PDE residual: yes.
- Metrics/noise/OOD: publisher abstract describes multi-task fitting of observations
  while reducing a PDE residual and discusses advantages/disadvantages; no option
  benchmark inferred.
- Limitation: general review, not a DH inverse benchmark.
- Relationship: supports requiring explicit residual/operator validation and honest
  cost reporting for Model3.

### `wang2022when`

- Verified metadata: Wang, Sifan; Yu, Xinling; Perdikaris, Paris (2022), Journal of
  Computational Physics 449, 110768, DOI `10.1016/j.jcp.2021.110768`.
- Stable identifier: <https://doi.org/10.1016/j.jcp.2021.110768>.
- Category: PINN training-failure analysis.
- Problem solved: explains when and why PINNs fail to train via neural tangent kernel
  perspective.
- Model/data/task: PINN optimization analysis.
- Parameter truth: not applicable.
- Architecture: PINNs analyzed through NTK.
- PDE residual: yes.
- Metrics/noise/OOD: training-pathology diagnosis; no option-noise benchmark.
- Limitation: mechanism-focused study, not a finance calibration result.
- Relationship: motivates Model3 readiness checks for finite gradients, residual
  identities, stability, and cost before any result.

### `sirignano2018dgm`

- Verified metadata: Sirignano, Justin; Spiliopoulos, Konstantinos (2018), Journal of
  Computational Physics 375, 1339--1364, DOI `10.1016/j.jcp.2018.08.029`.
- Stable identifier: <https://doi.org/10.1016/j.jcp.2018.08.029>.
- Category: deep PDE solving foundation.
- Problem solved: DGM algorithm for solving partial differential equations.
- Model/data/task: PDE solver methodology.
- Parameter truth: not applicable.
- Architecture: deep learning PDE algorithm.
- PDE residual: yes.
- Metrics/noise/OOD: no finance-specific values asserted from metadata alone.
- Limitation: forward PDE solver, not an option-surface inverse benchmark.
- Relationship: part of the PDE-learning lineage from which Model3 is distinguished
  from repricing-only learning.

### `lagaris1998artificial`

- Verified metadata: Lagaris, Isaac E.; Likas, Aristidis; Fotiadis, Dimitrios I.
  (1998), IEEE Transactions on Neural Networks 9(5), 987--1000, DOI
  `10.1109/72.712178`.
- Stable identifier: <https://doi.org/10.1109/72.712178>.
- Category: neural PDE-solving precursor.
- Problem solved: artificial neural networks for ordinary/partial differential
  equations.
- Model/data/task: general ODE/PDE solutions.
- Parameter truth: not applicable.
- Architecture: neural trial solutions.
- PDE residual: yes in the broad neural-PDE sense.
- Metrics/noise/OOD: no finance-specific values asserted from metadata alone.
- Limitation: predates modern PINNs and does not address option calibration.
- Relationship: shows that equation-residual neural methods precede modern PINNs;
  recency is not the criterion separating them from repricing losses.

### `kim2022physics`

- Verified metadata: Kim, Soohan; Yun, Seok-Bae; Bae, Hyeong-Ohk; Lee, Muhyun; Hong,
  Youngjoon (2022), arXiv:2209.10771.
- Stable identifier: <https://arxiv.org/abs/2209.10771>.
- Category: recent physics-informed volatility-surface computation.
- Problem solved: predicts volatility surface dynamics with a physics-informed
  convolutional transformer.
- Model/data/task: Black-Scholes-based physics setup in abstract; surface prediction;
  numerical comparison.
- Parameter truth: not a known-parameter inverse-recovery study.
- Architecture: PINN plus convolutional transformer.
- PDE residual: physics-informed formulation.
- Metrics/noise/OOD: abstract reports comparison against PINN/ConvLSTM/self-attention
  baselines; no parameter-noise/OOD recovery gate.
- Limitation: surface prediction differs from ten-parameter DH inversion.
- Relationship: illustrates physics-informed surface learning while emphasizing that
  pricing/surface accuracy still differs from parameter recovery.

### `dhiman2023physics`

- Verified metadata: Dhiman, Ashish; Hu, Yibei (2023), arXiv:2312.06711.
- Stable identifier: <https://arxiv.org/abs/2312.06711>.
- Category: recent PINN option pricing.
- Problem solved: applies PINNs to Black-Scholes American/European option pricing.
- Model/data/task: Black-Scholes PDE; simulated and real market data in abstract;
  primarily forward pricing.
- Parameter truth: not framed as known-vector DH recovery.
- Architecture: PINN with architecture/training experiments.
- PDE residual: yes.
- Metrics/noise/OOD: abstract reports comparison to analytical/numerical benchmarks
  and convergence/stability experiments; no parameter-noise/OOD gate.
- Limitation: simpler dynamics and forward target.
- Relationship: clarifies why a genuine PDE residual is categorically different from
  Model2 and why Model3 remains unreported until Stage A/B run.

## Explicit distinctions

- **Pricing versus inverse calibration:** `heston1993closed`, `duffie2000transform`,
  `albrecher2007little`, `sirignano2018dgm`, and `dhiman2023physics` concern forward
  solution/pricing machinery; `aitsahalia2007maximum`, `bollerslev2002estimating`,
  `liu2019neural`, `bayer2019deep`, `horvath2021deep`, `sridi2023applying`,
  `baschetti2024deep`, and `zhang2025calibrating` perform inverse estimation or
  calibration.
- **Repricing-informed versus PDE-informed:** Model2's differentiable Fourier
  repricing objective is closer to learned/surrogate pricing objectives than to
  `raissi2019physics`, `sirignano2018dgm`, or the planned Model3 autograd residual.
  None of these citations permits calling Model2 a PINN.
- **Fit versus recovery:** empirical papers compare observable fit; only a synthetic
  protocol with stored generating vectors can evaluate known-truth recovery. This
  distinction is retained throughout the manuscript.
- **Single Heston versus multifactor:** `heston1993closed`, `albrecher2007little`,
  `sridi2023applying`, and `zhang2025calibrating` are Heston-specific; CHJ motivates
  multifactor structure, while this project fixes a canonical two-factor DH contract.

## Unresolved bibliography gaps

- The Little Heston Trap item lacks a discovered Crossref DOI; its existing
  repository-referenced PDF URL is retained.
- Several formal records did not expose abstracts to Crossref, so their inventory
  rows deliberately avoid metric-level claims.
- The following requested areas need later verification if the final paper expands:
  dedicated DH identifiability studies, local-stochastic volatility calibration,
  noisy/incomplete option-quote benchmarks, and additional peer-reviewed PINN option
  pricing/calibration articles.
