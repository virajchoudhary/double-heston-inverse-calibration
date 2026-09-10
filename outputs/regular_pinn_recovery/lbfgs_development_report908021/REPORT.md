# Regular Heston PINN development evidence

**DEVELOPMENT VALIDATION — NOT AN UNSEEN FINAL TEST**

All results here use synthetic Heston development data. These four recovery-validation cases are repeatedly used to select checkpoints, so their scores do not establish unseen-test generalisation. Single and Double Heston each use their own generating model and parameter vectors; the model families are not fitted to a common NSE dataset in this report.

The regular PINN learns a smooth implied-total-variance function constrained by the pricing PDE. Inverse calibration optimizes the frozen network's IV predictions. Exact Fourier prices supply synthetic targets and training sensitivities; the calibration objective evaluates the neural network with its learned weights promoted to float64.

## Selected checkpoints

Each IV/recovery value below belongs to the selected checkpoint step. The latest recorded step is shown separately. An incomplete status means no completion artifact was found; this report does not inspect whether a process is running.

| Run | Status | Latest / selected step | Selected IV RMSE | Selected parameter RMSE | All-parameter cases passing |
|---|---|---:|---:|---:|---:|
| double_sobolev_s17 | complete | 12000 / 12000 | 0.00034849 | 0.12426 | 0/4 |
| double_lbfgs908021 | complete | 1000 / 0 | 0.00034849 | 0.12426 | 0/4 |

The case gate requires every positive parameter to be within 5% of its generating value and each correlation to be within 0.05 in absolute units. Optimizer success is recorded separately and does not imply that recovery passed. The training selection score divides positive-parameter errors by their true magnitudes and correlation errors by 0.5; it is therefore different from the gate.

## Training histories

![Synthetic validation IV error](validation_iv_history.png)

Interpretation: lower IV error means the learned forward function matches synthetic option-implied volatility more closely. A star marks the selected checkpoint. This alone does not show that the inverse problem recovers its generating parameters.

![Validation parameter recovery](validation_recovery_history.png)

Interpretation: lower values mean smaller parameter errors on the same four development surfaces. A star marks checkpoint selection. Repeated use of these cases makes this development evidence; independent final cases are required to assess reliability. Missing/nonfinite history values are counted in the summary and cannot appear on a logarithmic plot.

## Training data and continuation

| Run | Usable training / candidates | Usable IV validation | Collocation pool | Sensitivity weight | Resumed weights |
|---|---:|---:|---:|---:|---|
| double_sobolev_s17 | 131072 / 131072 | 16384 | 18000 | 0.2 | No |
| double_lbfgs908021 | 131072 / 131072 | 16384 | 18000 | 0.2 | outputs/regular_pinn_recovery/double_sobolev_s17 |

A resumed run continues existing network weights and is not an independent initialization. AdamW runs use their declared weight decay; L-BFGS runs use a fixed deterministic objective without weight decay, clipping or resampling. A plotted L-BFGS iteration is not computationally equivalent to an Adam step. PDE/shape penalties and optional parameter-sensitivity supervision are recorded in each effective configuration.

L-BFGS may use a fixed subset of the available training labels; fixed_training_subset in run_summary.csv gives the actual anchor count. The table's training count describes the source dataset, not necessarily all labels used in that fine-tuning phase.

Grouped-surface runs additionally use 1,024 independent synthetic training parameter surfaces (126 quotes each, 129,024 extra quotes). The control and parameter-aware arm use identical extra data. Their parameter-aware weights are recorded in run_summary.csv. These labels are not supplied to the inverse calibrator. Double Heston factor 1 is stored as the slow factor and factor 2 as the fast factor throughout this experiment.

## Every selected validation parameter

The full machine-readable values are in [selected_parameter_details.csv](selected_parameter_details.csv). Gate units are absolute error divided by the allowed tolerance: values at or below 1 pass. The signed scaled-error column uses the training selection scale described above.

<details><summary>double_sobolev_s17: all selected parameter estimates</summary>

| Case | Factor | Parameter | Truth | Estimate | Signed scaled error | Gate units | Pass |
|---:|---:|---|---:|---:|---:|---:|---|
| 0 | 1 | kappa | 1.649795 | 1.8093 | 0.096682 | 1.9336 | No |
| 0 | 1 | theta | 0.03939343 | 0.03930105 | -0.0023451 | 0.046902 | Yes |
| 0 | 1 | sigma | 0.2058855 | 0.2770408 | 0.34561 | 6.9121 | No |
| 0 | 1 | rho | -0.2464467 | -0.2008201 | 0.091253 | 0.91253 | Yes |
| 0 | 1 | v0 | 0.01845716 | 0.01627565 | -0.11819 | 2.3639 | No |
| 0 | 2 | kappa | 6.96594 | 7.686003 | 0.10337 | 2.0674 | No |
| 0 | 2 | theta | 0.04993438 | 0.04992448 | -0.00019833 | 0.0039665 | Yes |
| 0 | 2 | sigma | 0.5779429 | 0.5646666 | -0.022972 | 0.45943 | Yes |
| 0 | 2 | rho | -0.007524568 | -0.008954089 | -0.002859 | 0.02859 | Yes |
| 0 | 2 | v0 | 0.0333573 | 0.03554895 | 0.065702 | 1.314 | No |
| 1 | 1 | kappa | 1.650889 | 1.328145 | -0.1955 | 3.9099 | No |
| 1 | 1 | theta | 0.04780292 | 0.03885968 | -0.18709 | 3.7417 | No |
| 1 | 1 | sigma | 0.1600093 | 0.1578023 | -0.013793 | 0.27587 | Yes |
| 1 | 1 | rho | -0.2751483 | -0.2775395 | -0.0047824 | 0.047824 | Yes |
| 1 | 1 | v0 | 0.04971122 | 0.040495 | -0.1854 | 3.7079 | No |
| 1 | 2 | kappa | 4.620667 | 4.613913 | -0.0014616 | 0.029232 | Yes |
| 1 | 2 | theta | 0.03638423 | 0.04537536 | 0.24712 | 4.9423 | No |
| 1 | 2 | sigma | 0.1686148 | 0.1709038 | 0.013575 | 0.2715 | Yes |
| 1 | 2 | rho | -0.1250234 | -0.1672165 | -0.084386 | 0.84386 | Yes |
| 1 | 2 | v0 | 0.0202971 | 0.0294741 | 0.45213 | 9.0427 | No |
| 2 | 1 | kappa | 0.9412414 | 0.8936113 | -0.050603 | 1.0121 | No |
| 2 | 1 | theta | 0.0170392 | 0.01550357 | -0.090123 | 1.8025 | No |
| 2 | 1 | sigma | 0.0964517 | 0.1021539 | 0.05912 | 1.1824 | No |
| 2 | 1 | rho | -0.5605928 | -0.5652586 | -0.0093316 | 0.093316 | Yes |
| 2 | 1 | v0 | 0.03324041 | 0.03151174 | -0.052005 | 1.0401 | No |
| 2 | 2 | kappa | 6.113868 | 6.21674 | 0.016826 | 0.33652 | Yes |
| 2 | 2 | theta | 0.03023963 | 0.03161097 | 0.045349 | 0.90698 | Yes |
| 2 | 2 | sigma | 0.1975487 | 0.1749832 | -0.11423 | 2.2845 | No |
| 2 | 2 | rho | -0.04455256 | -0.04257038 | 0.0039644 | 0.039644 | Yes |
| 2 | 2 | v0 | 0.0148063 | 0.01649961 | 0.11436 | 2.2873 | No |
| 3 | 1 | kappa | 0.331245 | 0.3532944 | 0.066565 | 1.3313 | No |
| 3 | 1 | theta | 0.04721769 | 0.04950239 | 0.048386 | 0.96773 | Yes |
| 3 | 1 | sigma | 0.09961979 | 0.108208 | 0.086209 | 1.7242 | No |
| 3 | 1 | rho | -0.06578391 | -0.04916233 | 0.033243 | 0.33243 | Yes |
| 3 | 1 | v0 | 0.03544835 | 0.03775219 | 0.064992 | 1.2998 | No |
| 3 | 2 | kappa | 8.005495 | 7.98997 | -0.0019392 | 0.038785 | Yes |
| 3 | 2 | theta | 0.03985632 | 0.03748938 | -0.059387 | 1.1877 | No |
| 3 | 2 | sigma | 0.3634601 | 0.3556283 | -0.021548 | 0.43096 | Yes |
| 3 | 2 | rho | 0.2812903 | 0.2941989 | 0.025817 | 0.25817 | Yes |
| 3 | 2 | v0 | 0.08637276 | 0.08406193 | -0.026754 | 0.53508 | Yes |

</details>

<details><summary>double_lbfgs908021: all selected parameter estimates</summary>

| Case | Factor | Parameter | Truth | Estimate | Signed scaled error | Gate units | Pass |
|---:|---:|---|---:|---:|---:|---:|---|
| 0 | 1 | kappa | 1.649795 | 1.8093 | 0.096682 | 1.9336 | No |
| 0 | 1 | theta | 0.03939343 | 0.03930105 | -0.0023451 | 0.046902 | Yes |
| 0 | 1 | sigma | 0.2058855 | 0.2770408 | 0.34561 | 6.9121 | No |
| 0 | 1 | rho | -0.2464467 | -0.2008201 | 0.091253 | 0.91253 | Yes |
| 0 | 1 | v0 | 0.01845716 | 0.01627565 | -0.11819 | 2.3639 | No |
| 0 | 2 | kappa | 6.96594 | 7.686003 | 0.10337 | 2.0674 | No |
| 0 | 2 | theta | 0.04993438 | 0.04992448 | -0.00019833 | 0.0039665 | Yes |
| 0 | 2 | sigma | 0.5779429 | 0.5646666 | -0.022972 | 0.45943 | Yes |
| 0 | 2 | rho | -0.007524568 | -0.008954089 | -0.002859 | 0.02859 | Yes |
| 0 | 2 | v0 | 0.0333573 | 0.03554895 | 0.065702 | 1.314 | No |
| 1 | 1 | kappa | 1.650889 | 1.328145 | -0.1955 | 3.9099 | No |
| 1 | 1 | theta | 0.04780292 | 0.03885968 | -0.18709 | 3.7417 | No |
| 1 | 1 | sigma | 0.1600093 | 0.1578023 | -0.013793 | 0.27587 | Yes |
| 1 | 1 | rho | -0.2751483 | -0.2775395 | -0.0047824 | 0.047824 | Yes |
| 1 | 1 | v0 | 0.04971122 | 0.040495 | -0.1854 | 3.7079 | No |
| 1 | 2 | kappa | 4.620667 | 4.613913 | -0.0014616 | 0.029232 | Yes |
| 1 | 2 | theta | 0.03638423 | 0.04537536 | 0.24712 | 4.9423 | No |
| 1 | 2 | sigma | 0.1686148 | 0.1709038 | 0.013575 | 0.2715 | Yes |
| 1 | 2 | rho | -0.1250234 | -0.1672165 | -0.084386 | 0.84386 | Yes |
| 1 | 2 | v0 | 0.0202971 | 0.0294741 | 0.45213 | 9.0427 | No |
| 2 | 1 | kappa | 0.9412414 | 0.8936113 | -0.050603 | 1.0121 | No |
| 2 | 1 | theta | 0.0170392 | 0.01550357 | -0.090123 | 1.8025 | No |
| 2 | 1 | sigma | 0.0964517 | 0.1021539 | 0.05912 | 1.1824 | No |
| 2 | 1 | rho | -0.5605928 | -0.5652586 | -0.0093316 | 0.093316 | Yes |
| 2 | 1 | v0 | 0.03324041 | 0.03151174 | -0.052005 | 1.0401 | No |
| 2 | 2 | kappa | 6.113868 | 6.21674 | 0.016826 | 0.33652 | Yes |
| 2 | 2 | theta | 0.03023963 | 0.03161097 | 0.045349 | 0.90698 | Yes |
| 2 | 2 | sigma | 0.1975487 | 0.1749832 | -0.11423 | 2.2845 | No |
| 2 | 2 | rho | -0.04455256 | -0.04257038 | 0.0039644 | 0.039644 | Yes |
| 2 | 2 | v0 | 0.0148063 | 0.01649961 | 0.11436 | 2.2873 | No |
| 3 | 1 | kappa | 0.331245 | 0.3532944 | 0.066565 | 1.3313 | No |
| 3 | 1 | theta | 0.04721769 | 0.04950239 | 0.048386 | 0.96773 | Yes |
| 3 | 1 | sigma | 0.09961979 | 0.108208 | 0.086209 | 1.7242 | No |
| 3 | 1 | rho | -0.06578391 | -0.04916233 | 0.033243 | 0.33243 | Yes |
| 3 | 1 | v0 | 0.03544835 | 0.03775219 | 0.064992 | 1.2998 | No |
| 3 | 2 | kappa | 8.005495 | 7.98997 | -0.0019392 | 0.038785 | Yes |
| 3 | 2 | theta | 0.03985632 | 0.03748938 | -0.059387 | 1.1877 | No |
| 3 | 2 | sigma | 0.3634601 | 0.3556283 | -0.021548 | 0.43096 | Yes |
| 3 | 2 | rho | 0.2812903 | 0.2941989 | 0.025817 | 0.25817 | Yes |
| 3 | 2 | v0 | 0.08637276 | 0.08406193 | -0.026754 | 0.53508 | Yes |

</details>

## Incomplete or inconsistent evidence

- double_lbfgs908021: Effective L-BFGS settings overlay the explicitly preserved inherited-Adam metadata; see effective_optimizer_settings.json.

## Reproducibility

[evidence.json](evidence.json) contains the input snapshots, selected-checkpoint hashes, training manifests and exact report inputs. [run_summary.csv](run_summary.csv) and [selected_case_gates.csv](selected_case_gates.csv) retain run and case-level outcomes, including missing or failed recovery records.
