# Regular Heston PINN development evidence

**DEVELOPMENT VALIDATION — NOT AN UNSEEN FINAL TEST**

All results here use synthetic Heston development data. These four recovery-validation cases are repeatedly used to select checkpoints, so their scores do not establish unseen-test generalisation. Single and Double Heston each use their own generating model and parameter vectors; the model families are not fitted to a common NSE dataset in this report.

The regular PINN learns a smooth implied-total-variance function constrained by the pricing PDE. Inverse calibration optimizes the frozen network's IV predictions. Exact Fourier prices supply synthetic targets and training sensitivities; the calibration objective evaluates the neural network with its learned weights promoted to float64.

## Selected checkpoints

Each IV/recovery value below belongs to the selected checkpoint step. The latest recorded step is shown separately. An incomplete status means no completion artifact was found; this report does not inspect whether a process is running.

| Run | Status | Latest / selected step | Selected IV RMSE | Selected parameter RMSE | All-parameter cases passing |
|---|---|---:|---:|---:|---:|
| single_standard_s17 | complete | 12000 / 6000 | 0.00079039 | 0.025484 | 2/4 |
| single_sobolev_s17 | complete | 12000 / 12000 | 0.0001985 | 0.012533 | 4/4 |
| double_standard_s17 | complete | 12000 / 12000 | 0.00055274 | 0.16327 | 0/4 |
| double_sobolev_s17 | complete | 12000 / 12000 | 0.00034849 | 0.12426 | 0/4 |

The case gate requires every positive parameter to be within 5% of its generating value and each correlation to be within 0.05 in absolute units. Optimizer success is recorded separately and does not imply that recovery passed. The training selection score divides positive-parameter errors by their true magnitudes and correlation errors by 0.5; it is therefore different from the gate.

## Training histories

![Synthetic validation IV error](validation_iv_history.png)

Interpretation: lower IV error means the learned forward function matches synthetic option-implied volatility more closely. A star marks the selected checkpoint. This alone does not show that the inverse problem recovers its generating parameters.

![Validation parameter recovery](validation_recovery_history.png)

Interpretation: lower values mean smaller parameter errors on the same four development surfaces. A star marks checkpoint selection. Repeated use of these cases makes this development evidence; independent final cases are required to assess reliability. Missing/nonfinite history values are counted in the summary and cannot appear on a logarithmic plot.

## Training data and continuation

| Run | Usable training / candidates | Usable IV validation | Collocation pool | Sensitivity weight | Resumed weights |
|---|---:|---:|---:|---:|---|
| single_standard_s17 | 131065 / 131072 | 16384 | 18000 | 0 | No |
| single_sobolev_s17 | 131065 / 131072 | 16384 | 18000 | 0.2 | No |
| double_standard_s17 | 131072 / 131072 | 16384 | 18000 | 0 | No |
| double_sobolev_s17 | 131072 / 131072 | 16384 | 18000 | 0.2 | No |

A resumed run continues an existing set of network weights. It does not count as an independent initialization. Training uses AdamW weight decay, PDE/shape penalties, and optional parameter-sensitivity supervision; configuration files record their exact weights.

## Every selected validation parameter

The full machine-readable values are in [selected_parameter_details.csv](selected_parameter_details.csv). Gate units are absolute error divided by the allowed tolerance: values at or below 1 pass. The signed scaled-error column uses the training selection scale described above.

<details><summary>single_standard_s17: all selected parameter estimates</summary>

| Case | Factor | Parameter | Truth | Estimate | Signed scaled error | Gate units | Pass |
|---:|---:|---|---:|---:|---:|---:|---|
| 0 | 1 | kappa | 0.8087271 | 0.8306549 | 0.027114 | 0.54228 | Yes |
| 0 | 1 | theta | 0.1915743 | 0.1907404 | -0.0043525 | 0.087049 | Yes |
| 0 | 1 | sigma | 0.4163759 | 0.4196022 | 0.0077483 | 0.15497 | Yes |
| 0 | 1 | rho | -0.559658 | -0.5641436 | -0.0089713 | 0.089713 | Yes |
| 0 | 1 | v0 | 0.1030049 | 0.1034325 | 0.0041509 | 0.083019 | Yes |
| 1 | 1 | kappa | 4.732737 | 4.765892 | 0.0070053 | 0.14011 | Yes |
| 1 | 1 | theta | 0.05442037 | 0.05475351 | 0.0061216 | 0.12243 | Yes |
| 1 | 1 | sigma | 0.1932664 | 0.2052857 | 0.062191 | 1.2438 | No |
| 1 | 1 | rho | 0.007835175 | 0.002462308 | -0.010746 | 0.10746 | Yes |
| 1 | 1 | v0 | 0.1207404 | 0.120926 | 0.001537 | 0.03074 | Yes |
| 2 | 1 | kappa | 0.8074826 | 0.7885015 | -0.023506 | 0.47013 | Yes |
| 2 | 1 | theta | 0.08537082 | 0.08479666 | -0.0067254 | 0.13451 | Yes |
| 2 | 1 | sigma | 0.1043025 | 0.09631204 | -0.076609 | 1.5322 | No |
| 2 | 1 | rho | 0.02770397 | 0.02299496 | -0.009418 | 0.09418 | Yes |
| 2 | 1 | v0 | 0.123547 | 0.1236233 | 0.00061797 | 0.012359 | Yes |
| 3 | 1 | kappa | 1.051485 | 1.021901 | -0.028135 | 0.56271 | Yes |
| 3 | 1 | theta | 0.1704532 | 0.1739925 | 0.020764 | 0.41529 | Yes |
| 3 | 1 | sigma | 0.4387008 | 0.444639 | 0.013536 | 0.27072 | Yes |
| 3 | 1 | rho | 0.1149743 | 0.1130831 | -0.0037823 | 0.037823 | Yes |
| 3 | 1 | v0 | 0.06211292 | 0.06244611 | 0.0053642 | 0.10728 | Yes |

</details>

<details><summary>single_sobolev_s17: all selected parameter estimates</summary>

| Case | Factor | Parameter | Truth | Estimate | Signed scaled error | Gate units | Pass |
|---:|---:|---|---:|---:|---:|---:|---|
| 0 | 1 | kappa | 0.8087271 | 0.8113084 | 0.0031919 | 0.063838 | Yes |
| 0 | 1 | theta | 0.1915743 | 0.1909917 | -0.0030412 | 0.060823 | Yes |
| 0 | 1 | sigma | 0.4163759 | 0.4151102 | -0.0030399 | 0.060797 | Yes |
| 0 | 1 | rho | -0.559658 | -0.561207 | -0.003098 | 0.03098 | Yes |
| 0 | 1 | v0 | 0.1030049 | 0.1029598 | -0.00043762 | 0.0087524 | Yes |
| 1 | 1 | kappa | 4.732737 | 4.769938 | 0.0078602 | 0.1572 | Yes |
| 1 | 1 | theta | 0.05442037 | 0.05451292 | 0.0017008 | 0.034015 | Yes |
| 1 | 1 | sigma | 0.1932664 | 0.1957719 | 0.012964 | 0.25928 | Yes |
| 1 | 1 | rho | 0.007835175 | 0.009006316 | 0.0023423 | 0.023423 | Yes |
| 1 | 1 | v0 | 0.1207404 | 0.1208842 | 0.0011907 | 0.023815 | Yes |
| 2 | 1 | kappa | 0.8074826 | 0.817177 | 0.012006 | 0.24011 | Yes |
| 2 | 1 | theta | 0.08537082 | 0.0854254 | 0.0006393 | 0.012786 | Yes |
| 2 | 1 | sigma | 0.1043025 | 0.09919461 | -0.048972 | 0.97944 | Yes |
| 2 | 1 | rho | 0.02770397 | 0.02845563 | 0.0015033 | 0.015033 | Yes |
| 2 | 1 | v0 | 0.123547 | 0.1235816 | 0.00028009 | 0.0056019 | Yes |
| 3 | 1 | kappa | 1.051485 | 1.034074 | -0.016559 | 0.33117 | Yes |
| 3 | 1 | theta | 0.1704532 | 0.1714949 | 0.0061117 | 0.12223 | Yes |
| 3 | 1 | sigma | 0.4387008 | 0.4381223 | -0.0013186 | 0.026372 | Yes |
| 3 | 1 | rho | 0.1149743 | 0.1140118 | -0.0019249 | 0.019249 | Yes |
| 3 | 1 | v0 | 0.06211292 | 0.06217794 | 0.0010467 | 0.020934 | Yes |

</details>

<details><summary>double_standard_s17: all selected parameter estimates</summary>

| Case | Factor | Parameter | Truth | Estimate | Signed scaled error | Gate units | Pass |
|---:|---:|---|---:|---:|---:|---:|---|
| 0 | 1 | kappa | 1.649795 | 1.472286 | -0.10759 | 2.1519 | No |
| 0 | 1 | theta | 0.03939343 | 0.0431885 | 0.096338 | 1.9268 | No |
| 0 | 1 | sigma | 0.2058855 | 0.2124279 | 0.031777 | 0.63554 | Yes |
| 0 | 1 | rho | -0.2464467 | -0.1882223 | 0.11645 | 1.1645 | No |
| 0 | 1 | v0 | 0.01845716 | 0.02271862 | 0.23088 | 4.6177 | No |
| 0 | 2 | kappa | 6.96594 | 6.65108 | -0.0452 | 0.904 | Yes |
| 0 | 2 | theta | 0.04993438 | 0.046942 | -0.059926 | 1.1985 | No |
| 0 | 2 | sigma | 0.5779429 | 0.6046104 | 0.046142 | 0.92284 | Yes |
| 0 | 2 | rho | -0.007524568 | -0.005068187 | 0.0049128 | 0.049128 | Yes |
| 0 | 2 | v0 | 0.0333573 | 0.02912442 | -0.1269 | 2.5379 | No |
| 1 | 1 | kappa | 1.650889 | 1.683099 | 0.01951 | 0.39021 | Yes |
| 1 | 1 | theta | 0.04780292 | 0.04438719 | -0.071454 | 1.4291 | No |
| 1 | 1 | sigma | 0.1600093 | 0.1747814 | 0.09232 | 1.8464 | No |
| 1 | 1 | rho | -0.2751483 | -0.2767193 | -0.0031419 | 0.031419 | Yes |
| 1 | 1 | v0 | 0.04971122 | 0.04714478 | -0.051627 | 1.0325 | No |
| 1 | 2 | kappa | 4.620667 | 4.42971 | -0.041327 | 0.82654 | Yes |
| 1 | 2 | theta | 0.03638423 | 0.0396592 | 0.090011 | 1.8002 | No |
| 1 | 2 | sigma | 0.1686148 | 0.08891757 | -0.47266 | 9.4532 | No |
| 1 | 2 | rho | -0.1250234 | -0.1657521 | -0.081457 | 0.81457 | Yes |
| 1 | 2 | v0 | 0.0202971 | 0.02285857 | 0.1262 | 2.524 | No |
| 2 | 1 | kappa | 0.9412414 | 0.8471275 | -0.099989 | 1.9998 | No |
| 2 | 1 | theta | 0.0170392 | 0.01489537 | -0.12582 | 2.5164 | No |
| 2 | 1 | sigma | 0.0964517 | 0.07233518 | -0.25004 | 5.0007 | No |
| 2 | 1 | rho | -0.5605928 | -0.6898817 | -0.25858 | 2.5858 | No |
| 2 | 1 | v0 | 0.03324041 | 0.03132049 | -0.057759 | 1.1552 | No |
| 2 | 2 | kappa | 6.113868 | 6.219192 | 0.017227 | 0.34454 | Yes |
| 2 | 2 | theta | 0.03023963 | 0.03161712 | 0.045552 | 0.91104 | Yes |
| 2 | 2 | sigma | 0.1975487 | 0.242337 | 0.22672 | 4.5344 | No |
| 2 | 2 | rho | -0.04455256 | -0.09811555 | -0.10713 | 1.0713 | No |
| 2 | 2 | v0 | 0.0148063 | 0.01666912 | 0.12581 | 2.5163 | No |
| 3 | 1 | kappa | 0.331245 | 0.2590293 | -0.21801 | 4.3603 | No |
| 3 | 1 | theta | 0.04721769 | 0.06189065 | 0.31075 | 6.215 | No |
| 3 | 1 | sigma | 0.09961979 | 0.07467419 | -0.25041 | 5.0082 | No |
| 3 | 1 | rho | -0.06578391 | -0.02879606 | 0.073976 | 0.73976 | Yes |
| 3 | 1 | v0 | 0.03544835 | 0.04733229 | 0.33525 | 6.7049 | No |
| 3 | 2 | kappa | 8.005495 | 8.09221 | 0.010832 | 0.21664 | Yes |
| 3 | 2 | theta | 0.03985632 | 0.02799287 | -0.29766 | 5.9531 | No |
| 3 | 2 | sigma | 0.3634601 | 0.3890244 | 0.070336 | 1.4067 | No |
| 3 | 2 | rho | 0.2812903 | 0.3150603 | 0.06754 | 0.6754 | Yes |
| 3 | 2 | v0 | 0.08637276 | 0.07455027 | -0.13688 | 2.7375 | No |

</details>

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

## Reproducibility

[evidence.json](evidence.json) contains the input snapshots, selected-checkpoint hashes, training manifests and exact report inputs. [run_summary.csv](run_summary.csv) and [selected_case_gates.csv](selected_case_gates.csv) retain run and case-level outcomes, including missing or failed recovery records.
