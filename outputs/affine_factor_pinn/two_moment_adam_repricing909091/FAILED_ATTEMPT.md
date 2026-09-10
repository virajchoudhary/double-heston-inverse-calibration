# Failed development audit attempt

The first run of `audit_factor_repricing.py` stopped at its strict price-score
equality assertion for case 0, neural prices. The replay script had constructed
the JSON parameter list as a default float32 tensor, while the assessment used
float64. No assessment, fitted parameters, tolerance or source dataset was
modified. The script was corrected to construct explicit float64 tensors; its
subsequent L-BFGS checkpoint repricing audit reproduced every archived price/IV
metric exactly. This directory is retained as a failed audit attempt, not a
passed audit or model-training result.
