# Inherited PDE guard retained

Written before reading any v6 candidate result. In addition to the four v4
price/IV fidelity gates, retain the v5 PDE non-degradation condition: the
mean of per-seed PDE RMSE on the first 256 original collocation points must
not exceed 1.1 times the reproduced current baseline. This is stricter than
using the older C3 baseline. The original four gates are not weakened.
Full development-domain PDE metrics and spot-curve violations are also reported.
