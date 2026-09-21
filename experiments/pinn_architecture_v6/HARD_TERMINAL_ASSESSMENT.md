# ARCH_D_HARD_TERMINAL: already implemented; not a distinct trained arm

The inherited network returns c=C/(K exp(-r tau)). At tau=0 the existing
price function returns max(exp(x)-1,0) analytically, with x=log(F/K), F=S at
expiry. That is exactly max(S-K,0)/K. Learned weights cannot change this branch.
The audit checks both sides of the strike and the strike itself.

For positive tau, a positive predicted total variance feeds the smooth analytic
Black transform. The PDE is differentiated only at strictly positive tau. The
tau=0 payoff kink is not incorrectly differentiated as a smooth classical PDE
solution. No smoothing bias is added to the payoff.

Replacing this with payoff(S)+tau*N(S,tau), with smooth N, leaves the spatial
payoff kink present at every positive tau. Its second spatial derivative
contains the payoff distributional term, so that naive proposal does not repair
the current solver. There is also no terminal error below exact zero to improve.

Finite training edges x=±.36 are not the asymptotic boundaries S=0 and S=infinity.
We measure their exact-teacher error rather than impose zero/intrinsic price
there. A separate generic distance-function construction is therefore not
justified here. This is a mathematical assessment, not a fabricated additional
ablation result. All actual trained arms retain the existing hard condition.

Motivation: Sukumar and Srivastava, https://arxiv.org/abs/2104.08426.
