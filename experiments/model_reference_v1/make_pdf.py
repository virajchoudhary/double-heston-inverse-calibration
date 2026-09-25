"""Build the concise equations reference as a PDF (matplotlib mathtext, no LaTeX needed)."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.image import imread

HERE = Path(__file__).resolve().parent
OUT = HERE / 'MODEL_EQUATIONS_REFERENCE.pdf'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'mathtext.fontset': 'dejavusans',
                     'figure.facecolor': 'white', 'savefig.facecolor': 'white'})
W, H = 8.27, 11.69                                    # A4 portrait
L, R, TOP, BOT = .085, .915, .955, .055
INK, ACC, MUT, GRN = '#1a1a1a', '#1f3f8a', '#666666', '#16704f'


class Page:
    def __init__(self, pdf, header=None):
        self.fig = plt.figure(figsize=(W, H)); self.pdf = pdf; self.y = TOP
        if header:
            self.fig.text(L, .978, header, fontsize=7.5, color=MUT)
            self.fig.text(R, .978, 'Double Heston project — equations reference', fontsize=7.5, color=MUT, ha='right')
            self.fig.add_artist(plt.Line2D([L, R], [.9715, .9715], color='#cccccc', lw=.7))

    def gap(self, d=.012): self.y -= d

    def title(self, t, sub=None):
        self.fig.text(L, self.y, t, fontsize=16, color=INK, weight='bold', va='top'); self.y -= .036
        if sub:
            self.fig.text(L, self.y, sub, fontsize=9, color=MUT, va='top'); self.y -= .026

    def section(self, t):
        self.gap(.016)
        self.fig.text(L, self.y, t, fontsize=11, color=ACC, weight='bold', va='top'); self.y -= .0235

    def text(self, t, fs=8.6, color=INK, dy=None, indent=0.):
        n = t.count('\n') + 1
        self.fig.text(L + indent, self.y, t, fontsize=fs, color=color, va='top', linespacing=1.45)
        self.y -= (dy if dy is not None else .0155 * n + .004)

    def eq(self, t, fs=10.5, dy=.034, color=INK):
        self.fig.text(.5, self.y, t, fontsize=fs, color=color, va='top', ha='center'); self.y -= dy

    def table(self, rows, widths, fs=8.2, header=True, dy=.0145):
        for i, row in enumerate(rows):
            x = L
            for cell, w in zip(row, widths):
                self.fig.text(x, self.y, cell, fontsize=fs, va='top',
                              color=INK if i or not header else ACC,
                              weight='bold' if i == 0 and header else 'normal')
                x += w
            self.y -= dy
            if i == 0 and header:
                self.fig.add_artist(plt.Line2D([L, R], [self.y + .009, self.y + .009], color='#dddddd', lw=.6))
                self.y -= .003
        self.y -= .006

    def note(self, t, fs=7.8):
        self.text(t, fs=fs, color=MUT)

    def close(self, n=None):
        if n: self.fig.text(.5, .022, str(n), fontsize=7.5, color=MUT, ha='center')
        self.pdf.savefig(self.fig); plt.close(self.fig)


with PdfPages(OUT) as pdf:
    # ---------------------------------------------------------------- page 1
    p = Page(pdf)
    p.title('Pricing models, conditions and the PINN',
            'Every formula taken from the code. Notation: $x=\\log(F/K)$, $\\tau=T-t$, $c=C/F$, $w=\\sigma_{imp}^2\\tau$,\n'
            '$v_s,v_f$ = slow and fast instantaneous variance. Controlled benchmark uses $r=q=0$, so $F=S$.')
    p.section('1  Black-Scholes / Black-76')
    p.eq('$c(x,\\tau,\\sigma)=\\Phi(d_1)-e^{-x}\\Phi(d_2)$,     '
         '$d_1=\\dfrac{x}{\\sigma\\sqrt{\\tau}}+\\dfrac{\\sigma\\sqrt{\\tau}}{2}$,     $d_2=d_1-\\sigma\\sqrt{\\tau}$', dy=.042)
    p.text('Currency price $C=F\\,c$.   Vega (the calibration weight):  '
           '$\\nu=\\partial c/\\partial\\sigma=\\varphi(d_1)\\sqrt{\\tau}$.', fs=9)
    p.text('Implied volatility inverts  $C/K=e^{x}\\Phi(d)-\\Phi(d-\\sqrt{w})$  for $w$ by 75 bisection steps on '
           '$w\\in[10^{-12},40]$, then $\\sigma_{imp}=\\sqrt{w/\\tau}$.', fs=9)
    p.text('Inversion is defined only where   '
           '$\\max(e^{x}-1,0)+10^{-12} < C/K < e^{x}-10^{-12}$   — outside it the IV is NaN.', fs=9)
    p.text('Two baselines are used:  BS_FLAT, one $\\sigma$ for the surface;  BS_TERM / BS_EXPIRY, one $\\sigma_j$ per\n'
           'calibration expiry with total variance interpolated linearly in $\\tau$ and held flat beyond the knots.', fs=9)

    p.section('2  Single Heston   (5 parameters)')
    p.eq('$dS_t=\\sqrt{v_t}\\,S_t\\,dW^S_t$,       $dv_t=\\kappa(\\theta-v_t)\\,dt+\\sigma\\sqrt{v_t}\\,dW^v_t$,'
         '       $d\\langle W^S,W^v\\rangle_t=\\rho\\,dt$', fs=10, dy=.040)

    p.section('3  Double Heston   (10 parameters, separable four-shock)')
    p.eq('$dS_t=\\sqrt{v_{s,t}+v_{f,t}}\\;S_t\\,dW^S_t$', dy=.034)
    p.eq('$dv_{i,t}=\\kappa_i(\\theta_i-v_{i,t})\\,dt+\\sigma_i\\sqrt{v_{i,t}}\\,dW^{i}_t$,     '
         '$d\\langle W^S,W^i\\rangle_t=\\rho_i\\,dt$,     $i\\in\\{s,f\\}$', dy=.036)
    p.text('Each variance factor carries its own correlation with the price; the two factors are independent of each other.', fs=8.6)
    p.gap(.004)
    p.text('Per-factor characteristic exponent (Little-Heston-Trap stable form):', fs=9)
    p.eq('$b=\\kappa-\\rho\\sigma iu$,     $d=\\sqrt{b^2+\\sigma^2(u^2+iu)}$  with $\\Re\\,d\\geq 0$,     '
         '$g=\\dfrac{b-d}{b+d}$', fs=10, dy=.042)
    p.eq('$\\psi(u,\\tau)=\\dfrac{\\kappa\\theta}{\\sigma^2}\\left[(b-d)\\tau-2\\log\\dfrac{1-ge^{-d\\tau}}{1-g}\\right]'
         '+\\dfrac{b-d}{\\sigma^2}\\cdot\\dfrac{1-e^{-d\\tau}}{1-ge^{-d\\tau}}\\,v_0$', fs=10.5, dy=.050)
    p.text('The two factors multiply — this is the whole point of the separable construction:', fs=9)
    p.eq('$\\varphi(u,\\tau)=\\exp[\\,\\psi_s(u,\\tau)+\\psi_f(u,\\tau)\\,]$', dy=.036)
    p.text('Pricing integral (forward-normalised, undiscounted), evaluated by Gauss-Laguerre at 128 and again at 96 nodes:', fs=9)
    p.eq('$c(x,\\tau)=\\dfrac{1}{2}+\\dfrac{1}{\\pi}\\int_0^\\infty\\Re\\left[\\dfrac{e^{iux}\\varphi(u-i,\\tau)}{iu}\\right]du'
         '\\;-\\;e^{-x}\\left(\\dfrac{1}{2}+\\dfrac{1}{\\pi}\\int_0^\\infty\\Re\\left[\\dfrac{e^{iux}\\varphi(u,\\tau)}{iu}\\right]du\\right)$',
         fs=10, dy=.06)
    p.close(1)

    # ---------------------------------------------------------------- page 2
    p = Page(pdf, 'Conditions')
    p.section('4  Parameter admissibility')
    p.table([['condition', 'formula', 'where enforced'],
             ['positivity', '$\\kappa_i,\\theta_i,\\sigma_i,v_{0,i}>0$', 'always'],
             ['correlation bounds', '$-1<\\rho_i<1$', 'always'],
             ['slow-first ordering', '$\\kappa_s<\\kappa_f$', 'always (removes the label swap)'],
             ['Feller, strict variant', '$2\\kappa_i\\theta_i-\\sigma_i^2>0$', 'feller_strict only'],
             ['joint correlation disk', '$\\rho_s^2+\\rho_f^2<1$', 'reference contract, NOT the market tests']],
            widths=[.20, .28, .38], dy=.0165)
    p.text('Market tests use feller_free: the characteristic function stays valid without Feller, and validation dates\n'
           'chose it for both Heston models (median held-out IV RMSE 3.81 vs 6.88 for Double Heston).', fs=8.6)
    p.text('Under the strict variant the vol-of-vol is parameterised to satisfy Feller by construction:\n'
           '$\\sigma_i=\\eta_i\\sqrt{2\\kappa_i\\theta_i}$  with  $\\eta_i\\in[0.01,\\,0.999]$.', fs=8.6)

    p.section('5  Numerical acceptance (else recompute by adaptive quadrature and log the fallback)')
    p.eq('$|c_{128}-c_{96}|\\leq 10^{-7}$,       $c\\geq\\max(1-e^{-x},0)-10^{-9}$,       $c\\leq 1+10^{-9}$', fs=10, dy=.040)

    p.section('6  No-arbitrage conditions, checked on every plotted curve')
    p.eq('$\\dfrac{\\partial C}{\\partial S}\\in[0,1]$,     $\\dfrac{\\partial^2C}{\\partial S^2}\\geq 0$,     '
         '$\\dfrac{\\partial^2C}{\\partial K^2}\\geq 0$,     $\\dfrac{\\partial w}{\\partial\\tau}\\geq 0$,     '
         '$\\max(S-K,0)\\leq C\\leq S$', fs=10, dy=.046)
    p.text('and $C\\to\\max(S-K,0)$ as $\\tau\\to0$.   Result across all 30 controlled curves: zero violations.', fs=8.8)

    p.section('7  Statistical condition for a claimed win (frozen endpoint)')
    p.text('Double Heston beats a competitor only if BOTH hold on held-out quotes:', fs=8.8)
    p.eq('$p_{\\mathrm{Wilcoxon,\\;one\\!-\\!sided}}<0.05$       and       '
         '$\\mathrm{lower\\;bound\\;of\\;95\\%\\;cluster\\;bootstrap\\;of\\;}\\overline{\\Delta}>0$', fs=9.5, dy=.038)
    p.text('with  $\\Delta=\\mathrm{RMSE}_{\\mathrm{competitor}}-\\mathrm{RMSE}_{DH}$  averaged inside each cluster\n'
           '(shock episode, or calendar week), 5,000 replicates, seed 20260920.', fs=8.6)

    p.section('8  The PINN output construction')
    p.text('The network never predicts a price. It predicts a bounded correction to an analytic baseline volatility.', fs=8.8)
    p.eq('$\\bar v(\\tau)=\\sum_{i\\in\\{s,f\\}}\\left[\\theta_i+(v_i-\\theta_i)\\,'
         '\\dfrac{1-e^{-\\kappa_i\\tau}}{\\kappa_i\\tau}\\right]$', fs=11, dy=.050)
    p.eq('$\\mathrm{corr}=1.8\\tanh(\\mathrm{head})$,       $\\sigma_{imp}=\\sqrt{\\bar v}\\,e^{\\mathrm{corr}}$,       '
         '$w=\\sigma_{imp}^2\\tau$,       $C/K=e^{x}\\Phi(d)-\\Phi(d-\\sqrt{w})$', fs=9.5, dy=.040)
    p.text('Two properties follow by construction, not by training: the payoff at $\\tau=0$ is exact, and the price\n'
           'cannot leave its no-arbitrage bounds.', fs=8.8)
    p.close(2)

    # ---------------------------------------------------------------- page 3
    p = Page(pdf, 'PINN loss and training')
    p.section('9  The 24 engineered features (fixed formulas, nothing learned)')
    p.text('With  $z=x/\\sqrt{\\tau\\bar v+x^2/64}$  and  $\\eta_i=\\sigma_i/\\sqrt{2\\kappa_i\\theta_i}$:', fs=8.8)
    p.text('4 general:     $x/0.36$,     $z/8$,     $\\tanh(z/1.5)$,     scaled $\\log\\tau$', fs=9)
    p.text('9 per factor (18):   $\\log v_i$,  $\\log\\kappa_i$,  $\\log\\theta_i$,  $\\rho_i$,  $\\tanh(\\kappa_i\\tau)$,  '
           '$\\nu_i=\\tanh(\\sigma_i\\sqrt{\\tau}/\\sqrt{\\bar v})$,  $\\rho_i\\nu_i$,\n'
           '                              $\\tanh(\\frac{1}{2}\\log(v_i/\\theta_i))$,  Feller ratio  $2\\log(1+\\eta_i)/\\log 4-1$', fs=9)
    p.text('2 cross-factor:     $2v_s/(v_s+v_f)-1$,     $\\tanh(z/1.5)\\cdot(\\rho_s\\nu_s+\\rho_f\\nu_f)$', fs=9)

    p.section('10  Loss terms')
    p.eq('$L_{price}=\\langle(\\hat c_\\theta-c^{teacher})^2\\rangle$,          '
         '$L_{IV}=\\langle(\\hat\\sigma_\\theta-\\sigma^{teacher})^2\\rangle_{\\mathrm{valid\\,IV}}$', fs=10, dy=.038)
    p.eq('$L_{PDE}=\\langle\\mathcal{R}^2\\rangle$,          $L_{conv}=\\langle\\max(-\\mathcal{C},0)^2\\rangle$', fs=10, dy=.038)
    p.text('$\\mathcal{R}$ is the two-factor Heston PDE residual in log-total-variance form $\\ell=\\log w$, normalised by\n'
           '$\\mathrm{scale}=(v_s+v_f)+\\bar v$, with $a=\\frac{1}{2}x^2-\\frac{1}{8}w^2-\\frac{1}{2}w$:', fs=8.8)
    p.eq('$\\mathcal{R}=\\dfrac{1}{\\mathrm{scale}}\\left[w\\ell_\\tau-\\dfrac{v_s+v_f}{2}\\mathcal{D}_x'
         '-\\sum_i\\rho_i\\sigma_i v_i\\mathcal{D}_{xv_i}-\\dfrac{1}{2}\\sum_i\\sigma_i^2v_i\\mathcal{D}_{v_iv_i}'
         '-\\sum_i\\kappa_i(\\theta_i-v_i)w\\ell_{v_i}\\right]$', fs=10, dy=.050)
    p.text('$\\mathcal{D}_x=2+2(\\frac{w}{2}-x)\\ell_x+a\\ell_x^2+w(\\ell_{xx}+\\ell_x^2)-w\\ell_x$', fs=9.2, dy=.020)
    p.text('$\\mathcal{D}_{xv_i}=(\\frac{w}{2}-x)\\ell_{v_i}+a\\ell_x\\ell_{v_i}+w(\\ell_{xv_i}+\\ell_x\\ell_{v_i})$,     '
           '$\\mathcal{D}_{v_iv_i}=a\\ell_{v_i}^2+w(\\ell_{v_iv_i}+\\ell_{v_i}^2)$', fs=9.2, dy=.020)
    p.text('$\\mathcal{C}=(1-\\frac{1}{2}x\\ell_x)^2-\\frac{1}{4}w\\ell_x^2-\\frac{w^2\\ell_x^2}{16}'
           '+\\frac{1}{2}w(\\ell_{xx}+\\ell_x^2)$      (convexity diagnostic)', fs=9.2, dy=.026)
    p.text('All derivatives by automatic differentiation with create_graph=True.', fs=8.4, color=MUT)

    p.section('11  Total loss — each term divided by its own acceptance tolerance squared')
    p.eq('$L=\\dfrac{L_{price}}{(2\\times10^{-5})^2}+w_{IV}\\dfrac{L_{IV}}{(0.002)^2}'
         '+w_{PDE}\\cdot0.1\\cdot\\dfrac{L_{PDE}}{(0.01)^2}+w_{conv}\\cdot0.1\\cdot L_{conv}$', fs=11, dy=.052)
    p.text('Locked v4:  $w_{IV}=w_{PDE}=w_{conv}=1$, fixed.        Improved v5: updated every 100 steps by', fs=8.8)
    p.eq('$w_i\\leftarrow 0.9\\,w_i+0.1\\cdot\\min\\left(\\dfrac{\\|\\nabla_\\theta L_{price}\\|}{\\|\\nabla_\\theta L_i\\|},\\,10^3\\right)$', fs=10.5, dy=.048)
    p.text('Residual-adaptive collocation (v5): every 2,000 steps half of the 18,000 active points are redrawn from a\n'
           '120,000-point pool with  $p_j\\propto|\\mathcal{R}_j|/\\overline{|\\mathcal{R}|}+1$;  the other half stays uniform. Budget unchanged.', fs=8.8)
    p.text('Acceptance gates (frozen before training, all four must pass):   price RMSE $\\leq2\\times10^{-5}$,  '
           '$P_{95}\\leq5\\times10^{-5}$,\nmax $\\leq2\\times10^{-4}$,  IV RMSE $\\leq0.002$ (0.2 vol points).', fs=8.8)
    p.close(3)

    # ---------------------------------------------------------------- page 4
    p = Page(pdf, 'Hard-coded parameters')
    p.section('12  The ten Double Heston parameters   (published set, slow factor first)')
    p.table([['#', 'symbol', 'value', 'meaning'],
             ['1', '$\\kappa_s$', '0.9491', 'slow mean reversion — half-life 267 days'],
             ['2', '$\\theta_s$', '0.0257', 'slow long-run variance (16.0% vol)'],
             ['3', '$\\sigma_s$', '0.0517', 'slow vol-of-vol'],
             ['4', '$\\rho_s$', '+0.7009', 'slow price/variance correlation'],
             ['5', '$v_{s,0}$', '0.0003', 'slow initial variance (1.7% vol)'],
             ['6', '$\\kappa_f$', '10.7526', 'fast mean reversion — half-life 24 days'],
             ['7', '$\\theta_f$', '0.0330', 'fast long-run variance (18.2% vol)'],
             ['8', '$\\sigma_f$', '0.3613', 'fast vol-of-vol'],
             ['9', '$\\rho_f$', '-0.8916', 'fast price/variance correlation'],
             ['10', '$v_{f,0}$', '0.0252', 'fast initial variance (15.9% vol)']],
            widths=[.035, .075, .09, .50], dy=.0163)
    p.note('Source: Christoffersen, Heston & Jacobs (2009).')
    p.text('Structure:  $\\kappa_f/\\kappa_s=11.33$  — two clearly separated timescales; and  $\\rho_s>0>\\rho_f$  — opposite-signed\n'
           'correlations, the mechanism behind short-dated smirk flexibility and the reason the separable four-shock\n'
           'convention is required.', fs=8.8, color=GRN)
    p.eq('$v_{s,0}+v_{f,0}=0.0255\\;\\Rightarrow\\;\\sigma_{BS,fixed}=15.97\\%$,          '
         '$\\theta_s+\\theta_f=0.0587\\;\\Rightarrow\\;24.2\\%$ long-run', fs=9.5, dy=.038)
    p.text('Feller:  slow $2\\kappa_s\\theta_s-\\sigma_s^2=0.0461>0$,  fast $0.5791>0$  — both pass.\n'
           'Joint disk:  $\\rho_s^2+\\rho_f^2=1.286>1$  — the published set fails the stricter repository convention, which is\n'
           'why the market tests use the individual-$\\rho$ contract.', fs=8.8)

    p.section('13  Published Single Heston  (5 parameters)')
    p.eq('$(\\kappa,\\theta,\\sigma,\\rho,v_0)=(8.9814,\\;0.0409,\\;0.2970,\\;-0.9621,\\;0.0244)$,     Feller $=0.6465>0$', fs=9.5, dy=.038)

    p.section('14  The level scale $s$ — the only number ever fitted to a fixed-parameter model')
    p.eq('$\\theta_i\\mapsto s\\theta_i$,     $\\sigma_i\\mapsto\\sqrt{s}\\,\\sigma_i$,     $v_{i,0}\\mapsto s\\,v_{i,0}$,     '
         '$\\sigma_{BS}\\mapsto\\sqrt{s}\\,\\sigma_{BS}$,     $s\\in[0.7,\\,3.2]$', fs=9.5, dy=.040)
    p.text('Rescales the volatility LEVEL only; every timescale, correlation and shape is untouched. The range is exactly\n'
           'the range the PINN was trained on.', fs=8.8)

    p.section('15  Controlled scenario FIXED_TOTAL_TWIST')
    p.table([['state', '$v_{f,0}$', '$v_{s,0}$', 'total', 'instantaneous vol'],
             ['FAST-HEAVY', '0.035', '0.005', '0.04', '20%'],
             ['SLOW-HEAVY', '0.005', '0.035', '0.04', '20%']],
            widths=[.14, .09, .09, .08, .16], dy=.0165)
    p.text('Identical instantaneous volatility, materially different prices at every maturity, because variance in the fast\n'
           'factor decays 11.3x faster. No single-factor model can produce both curves from the same variance.', fs=8.8, color=GRN)
    p.close(4)

    # ---------------------------------------------------------------- page 5
    p = Page(pdf, 'Data ranges')
    p.section('16  PINN training domain — outside this box the network is untrained')
    p.table([['variable', 'range', 'sampling'],
             ['$x=\\log(F/K)$', '$[-0.36,\\,+0.36]$   i.e.  $S/K\\in[0.698,\\,1.433]$', 'uniform'],
             ['$\\tau$', '7 days to 2 years  (0.0192 - 2.0 y)', 'uniform in $\\log\\tau$'],
             ['$v_s$  slow variance', '$[1.5\\times10^{-4},\\,0.18]$   i.e.  1.2% - 42% vol', 'uniform in $\\log v_s$'],
             ['$v_f$  fast variance', '$[1.0\\times10^{-3},\\,0.22]$   i.e.  3.2% - 47% vol', 'uniform in $\\log v_f$'],
             ['level scale $s$', '$[0.7,\\,3.2]$', 'uniform'],
             ['sampler', '5-dimensional Latin hypercube, one seed per split', '']],
            widths=[.17, .40, .22], dy=.0165)
    p.text('Datasets:  teacher train 100,000 (seed 93101);  collocation 18,000 (93103);  v5 development 8,192 (93201);\n'
           'v5 UNTOUCHED final 16,384 (93202);  PDE final 4,096 (93203);  short-ATM diagnostic 6,144 (93204);\n'
           'RAD pool 120,000 (93205).', fs=8.6)

    p.section('17  Calibration bounds (market tests) and objective')
    p.table([['parameter', 'bound', 'parameter', 'bound'],
             ['$\\kappa$ (single)', '$[0.05,\\;50]$', '$\\sigma_i$', '$[0.01,\\;10]$'],
             ['$\\kappa_s$', '$[0.05,\\;20]$', '$\\rho_i$', '$[-0.99,\\;0.99]$'],
             ['$\\kappa_f-\\kappa_s$ (gap)', '$[0.05,\\;100]$', '$\\eta_i$ (strict Feller)', '$[0.01,\\;0.999]$'],
             ['$\\theta_i$', '$[0.001,\\;4]$', '$\\sigma_{BS}$', '$[0.05,\\;5]$'],
             ['$v_{i,0}$', '$[0.001,\\;4]$', '', '']],
            widths=[.19, .19, .21, .20], dy=.0165)
    p.eq('$\\min\\;\\left\\langle\\left(\\dfrac{c_{model}-c_{market}}{\\max(\\nu,\\,10^{-4})}\\right)^2\\right\\rangle$', fs=11, dy=.048)
    p.text('Dividing by vega makes this approximately an implied-volatility residual. Optimiser: differential evolution\n'
           '(popsize x6, 25 iterations, seed 20260919) then 12 bounded least-squares starts, all starts retained.', fs=8.6)

    p.section('18  Market data filters')
    p.table([['', 'crypto (Deribit)', 'equity / index (CBOE)'],
             ['maturity', '3 - 400 days', '7 - 730 days'],
             ['moneyness', '$|x|\\leq1.0$', '$|x|\\leq0.36$  (the PINN domain)'],
             ['implied vol', '0.10 - 3.00', '0.02 - 3.00'],
             ['option type', 'out-of-the-money only', 'out-of-the-money only'],
             ['minimum price', '0.0005 coin / 2 ticks', 'bid>0, ask>bid, ask $\\leq$ 3 bid'],
             ['quotes per expiry', '$\\geq3$', '$\\geq6$'],
             ['expiries per date', '$\\geq5$ (BTC/ETH), $\\geq4$ thin', '$\\geq6$'],
             ['window', '06:00-08:00 UTC or full day', 'single snapshot'],
             ['forward', 'solved per trade from (price, IV)', 'parity regression, $R^2\\geq0.999$']],
            widths=[.17, .33, .33], dy=.0163)

    p.section('19  Range of quantities plotted')
    p.table([['quantity', 'range'],
             ['$S$ (controlled)', '$0.70K$ to $1.30K$ with $K=100$;  ATM zoom $0.85K$ - $1.15K$'],
             ['$\\tau$', '7 - 730 days, 300-500 grid points; panels at 30, 90, 180, 365, 730 d'],
             ['calendar time $t$', '0 to $T$ with $\\tau=T-t$; PINN curves stop at $\\tau=7$ d'],
             ['$c=C/F$', '$[0,1)$ by construction'],
             ['time value', '0 to about 4 at 90 days ($K=100$)'],
             ['implied volatility', 'about 15-26% controlled,  9-50% market']],
            widths=[.20, .60], dy=.0163)
    p.close(5)

    # ---------------------------------------------------------------- page 6
    p = Page(pdf, 'Architecture')
    p.section('20  Block architecture: locked v4 and improved v5')
    img = imread(str(HERE / 'figures' / 'pinn_architecture_comparison.png'))
    ax = p.fig.add_axes([.045, .565, .91, .36]); ax.imshow(img); ax.axis('off')
    p.y = .565
    p.text('Identical in both: 12 inputs, 24 engineered features, analytic variance baseline, bounded correction,\n'
           'Black pricing layer. Only the learned body and the training mechanics differ.', fs=8.6)

    p.section('21  Measured effect, untouched 16,384-point test set opened once')
    p.table([['metric', 'locked v4', 'improved v5', 'change'],
             ['price RMSE vs exact DH', '1.062e-5', '6.610e-6', '1.61x better'],
             ['IV RMSE (vol points)', '0.1175', '0.0738', '1.59x better'],
             ['worst-case price error', '1.523e-4', '5.938e-5', '2.56x better'],
             ['PDE residual', '0.0309', '0.0558', '1.8x WORSE'],
             ['parameters', '269,825', '282,632', '+4.7%'],
             ['inference latency', '14.2 us/quote', '24.8 us/quote', '1.7x slower']],
            widths=[.26, .17, .17, .20], dy=.0165)
    p.text('The improvement is a better numerical solver, not a better market model: it moved real SPX pricing error by\n'
           '0.003 vol points out of 5.8, because network error is about 1% of model error.', fs=8.8, color=GRN)

    p.section('22  Where each equation lives in the code')
    p.table([['equation', 'file'],
             ['Black-76 price, vega, implied vol', 'experiments/btc_multifactor_v1/engine.py'],
             ['IV inversion and validity window', 'src/mentor_dh_pinn/regular_pinn_data.py'],
             ['factor characteristic exponent', 'src/double_heston_reference.py::_factor_exponent'],
             ['Fourier pricing integral', 'experiments/btc_multifactor_v1/engine.py::Grid'],
             ['admissibility, Feller, disk', 'engine.py::admissible, src/constraints.py'],
             ['baseline, features, correction, PDE', 'src/mentor_dh_pinn/regular_pinn_torch.py'],
             ['loss, gradient balancing, RAD', 'experiments/dh_pinn_v5/train.py'],
             ['architecture variants', 'experiments/dh_pinn_v5/v5_models.py'],
             ['published parameters, domains', 'experiments/nifty_multifactor_v4/config.json'],
             ['calibration bounds and objective', 'experiments/btc_multifactor_v1/config.json']],
            widths=[.33, .50], dy=.0163)
    p.close(6)

print('written', OUT, OUT.stat().st_size // 1024, 'KB')
