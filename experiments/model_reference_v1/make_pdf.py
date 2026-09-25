"""Formula sheet, organised as: one section per model (equations + that model's own parameters),
then the PINN loss function, then the shared conditions and data ranges."""
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
W, H = 8.27, 11.69
L, R, TOP = .085, .915, .955
INK, ACC, MUT, GRN = '#1a1a1a', '#1f3f8a', '#666666', '#16704f'


class Page:
    def __init__(self, pdf, header=None):
        self.fig = plt.figure(figsize=(W, H)); self.pdf = pdf; self.y = TOP
        if header:
            self.fig.text(L, .978, header, fontsize=7.5, color=MUT)
            self.fig.text(R, .978, 'Double Heston project - formula sheet', fontsize=7.5, color=MUT, ha='right')
            self.fig.add_artist(plt.Line2D([L, R], [.9715, .9715], color='#cccccc', lw=.7))

    def title(self, t, sub=None):
        self.fig.text(L, self.y, t, fontsize=16, color=INK, weight='bold', va='top'); self.y -= .036
        if sub:
            self.fig.text(L, self.y, sub, fontsize=8.8, color=MUT, va='top')
            self.y -= .020 * (sub.count('\n') + 1) + .006

    def part(self, t):
        self.y -= .010
        self.fig.add_artist(plt.Rectangle((L - .012, self.y - .026), R - L + .024, .030,
                                          facecolor='#eef2f9', edgecolor='none',
                                          transform=self.fig.transFigure, zorder=0))
        self.fig.text(L, self.y - .004, t, fontsize=12.5, color=ACC, weight='bold', va='top'); self.y -= .040

    def sub(self, t, color=ACC):
        self.y -= .010
        self.fig.text(L, self.y, t, fontsize=10, color=color, weight='bold', va='top'); self.y -= .022

    def text(self, t, fs=8.6, color=INK, dy=None, indent=0.):
        n = t.count('\n') + 1
        self.fig.text(L + indent, self.y, t, fontsize=fs, color=color, va='top', linespacing=1.45)
        self.y -= (dy if dy is not None else .0155 * n + .004)

    def eq(self, t, fs=10.5, dy=.034, color=INK):
        self.fig.text(.5, self.y, t, fontsize=fs, color=color, va='top', ha='center'); self.y -= dy

    def table(self, rows, widths, fs=8.2, dy=.0148, header=True):
        for i, row in enumerate(rows):
            x = L
            for cell, w in zip(row, widths):
                self.fig.text(x, self.y, cell, fontsize=fs, va='top',
                              color=ACC if (i == 0 and header) else INK,
                              weight='bold' if (i == 0 and header) else 'normal')
                x += w
            self.y -= dy
            if i == 0 and header:
                self.fig.add_artist(plt.Line2D([L, R], [self.y + .009, self.y + .009], color='#dddddd', lw=.6))
                self.y -= .003
        self.y -= .007

    def note(self, t, fs=7.9): self.text(t, fs=fs, color=MUT)

    def close(self, n):
        self.fig.text(.5, .022, str(n), fontsize=7.5, color=MUT, ha='center')
        self.pdf.savefig(self.fig); plt.close(self.fig)


with PdfPages(OUT) as pdf:
    # ================================================================= page 1
    p = Page(pdf)
    p.title('Formula sheet: models, parameters, PINN loss',
            'Notation:  $x=\\log(F/K)$,  $\\tau=T-t$,  $c=C/F$,  $w=\\sigma_{imp}^2\\tau$,  $v_s,v_f$ = slow and fast variance.\n'
            'Every formula is taken from the code; the file for each is listed in section E.5.')

    p.part('PART A    BLACK-SCHOLES')
    p.sub('A.1  Equations')
    p.eq('$c(x,\\tau,\\sigma)=\\Phi(d_1)-e^{-x}\\Phi(d_2)$,      '
         '$d_1=\\dfrac{x}{\\sigma\\sqrt{\\tau}}+\\dfrac{\\sigma\\sqrt{\\tau}}{2}$,      $d_2=d_1-\\sigma\\sqrt{\\tau}$', dy=.044)
    p.text('Currency price  $C=F\\,c$;  with $r=q=0$ (controlled benchmark) $F=S$, so $C=S\\,c$.', fs=8.8)
    p.eq('vega:     $\\nu=\\dfrac{\\partial c}{\\partial\\sigma}=\\varphi(d_1)\\sqrt{\\tau}$,      '
         '$\\varphi(z)=\\dfrac{1}{\\sqrt{2\\pi}}e^{-z^2/2}$', fs=10, dy=.044)
    p.text('Implied volatility: invert  $C/K=e^{x}\\Phi(d)-\\Phi(d-\\sqrt{w})$,  $d=x/\\sqrt{w}+\\sqrt{w}/2$,  for $w$ by\n'
           '75 bisection steps on $w\\in[10^{-12},40]$, then $\\sigma_{imp}=\\sqrt{w/\\tau}$.', fs=8.8)
    p.text('Defined only where   $\\max(e^{x}-1,0)+10^{-12} < C/K < e^{x}-10^{-12}$   - outside it the IV is NaN, which\n'
           'is what produces the gaps in deep out-of-the-money short-dated smiles.', fs=8.8)
    p.text('Term-structure variant (BS_TERM / BS_EXPIRY): one $\\sigma_j$ per calibration expiry, total variance\n'
           'interpolated linearly in $\\tau$ between knots and held flat outside:', fs=8.8)
    p.eq('$w(\\tau)=\\mathrm{interp}(\\tau;\\;\\{\\tau_j,\\;\\tau_j\\sigma_j^2\\})$,       $\\sigma(\\tau)=\\sqrt{w(\\tau)/\\tau}$',
         fs=10, dy=.040)

    p.sub('A.2  Parameters')
    p.table([['parameter', 'what it captures', 'value or bound used'],
             ['$\\sigma$  (flat)', 'one constant volatility for the whole surface', 'fitted, bounds $[0.05,\\,5]$'],
             ['$\\sigma_j$  (per expiry)', 'one volatility per calibration expiry', 'fitted, same bounds; 22-33 knots'],
             ['$\\sigma_{BS,fixed}$', 'the hard-coded level in fixed-parameter tests', '$\\sqrt{0.0255}=15.97\\%$']],
            widths=[.165, .38, .30], dy=.017)
    p.note('The fixed value is not an independent choice: it is the initial total volatility of the published Double Heston\n'
           'set, so all three models start from the same instantaneous volatility.')

    p.part('PART B    SINGLE HESTON')
    p.sub('B.1  Equations')
    p.eq('$dS_t=\\sqrt{v_t}\\,S_t\\,dW^S_t$,        $dv_t=\\kappa(\\theta-v_t)\\,dt+\\sigma\\sqrt{v_t}\\,dW^v_t$,        '
         '$d\\langle W^S,W^v\\rangle_t=\\rho\\,dt$', fs=10, dy=.042)
    p.text('Characteristic exponent - the same function used once (Double Heston uses it twice):', fs=8.8)
    p.eq('$b=\\kappa-\\rho\\sigma iu$,      $d=\\sqrt{b^2+\\sigma^2(u^2+iu)}$  with $\\Re\\,d\\geq0$,      '
         '$g=\\dfrac{b-d}{b+d}$', fs=10, dy=.042)
    p.eq('$\\psi(u,\\tau)=\\dfrac{\\kappa\\theta}{\\sigma^2}\\left[(b-d)\\tau-2\\log\\dfrac{1-ge^{-d\\tau}}{1-g}\\right]'
         '+\\dfrac{b-d}{\\sigma^2}\\cdot\\dfrac{1-e^{-d\\tau}}{1-ge^{-d\\tau}}\\,v_0$', fs=10.5, dy=.052)
    p.eq('$\\varphi(u,\\tau)=\\exp[\\psi(u,\\tau)]$,        then the pricing integral of section C.3', fs=9.8, dy=.034)
    p.close(1)

    # ================================================================= page 2
    p = Page(pdf, 'Single Heston parameters   ·   Double Heston equations')
    p.sub('B.2  Parameters   (5)')
    p.table([['symbol', 'name', 'what it captures', 'published', 'bound'],
             ['$\\kappa$', 'mean reversion', 'speed volatility returns to normal', '8.9814', '$[0.05,\\,50]$'],
             ['$\\theta$', 'long-run variance', 'level volatility drifts back to', '0.0409', '$[0.001,\\,4]$'],
             ['$\\sigma$', 'vol-of-vol', 'uncertainty of volatility itself', '0.2970', '$[0.01,\\,10]$'],
             ['$\\rho$', 'correlation', 'leverage effect / skew tilt', '$-0.9621$', '$[-0.99,\\,0.99]$'],
             ['$v_0$', 'initial variance', 'volatility today', '0.0244', '$[0.001,\\,4]$']],
            widths=[.075, .145, .275, .12, .15], dy=.0165)
    p.text('Feller on the published set:  $2\\kappa\\theta-\\sigma^2=0.6465>0$ - satisfied.\n'
           'One $\\kappa$ means ONE timescale: short-dated and long-dated behaviour cannot be matched independently.', fs=8.8)

    p.part('PART C    DOUBLE HESTON   (separable four-shock)')
    p.sub('C.1  Dynamics')
    p.eq('$dS_t=\\sqrt{v_{s,t}+v_{f,t}}\\;S_t\\,dW^S_t$', fs=11, dy=.036)
    p.eq('$dv_{i,t}=\\kappa_i(\\theta_i-v_{i,t})\\,dt+\\sigma_i\\sqrt{v_{i,t}}\\,dW^{i}_t$,       '
         '$d\\langle W^S,W^i\\rangle_t=\\rho_i\\,dt$,       $i\\in\\{s,f\\}$', fs=10.5, dy=.040)
    p.text('Four separate shocks: each variance factor carries its own correlation with the price, and the two variance\n'
           'factors are independent of each other. This is what allows $\\rho_s$ and $\\rho_f$ to take opposite signs.', fs=8.8)

    p.sub('C.2  Characteristic function - the two factors multiply')
    p.eq('$\\psi_i(u,\\tau)$  as in B.1, evaluated with $(\\kappa_i,\\theta_i,\\sigma_i,\\rho_i,v_{i,0})$', fs=9.8, dy=.032)
    p.eq('$\\varphi(u,\\tau)=\\exp\\left[\\,\\psi_s(u,\\tau)+\\psi_f(u,\\tau)\\,\\right]$', fs=11.5, dy=.044)
    p.text('The product form is the point of the separable construction: two independent affine factors compose exactly,\n'
           'so a two-factor price costs the same single integral as a one-factor price.', fs=8.8)

    p.sub('C.3  Pricing integral   (forward-normalised, undiscounted; same for Single and Double)')
    p.eq('$c(x,\\tau)=\\dfrac{1}{2}+\\dfrac{1}{\\pi}\\int_0^\\infty\\Re\\left[\\dfrac{e^{iux}\\varphi(u-i,\\tau)}{iu}\\right]du'
         '\\;-\\;e^{-x}\\left(\\dfrac{1}{2}+\\dfrac{1}{\\pi}\\int_0^\\infty\\Re\\left[\\dfrac{e^{iux}\\varphi(u,\\tau)}{iu}\\right]du\\right)$',
         fs=10, dy=.056)
    p.text('Evaluated by Gauss-Laguerre quadrature at 128 nodes and again at 96; the two must agree to $10^{-7}$ or the\n'
           'point is recomputed by independent adaptive quadrature and the fallback is logged.', fs=8.8)
    p.close(2)

    # ================================================================= page 3
    p = Page(pdf, 'Double Heston parameters')
    p.sub('C.4  Parameters   (10 = 5 per factor)')
    p.table([['#', 'symbol', 'what it captures', 'published', 'bound'],
             ['1', '$\\kappa_s$', 'slow mean reversion - half-life 267 days', '0.9491', '$[0.05,\\,20]$'],
             ['2', '$\\theta_s$', 'slow long-run variance  (16.0% vol)', '0.0257', '$[0.001,\\,4]$'],
             ['3', '$\\sigma_s$', 'slow vol-of-vol', '0.0517', '$[0.01,\\,10]$'],
             ['4', '$\\rho_s$', 'slow price/variance correlation', '$+0.7009$', '$[-0.99,\\,0.99]$'],
             ['5', '$v_{s,0}$', 'slow initial variance  (1.7% vol)', '0.0003', '$[0.001,\\,4]$'],
             ['6', '$\\kappa_f$', 'fast mean reversion - half-life 24 days', '10.7526', 'gap $[0.05,\\,100]$'],
             ['7', '$\\theta_f$', 'fast long-run variance  (18.2% vol)', '0.0330', '$[0.001,\\,4]$'],
             ['8', '$\\sigma_f$', 'fast vol-of-vol', '0.3613', '$[0.01,\\,10]$'],
             ['9', '$\\rho_f$', 'fast price/variance correlation', '$-0.8916$', '$[-0.99,\\,0.99]$'],
             ['10', '$v_{f,0}$', 'fast initial variance  (15.9% vol)', '0.0252', '$[0.001,\\,4]$']],
            widths=[.03, .07, .365, .135, .17], dy=.0163)
    p.note('Source: Christoffersen, Heston and Jacobs (2009).  $\\kappa_f$ is calibrated through the gap $\\kappa_f-\\kappa_s$, which\n'
           'enforces the slow-first ordering automatically.')
    p.text('Structure:  $\\kappa_f/\\kappa_s=11.33$ - two genuinely separated timescales;  and $\\rho_s>0>\\rho_f$ - opposite-signed\n'
           'correlations, the mechanism behind maturity-dependent skew and the reason the separable four-shock\n'
           'convention is required rather than a restricted single-correlation form.', fs=8.8, color=GRN)
    p.eq('$v_{s,0}+v_{f,0}=0.0255\\;\\Rightarrow\\;15.97\\%$ today,         '
         '$\\theta_s+\\theta_f=0.0587\\;\\Rightarrow\\;24.2\\%$ long-run', fs=9.6, dy=.038)
    p.text('Feller:  slow $2\\kappa_s\\theta_s-\\sigma_s^2=0.0461>0$,  fast $0.5791>0$ - both satisfied.\n'
           'Joint disk:  $\\rho_s^2+\\rho_f^2=1.286>1$ - the published set fails the stricter repository convention, which is\n'
           'why the market tests use the individual-$\\rho$ contract.', fs=8.8)

    p.sub('C.5  The level scale $s$ - the only number fitted in fixed-parameter experiments')
    p.eq('$\\theta_i\\mapsto s\\theta_i$,      $v_{i,0}\\mapsto s\\,v_{i,0}$,      $\\sigma_i\\mapsto\\sqrt{s}\\,\\sigma_i$,      '
         '$\\sigma_{BS}\\mapsto\\sqrt{s}\\,\\sigma_{BS}$,      $s\\in[0.7,\\,3.2]$', fs=9.8, dy=.040)
    p.text('Shifts the volatility LEVEL only; every timescale, correlation and shape is untouched. The range is exactly\n'
           'the range the PINN was trained on.', fs=8.8)

    p.sub('C.6  Controlled scenario FIXED_TOTAL_TWIST')
    p.table([['state', '$v_{f,0}$', '$v_{s,0}$', 'total', 'instantaneous vol', 'ATM IV at 90 d'],
             ['FAST-HEAVY', '0.035', '0.005', '0.04', '20%', '19.95%'],
             ['SLOW-HEAVY', '0.005', '0.035', '0.04', '20%', '23.77%']],
            widths=[.135, .085, .085, .075, .155, .15], dy=.0165)
    p.text('Identical variance today, materially different prices at every maturity, because variance in the fast factor\n'
           'decays 11.3x faster. No single-factor model can produce both from the same starting volatility.', fs=8.8, color=GRN)
    p.close(3)

    # ================================================================= page 4
    p = Page(pdf, 'PINN: construction and loss terms')
    p.part('PART D    THE PINN    (a surrogate for exact Double Heston)')
    p.sub('D.1  Output construction - analytic prior plus bounded correction')
    p.eq('$\\bar v(\\tau)=\\sum_{i\\in\\{s,f\\}}\\left[\\theta_i+(v_i-\\theta_i)\\,'
         '\\dfrac{1-e^{-\\kappa_i\\tau}}{\\kappa_i\\tau}\\right]$         (expected average variance)', fs=11, dy=.052)
    p.eq('$\\mathrm{corr}=1.8\\tanh(\\mathrm{head})$,     $\\sigma_{imp}=\\sqrt{\\bar v}\\,e^{\\mathrm{corr}}$,     '
         '$w=\\sigma_{imp}^2\\tau$,     $C/K=e^{x}\\Phi(d)-\\Phi(d-\\sqrt{w})$', fs=9.6, dy=.042)
    p.text('Consequences by construction, not by training: the payoff at $\\tau=0$ is exact, and the price cannot leave its\n'
           'no-arbitrage bounds.', fs=8.8)

    p.sub('D.2  Inputs and features')
    p.text('12 raw inputs: $x$, $v_s$, $v_f$, $\\tau$ and the 8 structural parameters $\\kappa_i,\\theta_i,\\sigma_i,\\rho_i$. Expanded to 24 fixed\n'
           'features (nothing learned), with $z=x/\\sqrt{\\tau\\bar v+x^2/64}$ and $\\eta_i=\\sigma_i/\\sqrt{2\\kappa_i\\theta_i}$:', fs=8.8)
    p.text('   4 general:  $x/0.36$,   $z/8$,   $\\tanh(z/1.5)$,   scaled $\\log\\tau$', fs=8.8)
    p.text('   9 per factor (18):  $\\log v_i$,  $\\log\\kappa_i$,  $\\log\\theta_i$,  $\\rho_i$,  $\\tanh(\\kappa_i\\tau)$,  '
           '$\\nu_i=\\tanh(\\sigma_i\\sqrt{\\tau}/\\sqrt{\\bar v})$,  $\\rho_i\\nu_i$,\n'
           '        $\\tanh(\\frac{1}{2}\\log(v_i/\\theta_i))$,   $2\\log(1+\\eta_i)/\\log4-1$', fs=8.8)
    p.text('   2 cross-factor:  $2v_s/(v_s+v_f)-1$,   $\\tanh(z/1.5)\\cdot(\\rho_s\\nu_s+\\rho_f\\nu_f)$', fs=8.8)

    p.sub('D.3  The four loss terms')
    p.table([['term', 'definition', 'what it enforces', 'points'],
             ['$L_{price}$', '$\\langle(\\hat c_\\theta-c^{teacher})^2\\rangle$', 'reproduce the exact Fourier price', '100,000 labelled'],
             ['$L_{IV}$', '$\\langle(\\hat\\sigma_\\theta-\\sigma^{teacher})^2\\rangle$', 'accuracy where prices are small', 'same, valid IV only'],
             ['$L_{PDE}$', '$\\langle\\mathcal{R}^2\\rangle$', 'the two-factor Heston equation', '18,000 unlabelled'],
             ['$L_{conv}$', '$\\langle\\max(-\\mathcal{C},0)^2\\rangle$', 'no butterfly arbitrage', 'same collocation set']],
            widths=[.075, .225, .29, .19], dy=.0175)
    p.text('$\\mathcal{R}$ is the PDE residual in log-total-variance form $\\ell=\\log w$, normalised by $\\mathrm{scale}=(v_s+v_f)+\\bar v$, with\n'
           '$a=\\frac{1}{2}x^2-\\frac{1}{8}w^2-\\frac{1}{2}w$; all derivatives by automatic differentiation:', fs=8.8)
    p.eq('$\\mathcal{R}=\\dfrac{1}{\\mathrm{scale}}\\left[w\\ell_\\tau-\\dfrac{v_s+v_f}{2}\\mathcal{D}_x'
         '-\\sum_i\\rho_i\\sigma_i v_i\\mathcal{D}_{xv_i}-\\dfrac{1}{2}\\sum_i\\sigma_i^2v_i\\mathcal{D}_{v_iv_i}'
         '-\\sum_i\\kappa_i(\\theta_i-v_i)w\\ell_{v_i}\\right]$', fs=9.8, dy=.048)
    p.text('$\\mathcal{D}_x=2+2(\\frac{w}{2}-x)\\ell_x+a\\ell_x^2+w(\\ell_{xx}+\\ell_x^2)-w\\ell_x$', fs=9, dy=.019)
    p.text('$\\mathcal{D}_{xv_i}=(\\frac{w}{2}-x)\\ell_{v_i}+a\\ell_x\\ell_{v_i}+w(\\ell_{xv_i}+\\ell_x\\ell_{v_i})$,       '
           '$\\mathcal{D}_{v_iv_i}=a\\ell_{v_i}^2+w(\\ell_{v_iv_i}+\\ell_{v_i}^2)$', fs=9, dy=.019)
    p.text('$\\mathcal{C}=(1-\\frac{1}{2}x\\ell_x)^2-\\frac{1}{4}w\\ell_x^2-\\frac{w^2\\ell_x^2}{16}+\\frac{1}{2}w(\\ell_{xx}+\\ell_x^2)$       '
           '(convexity diagnostic)', fs=9, dy=.024)
    p.close(4)

    # ================================================================= page 5
    p = Page(pdf, 'PINN: total loss and minimisation')
    p.sub('D.4  The total loss function')
    p.text('Each supervised term is divided by the square of its own acceptance tolerance, so no term dominates merely\n'
           'because of its units:', fs=8.8)
    p.eq('$L=\\dfrac{L_{price}}{(2\\times10^{-5})^2}\\;+\\;w_{IV}\\dfrac{L_{IV}}{(0.002)^2}'
         '\\;+\\;w_{PDE}\\cdot0.1\\cdot\\dfrac{L_{PDE}}{(0.01)^2}\\;+\\;w_{conv}\\cdot0.1\\cdot L_{conv}$', fs=11, dy=.056)
    p.table([['weights', 'locked v4', 'improved v5'],
             ['$w_{IV},\\;w_{PDE},\\;w_{conv}$', 'fixed at 1', 'updated every 100 steps by gradient balancing']],
            widths=[.21, .18, .42], dy=.017)
    p.eq('$w_i\\leftarrow0.9\\,w_i+0.1\\cdot\\min\\left(\\dfrac{\\|\\nabla_\\theta L_{price}\\|}{\\|\\nabla_\\theta L_i\\|},\\;10^3\\right)$',
         fs=10.5, dy=.050)
    p.note('Learning-rate annealing of Wang, Teng and Perdikaris: if one term produces gradients orders of magnitude larger\n'
           'than the price term it silently dominates; this rebalances toward the ratio of gradient norms, smoothed 90/10.')

    p.sub('D.5  How the loss is minimised')
    p.table([['stage', 'setting'],
             ['Adam', '40,000 steps;  lr $10^{-3}\\to10^{-5}$ cosine;  batch 512 labelled + 128 collocation'],
             ['L-BFGS', '300 steps, strong-Wolfe line search, history 30, on 4,096 labels + 512 collocation'],
             ['seeds', '17 and 43 trained independently; the two predicted prices are averaged'],
             ['collocation (v5)', 'half of the 18,000 points redrawn every 2,000 steps, $p_j\\propto|\\mathcal{R}_j|/\\overline{|\\mathcal{R}|}+1$'],
             ['checkpointing', 'every 2,000 steps, so an interrupted run costs minutes not hours']],
            widths=[.16, .66], dy=.017)
    p.text('Acceptance gates, frozen before training - all four must pass on the development split:', fs=8.8)
    p.eq('price RMSE $\\leq2\\times10^{-5}$,      $P_{95}\\leq5\\times10^{-5}$,      max $\\leq2\\times10^{-4}$,      '
         'IV RMSE $\\leq0.002$  (0.2 vol pts)', fs=9.4, dy=.042)

    p.sub('D.6  Architecture and measured effect')
    p.table([['', 'locked v4', 'improved v5'],
             ['body', 'plain stack: 5 tanh layers, width 256', '5 gated blocks, width 256'],
             ['gating', 'none', '$h\\leftarrow(1-Z_\\ell)\\odot U+Z_\\ell\\odot V$'],
             ['activation', 'fixed $\\tanh$', '$\\tanh(a_\\ell z)$, $a_\\ell$ learned per block'],
             ['parameters', '269,825', '282,632    (+4.7%)'],
             ['price RMSE vs exact DH', '1.062e-5', '6.610e-6    (1.61x better)'],
             ['IV RMSE (vol points)', '0.1175', '0.0738    (1.59x better)'],
             ['worst-case price error', '1.523e-4', '5.938e-5    (2.56x better)'],
             ['PDE residual', '0.0309', '0.0558    (1.8x worse)'],
             ['inference latency', '14.2 $\\mu$s/quote', '24.8 $\\mu$s/quote    (1.7x slower)']],
            widths=[.215, .295, .31], dy=.017)
    p.text('Measured once on an untouched 16,384-point test set. The improvement is to the SOLVER: it changed real SPX\n'
           'market error by 0.003 vol points out of 5.8, because network error is about 1% of model error.', fs=8.8, color=GRN)
    p.close(5)

    # ================================================================= page 6
    p = Page(pdf, 'Conditions and data ranges')
    p.part('PART E    CONDITIONS, DATA RANGES, CODE MAP')
    p.sub('E.1  Parameter admissibility')
    p.table([['condition', 'formula', 'where enforced'],
             ['positivity', '$\\kappa_i,\\theta_i,\\sigma_i,v_{0,i}>0$', 'always'],
             ['correlation bounds', '$-1<\\rho_i<1$', 'always'],
             ['slow-first ordering', '$\\kappa_s<\\kappa_f$', 'always (removes the label swap)'],
             ['Feller (strict variant)', '$2\\kappa_i\\theta_i-\\sigma_i^2>0$ via $\\sigma_i=\\eta_i\\sqrt{2\\kappa_i\\theta_i}$', 'feller_strict only'],
             ['joint correlation disk', '$\\rho_s^2+\\rho_f^2<1$', 'reference contract, NOT market tests']],
            widths=[.175, .35, .32], dy=.0165)

    p.sub('E.2  Numerical and no-arbitrage conditions')
    p.eq('$|c_{128}-c_{96}|\\leq10^{-7}$,       $c\\geq\\max(1-e^{-x},0)-10^{-9}$,       $c\\leq1+10^{-9}$', fs=9.6, dy=.036)
    p.eq('$\\partial C/\\partial S\\in[0,1]$,    $\\partial^2C/\\partial S^2\\geq0$,    $\\partial^2C/\\partial K^2\\geq0$,    '
         '$\\partial w/\\partial\\tau\\geq0$,    $\\max(S-K,0)\\leq C\\leq S$', fs=9.6, dy=.038)
    p.text('Checked on all 30 controlled curves: zero violations.', fs=8.8)

    p.sub('E.3  Calibration objective and PINN training domain')
    p.eq('$\\min\\;\\left\\langle\\left(\\dfrac{c_{model}-c_{market}}{\\max(\\nu,\\,10^{-4})}\\right)^2\\right\\rangle$       '
         'differential evolution (popsize x6, 25 iters), then 12 local starts', fs=9.4, dy=.048)
    p.table([['variable', 'PINN training range', 'sampling'],
             ['$x=\\log(F/K)$', '$[-0.36,\\,+0.36]$   i.e. $S/K\\in[0.698,\\,1.433]$', 'uniform'],
             ['$\\tau$', '7 days to 2 years', 'uniform in $\\log\\tau$'],
             ['$v_s$', '$[1.5\\times10^{-4},\\,0.18]$   i.e. 1.2% - 42% vol', 'uniform in $\\log v_s$'],
             ['$v_f$', '$[1.0\\times10^{-3},\\,0.22]$   i.e. 3.2% - 47% vol', 'uniform in $\\log v_f$'],
             ['level scale $s$', '$[0.7,\\,3.2]$', 'uniform'],
             ['datasets', 'train 100k, collocation 18k, dev 8,192, untouched test 16,384', '5-D Latin hypercube']],
            widths=[.135, .43, .25], dy=.0165)

    p.sub('E.4  Market data filters')
    p.table([['', 'crypto (Deribit)', 'equity / index (CBOE)'],
             ['maturity', '3 - 400 days', '7 - 730 days'],
             ['moneyness', '$|x|\\leq1.0$', '$|x|\\leq0.36$  (the PINN domain)'],
             ['implied vol', '0.10 - 3.00', '0.02 - 3.00'],
             ['quotes / expiries', '$\\geq3$ per expiry, $\\geq5$ expiries (BTC, ETH)', '$\\geq6$ per expiry, $\\geq6$ expiries'],
             ['forward', 'solved per trade from (price, IV)', 'parity regression, $R^2\\geq0.999$']],
            widths=[.165, .35, .31], dy=.0165)

    p.sub('E.5  Where each equation lives')
    p.table([['equation', 'file'],
             ['Black-76 price, vega, implied vol', 'experiments/btc_multifactor_v1/engine.py'],
             ['IV inversion and validity window', 'src/mentor_dh_pinn/regular_pinn_data.py'],
             ['characteristic exponent $\\psi$', 'src/double_heston_reference.py::_factor_exponent'],
             ['pricing integral', 'experiments/btc_multifactor_v1/engine.py::Grid'],
             ['admissibility, Feller, disk', 'engine.py::admissible,  src/constraints.py'],
             ['baseline, features, correction, PDE residual', 'src/mentor_dh_pinn/regular_pinn_torch.py'],
             ['loss, gradient balancing, RAD sampling', 'experiments/dh_pinn_v5/train.py'],
             ['published parameters, domains', 'experiments/nifty_multifactor_v4/config.json'],
             ['calibration bounds and objective', 'experiments/btc_multifactor_v1/config.json']],
            widths=[.36, .47], dy=.0155)
    p.close(6)

    # ================================================================= page 7
    p = Page(pdf, 'Architecture and summary')
    p.sub('D.7  Block architecture, locked v4 and improved v5')
    img = imread(str(HERE / 'figures' / 'pinn_architecture_comparison.png'))
    ax = p.fig.add_axes([.045, .52, .91, .40]); ax.imshow(img); ax.axis('off')
    p.y = .50
    p.text('Identical in both: 12 inputs, 24 engineered features, analytic variance baseline, bounded correction and the\n'
           'Black pricing layer. Only the learned body and the training mechanics differ.', fs=8.8)
    p.sub('Summary of the three model sections')
    p.table([['', 'free parameters', 'smile?', 'two horizons?', 'held-out IWM error'],
             ['Black-Scholes (per expiry)', '22', 'no', 'partly, by expiry', '5.39 vol pts'],
             ['Single Heston', '5', 'yes', 'no - one timescale', '1.89 vol pts'],
             ['Double Heston', '10', 'yes', 'yes - fast and slow', '0.58 vol pts'],
             ['DH-PINN (surrogate)', 'reproduces DH', 'inherits DH', 'inherits DH', '7.5e-4 vs exact DH']],
            widths=[.225, .155, .125, .195, .17], dy=.0168)
    p.text('Companion document: HOW_IT_WORKS.pdf - the same material explained in plain language.', fs=8.6, color=MUT)
    p.close(7)

print('written', OUT, OUT.stat().st_size // 1024, 'KB')
