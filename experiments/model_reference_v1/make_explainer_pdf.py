"""Build the plain-language explainer PDF: what the models are, what each parameter captures,
what the loss function is, how it is minimised, and how the whole pipeline fits together."""
import textwrap
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

HERE = Path(__file__).resolve().parent
OUT = HERE / 'HOW_IT_WORKS.pdf'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'mathtext.fontset': 'dejavusans',
                     'figure.facecolor': 'white', 'savefig.facecolor': 'white'})
W, H = 8.27, 11.69
L, R, TOP = .085, .915, .955
INK, ACC, MUT, GRN, RED = '#1a1a1a', '#1f3f8a', '#666666', '#16704f', '#a03030'
WRAP = 103


class Page:
    def __init__(self, pdf, header=None):
        self.fig = plt.figure(figsize=(W, H)); self.pdf = pdf; self.y = TOP
        if header:
            self.fig.text(L, .978, header, fontsize=7.5, color=MUT)
            self.fig.text(R, .978, 'How the Double Heston project works', fontsize=7.5, color=MUT, ha='right')
            self.fig.add_artist(plt.Line2D([L, R], [.9715, .9715], color='#cccccc', lw=.7))

    def title(self, t, sub=None):
        self.fig.text(L, self.y, t, fontsize=17, color=INK, weight='bold', va='top'); self.y -= .038
        if sub:
            for line in textwrap.wrap(sub, WRAP - 6):
                self.fig.text(L, self.y, line, fontsize=9.3, color=MUT, va='top'); self.y -= .0165
            self.y -= .008

    def h(self, t, color=ACC, fs=11.5):
        self.y -= .014
        self.fig.text(L, self.y, t, fontsize=fs, color=color, weight='bold', va='top'); self.y -= .025

    def p(self, t, fs=9.2, color=INK, wrap=WRAP, indent=0.):
        for line in textwrap.wrap(t, wrap):
            self.fig.text(L + indent, self.y, line, fontsize=fs, color=color, va='top'); self.y -= .0163
        self.y -= .006

    def bullet(self, t, fs=9.2, color=INK):
        lines = textwrap.wrap(t, WRAP - 4)
        for i, line in enumerate(lines):
            self.fig.text(L + (.012 if i else 0), self.y, ('•  ' if i == 0 else '') + line, fontsize=fs, color=color, va='top')
            self.y -= .0163
        self.y -= .003

    def eq(self, t, fs=10.5, dy=.036, color=INK):
        self.fig.text(.5, self.y, t, fontsize=fs, color=color, va='top', ha='center'); self.y -= dy

    def table(self, rows, widths, fs=8.4, dy=.0155):
        for i, row in enumerate(rows):
            x = L
            for cell, w in zip(row, widths):
                self.fig.text(x, self.y, cell, fontsize=fs, va='top', color=ACC if i == 0 else INK,
                              weight='bold' if i == 0 else 'normal')
                x += w
            self.y -= dy
            if i == 0:
                self.fig.add_artist(plt.Line2D([L, R], [self.y + .009, self.y + .009], color='#dddddd', lw=.6))
                self.y -= .003
        self.y -= .008

    def box(self, t, color=GRN, fs=9.2):
        lines = textwrap.wrap(t, WRAP - 6)
        h = .0163 * len(lines) + .016
        self.fig.add_artist(plt.Rectangle((L - .012, self.y - h + .012), R - L + .024, h,
                                          facecolor='#f4f9f6' if color == GRN else '#fdf4f4',
                                          edgecolor=color, lw=.8, transform=self.fig.transFigure, zorder=0))
        yy = self.y - .002
        for line in lines:
            self.fig.text(L, yy, line, fontsize=fs, color=color, va='top'); yy -= .0163
        self.y -= h + .008

    def close(self, n):
        self.fig.text(.5, .022, str(n), fontsize=7.5, color=MUT, ha='center')
        self.pdf.savefig(self.fig); plt.close(self.fig)


with PdfPages(OUT) as pdf:
    # ============================================================ 1
    p = Page(pdf)
    p.title('How the whole thing works',
            'A plain-language walkthrough: what we are pricing, what each model assumes, what every parameter '
            'captures, what the PINN loss function is, how it is minimised, and how the results are tested.')

    p.h('1  The problem')
    p.p('A European call option gives the right to buy an asset at a fixed strike K on a fixed date. Its price C '
        'splits into two parts:')
    p.eq('$C \\;=\\; \\max(S-K,\\,0) \\;+\\; \\mathrm{time\\;value}$', fs=12, dy=.030)
    p.eq('(intrinsic value: arithmetic)          (what the model decides)', fs=8.6, dy=.034, color=MUT)
    p.p('Every model agrees on the intrinsic value — it is arithmetic. All the disagreement lives in the time '
        'value, which is the market paying for the chance that the price moves before expiry. How much that '
        'chance is worth depends entirely on what you assume about volatility. That assumption is the model.')

    p.h('2  The three models, in one sentence each')
    p.table([['model', 'assumption about volatility', 'free numbers'],
             ['Black-Scholes', 'constant, one number forever', '1 (or 1 per expiry)'],
             ['Single Heston', 'random, pulled back to one long-run level', '5'],
             ['Double Heston', 'random, two independent components: one fast, one slow', '10']],
            widths=[.17, .50, .18])
    p.p('That is the entire hierarchy. Each step adds a way for volatility to behave that the previous model '
        'could not express.')

    p.h('3  What goes wrong with each, and why it matters')
    p.p('Black-Scholes assumes one volatility for all strikes. But the market charges more for low-strike puts '
        '(crash protection) than for high-strike calls. Plot the market\'s implied volatility against strike and '
        'you get a "smile" or "skew". Black-Scholes must draw a horizontal line through it. On the real IWM '
        'surface that costs it 5.39 volatility points of error.')
    p.p('Single Heston lets volatility move randomly and pulls it back toward a long-run level at a single speed '
        'kappa. One speed means one timescale. If today\'s volatility is elevated, Single Heston must decay it '
        'back at one rate, so it cannot simultaneously match what short-dated options say (volatility falls fast) '
        'and what long-dated options say (volatility stays elevated). On our controlled benchmark that mismatch '
        'peaks at 0.85 volatility points around 7 days, falls to zero near 110 days, then flips sign.')
    p.p('Double Heston splits volatility into two pieces with their own speeds — one that decays in about 24 days '
        'and one that takes 267 days. Short-dated options are priced mostly by the fast piece, long-dated ones '
        'mostly by the slow piece, and the model can match both ends at once.')

    p.h('4  The key idea in one experiment')
    p.p('Take two situations with exactly the same volatility today, 20%. In the first, the volatility sits in the '
        'fast component; in the second, in the slow one. Total variance is 0.04 in both cases.')
    p.table([['maturity', 'fast-heavy ATM vol', 'slow-heavy ATM vol', 'call price, fast vs slow'],
             ['30 days', '19.8%', '22.1%', '2.29  vs  2.55'],
             ['90 days', '20.0%', '23.8%', '3.96  vs  4.71'],
             ['1 year', '21.1%', '24.8%', '8.42  vs  9.87'],
             ['2 years', '22.1%', '24.7%', '12.42  vs  13.87']],
            widths=[.13, .20, .20, .27])
    p.box('Same volatility today, prices differ by up to 12% at every maturity — because variance sitting in the '
          'fast component evaporates 11.3 times faster. A one-factor model cannot produce both columns from the '
          'same starting volatility. That is the entire case for Double Heston.')
    p.close(1)

    # ============================================================ 2
    p = Page(pdf, 'What the parameters capture')
    p.h('5  What each of the ten parameters actually controls')
    p.p('Each variance factor has five parameters. Double Heston has two factors, so ten. Here is what each one '
        'does to the price, in words.')
    p.table([['symbol', 'name', 'what it captures', 'raise it and ...'],
             ['$\\kappa$', 'mean reversion', 'how fast volatility forgets today and returns to normal',
              "today's vol matters less"],
             ['$\\theta$', 'long-run variance', 'the level volatility drifts back to',
              'long-dated options cost more'],
             ['$\\sigma$', 'vol-of-vol', 'how uncertain the future volatility itself is',
              'the smile gets more curved'],
             ['$\\rho$', 'correlation', 'whether vol rises when price falls (the leverage effect)',
              'the skew tilts (neg = crash-fear)'],
             ['$v_0$', 'initial variance', 'how volatile the asset is right now',
              'everything costs more, short end most']],
            widths=[.075, .13, .39, .235], dy=.027)
    p.p('So kappa and theta shape the TERM STRUCTURE (how price varies with maturity), while sigma and rho shape '
        'the SMILE (how price varies with strike), and v0 sets the overall level today.')

    p.h('6  The published values we use, and what they say')
    p.table([['', '$\\kappa$', '$\\theta$', '$\\sigma$', '$\\rho$', '$v_0$', 'reading'],
             ['slow factor', '0.9491', '0.0257', '0.0517', '+0.7009', '0.0003', 'calm, persistent, 267-day memory'],
             ['fast factor', '10.7526', '0.0330', '0.3613', '-0.8916', '0.0252', 'jumpy, 24-day memory, crash-fear']],
            widths=[.095, .082, .082, .082, .092, .082, .30], dy=.019)
    p.p('Two things stand out. The speeds differ by 11.3 times, which is what makes the two factors genuinely '
        'different rather than redundant. And the correlations have OPPOSITE SIGNS: the fast factor is strongly '
        'negative (-0.89, the usual crash-fear tilt), while the slow one is positive (+0.70). A single factor '
        'must pick one number, so it cannot produce a skew that changes character with maturity. This is why the '
        'model must be written with four separate shocks rather than the restricted form.')

    p.h('7  Why "hard-coded" and what the one fitted number does')
    p.p('In several experiments we deliberately do NOT fit these ten numbers — we use the published values and '
        'ask what the STRUCTURE alone is worth. The only number ever fitted in that setting is a single level '
        'scale s, which multiplies the variances:')
    p.eq('$\\theta_i \\mapsto s\\,\\theta_i$,      $v_{i,0} \\mapsto s\\,v_{i,0}$,      '
         '$\\sigma_i \\mapsto \\sqrt{s}\\,\\sigma_i$,      $s \\in [0.7,\\,3.2]$', fs=10, dy=.040)
    p.p('This shifts the overall volatility level up or down without touching any timescale, correlation or '
        'shape. It is there so that a parameter set fitted to 1990s equities can be tried on a market trading at '
        'a different volatility level, without secretly refitting the structure.')
    p.box('Result worth knowing: with the structure frozen this way, Double Heston beats Single Heston by only '
          'about 0.1-0.3 volatility points on real surfaces. When all ten parameters are fitted to the surface, '
          'the same model beats Single Heston by 3.2 times on held-out quotes. The two-factor structure is real, '
          'but it only pays once the parameters are allowed to describe the market in front of you.')
    p.close(2)

    # ============================================================ 3
    p = Page(pdf, 'The PINN')
    p.h('8  Why a neural network at all')
    p.p('Pricing one Double Heston option exactly means numerically integrating a complex-valued function. That '
        'is accurate but slow, and calibration needs hundreds of thousands of such prices. The PINN is trained '
        'to reproduce the exact answer in a single fast forward pass. It is a SURROGATE for the model, not a '
        'new model: the target it is trained against is exact Double Heston.')

    p.h('9  The design that makes it work: prior plus correction')
    p.p('A naive network would try to predict the price directly and would have to learn the payoff kink, the '
        'no-arbitrage bounds and the shape of the entire surface from scratch. Instead the network is given a '
        'head start. First we compute an analytic baseline volatility — the average variance the model expects '
        'over the life of the option:')
    p.eq('$\\bar v(\\tau) \\;=\\; \\sum_{i\\in\\{s,f\\}} \\left[\\theta_i + (v_i-\\theta_i)\\,'
         '\\frac{1-e^{-\\kappa_i\\tau}}{\\kappa_i\\tau}\\right]$', fs=11.5, dy=.050)
    p.p('This formula already knows the two-timescale behaviour: the bracket interpolates between today\'s '
        'variance v_i at short maturity and the long-run level theta_i at long maturity, at each factor\'s own '
        'speed. It is not exact, because it ignores the smile and the correlation effects. The network only has '
        'to supply the correction:')
    p.eq('$\\sigma_{imp} \\;=\\; \\sqrt{\\bar v}\\;\\times\\;e^{\\,1.8\\tanh(\\mathrm{network\\;output})}$', fs=11, dy=.044)
    p.p('The tanh bounds the correction, so the network can scale the baseline volatility by at most a factor of '
        'about 6 either way and can never produce nonsense. The resulting volatility is then fed through the '
        'Black formula to get the price. Two properties come for free: at expiry the price is exactly the '
        'payoff, and the price can never violate its basic bounds. The network cannot break them even if it '
        'wanted to.')

    p.h('10  What the network sees')
    p.p('Twelve raw inputs: log-moneyness x = log(F/K), the two variances, time to maturity, and the eight '
        'structural parameters. These are expanded into 24 fixed financial features — things like scaled '
        'log-maturity, a standardised moneyness, tanh(kappa*tau) which says how far through the factor\'s memory '
        'we are, a vol-of-vol measure, and skew terms. None of these are learned; they are hand-built so the '
        'network starts from quantities that already mean something financially.')

    p.h('11  The body of the network')
    p.table([['', 'original (v4)', 'improved (v5)'],
             ['layers', '5 layers, width 256, plain tanh', '5 gated blocks, width 256'],
             ['connections', 'none beyond layer to layer', 'gated skip paths through every block'],
             ['activation', 'fixed tanh', 'tanh($a_\\ell z$) with $a_\\ell$ learned per block'],
             ['parameters', '269,825', '282,632  (+4.7%)']],
            widths=[.14, .35, .40], dy=.018)
    p.p('The gated block keeps two encodings U and V of the input features available at every depth and lets the '
        'network mix between them, so information and gradients have a short path from input to output. That is '
        'the single biggest architectural gain we measured.')
    p.close(3)

    # ============================================================ 4
    p = Page(pdf, 'The loss function')
    p.h('12  The loss function, term by term')
    p.p('Training minimises one number. It is built from four pieces, each measuring a different kind of being '
        'wrong.')
    p.table([['term', 'what it measures', 'why it is there'],
             ['$L_{price}$', 'squared price error against the exact Fourier price',
              'the main job: reproduce the model'],
             ['$L_{IV}$', 'squared implied-volatility error against the exact one',
              'price error alone under-weights cheap options'],
             ['$L_{PDE}$', 'how badly the output violates the pricing equation',
              'physics: forces a valid solution everywhere'],
             ['$L_{conv}$', 'penalty when the surface bends the wrong way',
              'blocks butterfly arbitrage']],
            widths=[.09, .42, .38], dy=.024)
    p.p('The first two are supervised: they compare against 100,000 exactly computed teacher prices. The third is '
        'unsupervised: at 18,000 extra "collocation" points, with no labels at all, automatic differentiation '
        'computes the derivatives of the network output and checks whether they satisfy the two-factor Heston '
        'partial differential equation. That is the "physics-informed" part — the network is told the law its '
        'answer must obey, not just examples of correct answers.')
    p.p('The four pieces have wildly different natural sizes: prices are around 1e-5, implied vols around 1e-3. '
        'Adding them raw would let one term dominate. So each is divided by the square of the accuracy we '
        'actually require of it — its acceptance tolerance:')
    p.eq('$L \\;=\\; \\frac{L_{price}}{(2\\times10^{-5})^2} \\;+\\; \\frac{L_{IV}}{(0.002)^2} '
         '\\;+\\; 0.1\\cdot\\frac{L_{PDE}}{(0.01)^2} \\;+\\; 0.1\\cdot L_{conv}$', fs=11, dy=.048)
    p.box('This normalisation was itself a bug fix. An earlier version divided the price term by 1e-3 instead of '
          '2e-5, which made it only 1.5-2% of the total loss. The network was underfitting the thing we cared '
          'about most, and no amount of extra training fixed it. Rescaling each term by its own acceptance '
          'tolerance cut the error by an order of magnitude.')

    p.h('13  How it is minimised')
    p.bullet('Adam for 40,000 steps. Each step draws a random batch of 512 teacher points and 128 physics points, '
             'computes the loss, backpropagates, and nudges all 282,632 weights downhill. The learning rate '
             'follows a cosine schedule from 1e-3 down to 1e-5, so it explores early and settles late.')
    p.bullet('Then L-BFGS for 300 steps on a fixed large batch. This is a second-order method: it builds an '
             'approximation of the curvature and takes much better-aimed steps. It is too memory-hungry for the '
             'whole run but excellent for polishing the last digits.')
    p.bullet('Two random seeds are trained independently and their prices averaged, which removes some of the '
             'luck of initialisation.')
    p.p('The improved version adds two mechanisms during training:')
    p.bullet('Gradient balancing. Every 100 steps we measure how large a gradient each loss term produces. If the '
             'physics term is pushing 100 times harder than the price term, it silently dominates. The weights '
             'are rebalanced toward the ratio of gradient norms, smoothed 90/10 so they cannot oscillate.')
    p.bullet('Residual-adaptive sampling. Every 2,000 steps half of the physics points are redrawn, with '
             'probability proportional to how badly the equation is currently violated there. The network gets '
             'more practice exactly where it is failing. The budget never grows; the points just move.')
    p.close(4)

    # ============================================================ 5
    p = Page(pdf, 'Training, calibration, testing')
    p.h('14  How we know the training worked')
    p.p('Four acceptance gates were fixed before any training: price RMSE below 2e-5, 95th percentile below 5e-5, '
        'worst case below 2e-4, and implied-vol RMSE below 0.2 volatility points. A candidate architecture is '
        'only accepted if all four pass on a development set. The final network is then measured once on a test '
        'set of 16,384 points that had never been opened.')
    p.table([['measured on the untouched test set', 'original v4', 'improved v5'],
             ['typical price error', '1.06e-5', '6.61e-6'],
             ['worst price error', '1.52e-4', '5.94e-5'],
             ['implied-vol error (vol points)', '0.1175', '0.0738'],
             ['physics residual', '0.0309', '0.0558  (worse)']],
            widths=[.36, .20, .22], dy=.018)
    p.p('So the improved network is about 1.6 times more accurate as a solver, and its worst case is 2.6 times '
        'better — at the cost of 4.7% more parameters, 1.7 times slower inference, and a worse physics residual, '
        'because gradient balancing shifted weight toward the supervised terms.')

    p.h('15  How the models are fitted to real market prices')
    p.p('Separately from the PINN, the models are calibrated to observed option prices. We minimise a '
        'vega-weighted price error:')
    p.eq('$\\min \\;\\; \\left\\langle \\left(\\frac{C_{model}-C_{market}}{\\mathrm{vega}}\\right)^2\\right\\rangle$', fs=11, dy=.046)
    p.p('Dividing by vega — the sensitivity of price to volatility — converts a price error into approximately a '
        'volatility error. Without it, the fit would obsess over expensive at-the-money options and ignore the '
        'wings, where the interesting structure lives. The optimiser is differential evolution (a global search '
        'that does not need a good starting guess) followed by 12 local least-squares refinements from different '
        'starting points, because these objectives have many local minima. Every start is kept and reported.')

    p.h('16  How the comparison is kept honest')
    p.bullet('Held-out quotes. Models are fitted on half the surface, in a checkerboard pattern, and scored on '
             'the other half, which they never saw. A model that merely has more parameters gains nothing.')
    p.bullet('Rules frozen first. Filters, date rules, endpoints and statistical tests are written down and '
             'hashed before any data is downloaded. Data selection uses market structure only, never model error.')
    p.bullet('A win must clear two bars: a one-sided Wilcoxon test with p < 0.05, AND a bootstrap confidence '
             'interval whose lower bound is above zero, clustered so correlated dates cannot fake significance.')
    p.bullet('Failures are kept. Flawed runs, superseded results and unfavourable outcomes stay in the repository '
             'and in the reports.')
    p.close(5)

    # ============================================================ 6
    p = Page(pdf, 'What it all showed')
    p.h('17  The results, briefly')
    p.p('On a real surface chosen in advance for its two-timescale structure (IWM), with every model given its '
        'strongest fair calibration and scored on 707 quotes it never saw:')
    p.table([['model', 'free parameters', 'held-out error (vol points)'],
             ['Black-Scholes, one vol per expiry', '22', '5.39'],
             ['Single Heston', '5', '1.89'],
             ['Double Heston', '10', '0.58']],
            widths=[.33, .18, .28], dy=.019)
    p.p('Double Heston is 3.2 times better than Single Heston, and Black-Scholes is worst despite having the most '
        'free parameters — extra volatility levels cannot make a smile.')
    p.p('On crypto options with per-date calibration, Double Heston beat Single Heston significantly on BTC and '
        'ETH (about 7 expiries per day) but not on SOL, XRP or HYPE (about 4 expiries per day). The advantage is '
        'real but needs enough liquid expiries to identify the second factor.')
    p.p('With the published parameters frozen, the advantage shrinks to 0.0-0.3 volatility points on every real '
        'surface tested. And improving the PINN itself moved real market error by 0.003 volatility points out of '
        '5.8, because network error is about 1% of model error.')

    p.h('18  The mental model to take away')
    p.table([['layer', 'what it fixes', 'what it costs'],
             ['Black-Scholes', 'nothing; it is the reference shape', '1 number, no smile, no term structure'],
             ['+ one variance factor', 'smile and a single decay speed', '5 numbers'],
             ['+ second factor', 'short and long horizons independently', '10 numbers, needs rich data'],
             ['+ PINN surrogate', 'speed: one forward pass instead of an integral', 'approximation error 7e-6'],
             ['+ residual head', 'whatever the model still cannot say', 'no longer a pure model']],
            widths=[.18, .40, .34], dy=.019)
    p.box('The single most useful thing this project established: the gap between a fixed-parameter model and the '
          'market is roughly a hundred times larger than the error of the neural surrogate. So improving the '
          'network is a speed and accuracy win for the solver, but it is not a route to better market prices. '
          'Better market prices come from letting the parameters describe the market, or from admitting a '
          'correction term and labelling it honestly.', color=GRN)
    p.p('Companion documents: MODEL_EQUATIONS_REFERENCE.pdf for every formula and range; the per-experiment '
        'REPORT.md files for the full results and their caveats.', fs=8.6, color=MUT)
    p.close(6)

print('written', OUT, OUT.stat().st_size // 1024, 'KB')
