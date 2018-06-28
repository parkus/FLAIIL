import celerite
import numpy as np
import plots
import ranges
from numbers import gappy_runs, exposure_gaps, get_repeats
from matplotlib import pyplot as plt
from scipy.optimize import minimize


def identify_flares(t0, t1, f, e, options={}, plot_steps=False):
    # parse options
    maxiter = options.get('maxiter', len(f)/5)
    sigma_clip_factor = options.get('sigma_clip_factor', 2.)
    sigma_suspect = options.get('sigma_suspect', 3.)
    sigma_flare = options.get('sigma_flare', 5.)
    preclean = options.get('preclean', None)
    extend_factor = options.get('extend_factor', 2.)
    prepend_time = options.get('prepend_time', 30.)
    tau_min = options.get('tau_min', 100.)
    tau_logprior = options.get('tau_logprior', None)#lambda tau: np.log(tau))

    #region setup to handle gaps and other unchanging arrays
    t = (t0 + t1)/2.0
    i_gaps = exposure_gaps(t0, t1)
    t_gap_beg, t_gap_end = t1[i_gaps - 1], t0[i_gaps]
    t_ends = np.sort(np.concatenate([t0[:1], t_gap_beg, t_gap_end, t1[-1:]]))
    t_groups = np.array([t_ends[:-1], t_ends[1:]]).T[::2]
    t_gap_mid = (t_ends[1:-1:2] + t_ends[2::2]) / 2.
    dt_gaps = t_gap_end - t_gap_beg
    dt = t1 - t0
    t_edges = np.insert(t0, i_gaps, t1[i_gaps-1])
    t_edges = np.append(t_edges, t1[-1])
    #endregion

    qmodel = QuiescenceModel(t, f, e, tau_min=tau_min, tau_logprior=tau_logprior)
    if preclean is None:
        # first pass using sigma-clipped points
        clean = np.abs(f - np.nanmedian(f)) < sigma_clip_factor * np.nanstd(f)
        qmodel.fit(clean)
        lo, lo_var = qmodel.curve(t)
    else:
        clean = preclean
        qmodel.fit(clean)
        lo, lo_var = qmodel.curve(t)
    count = 0
    oldclean = []
    breaknext = False
    while True:
        # get quiescence-subtracted values
        hi = f - lo
        hi_var = e**2 + lo_var

        #region compute fluences of runs above and below quiescence
        # compute cumulative integral
        F = dt * hi
        V = dt**2 * hi_var
        F = np.insert(F, i_gaps, 0)
        V = np.insert(V, i_gaps, dt_gaps**2*qmodel.curve(t_gap_mid)[1])
        Iedges = np.insert(np.cumsum(F), 0, 0)
        Vedges = np.insert(np.cumsum(V), 0, 0)
        Iinterp = lambda t: np.interp(t, t_edges, Iedges)
        Vinterp = lambda t: np.interp(t, t_edges, Vedges)

        # interpolate fluences of runs above and below quiescence from cumulative integral
        begs, ends = gappy_runs(t, hi, t_ends, t_gap_mid)
        fluences = Iinterp(ends) - Iinterp(begs)
        fluence_vars = Vinterp(ends) - Vinterp(begs)

        assert not (np.any(np.isnan(fluences)) or np.any(np.isnan(fluence_vars)))

        # flag runs that are anomalous
        anom = abs(fluences/np.sqrt(fluence_vars)) > sigma_suspect

        # update the flare point mask, extending the mask from the end of flares to be a bit to be conservative
        anom_ranges = np.array([begs[anom], ends[anom]]).T
        oldclean.append(clean)
        spans = anom_ranges[:,1] - anom_ranges[:,0]
        anom_ranges[:,1] = anom_ranges[:,1] + extend_factor*spans
        anom_ranges[:,0] = anom_ranges[:,0] - prepend_time

        # plot step if desired
        if plot_steps:
            tnan = np.insert(t, i_gaps, np.nan)
            fnan = np.insert(f, i_gaps, np.nan)
            plots.standard_flareplot(tnan, fnan, anom_ranges, anom_ranges, qmodel)
            plt.title('Iteration {}'.format(count))
            raw_input('Enter to close figure and continue.')

        # check for convergence (or oscillating convergence)
        if breaknext:
            break
        repeats = get_repeats(oldclean)
        if repeats:
            clean = np.mean(repeats, 0) >= 0.5
            breaknext = True
        else:
            in_flare = [np.searchsorted(rng, t) == 1 for rng in anom_ranges]
            clean = ~np.any(in_flare, 0) if len(anom_ranges) > 0 else np.ones(len(t), bool)
        if count > maxiter:
            raise StopIteration('Iteration limit of {} exceeded.'.format(maxiter))

        if plot_steps:
            plt.close()

        # fit new quiescence points
        qmodel.fit(clean)
        lo, lo_var = qmodel.curve(t)
        count += 1

    # divide anomalous ranges into actual flares and just suspect ranges
    anom_ranges = reduce(ranges.rangeset_union, anom_ranges[1:], anom_ranges[:1])
    sigmas = []
    for rng in anom_ranges:
        inrng = ranges.inranges(begs, rng, inclusive=[True, False])
        sigma = fluences[inrng]/np.sqrt(fluence_vars[inrng])
        imx = np.argmax(np.abs(sigma))
        sigma = sigma[imx]
        sigmas.append(sigma)
    sigmas = np.array(sigmas)
    # print sigmas
    flare = sigmas > sigma_flare
    flare_ranges = anom_ranges[flare]
    suspect_ranges = anom_ranges[~flare]

    return flare_ranges, suspect_ranges, qmodel


class QuiescenceModel(celerite.GP):
    def __init__(self, t, f, e, tau_min=100., tau_logprior=None, params=None, mask=None):
        terms = celerite.terms
        kernel = terms.RealTerm(log_a=np.log(np.var(f)), log_c=-10.)
        super(QuiescenceModel, self).__init__(kernel, mean=np.median(f), fit_mean=True)
        if params is not None:
            self.set_parameter_vector(params)
        if mask is not None:
            self.mask = mask
        else:
            self.mask = np.ones(len(t), bool)
        self.tau_min = tau_min
        self.tau_logprior = tau_logprior
        self.t, self.f, self.e = t, f, e
        self.var_med = np.median(e)**2
        self.n = len(self.t)
        self.fit_params = None
        self.quick_compute()

    def tau_loglike(self, tau):
        if tau < self.tau_min:
            return -np.inf
        if self.tau_logprior is None:
            return 0.0
        else:
            return self.tau_logprior(tau)

    def smoothness_penalty(self, params):
        self.set_parameter_vector(params)
        power10 = self.kernel.get_psd(2*np.pi/10)
        if power10 == 0:
            return 0.0
        if power10 == np.inf or np.isnan(power10):
            return -np.inf
        else:
            return -5*power10

    def quick_compute(self):
        super(QuiescenceModel, self).compute(self.t[self.mask], self.e[self.mask])

    def _get_set_mask(self, mask=None):
        if mask is None:
            return self.mask
        else:
            self.mask = mask
            return mask

    def log_likelihood(self, params):
        self.set_parameter_vector(params)
        data_loglike = super(QuiescenceModel, self).log_likelihood(self.f[self.mask])
        tau = np.exp(-params[1])
        return data_loglike + self.tau_loglike(tau)

    def cost(self, params):
        return -(self.log_likelihood(params) + self.smoothness_penalty(params))
        # return -self.log_likelihood(params)

    def log_likelihood_white_noise(self, log_sig2_and_mu):
        self.set_parameter_vector([log_sig2_and_mu[0], np.inf, log_sig2_and_mu[1]])
        return super(QuiescenceModel, self).log_likelihood(self.f[self.mask])

    def log_likelihood_no_noise(self, mu):
        self.set_parameter_vector([-np.inf, np.inf, mu])
        return super(QuiescenceModel, self).log_likelihood(self.f[self.mask])

    def fit(self, mask=None, method='Nelder-Mead'):
        self._get_set_mask(mask)
        self.quick_compute()
        guess = self.get_parameter_vector()
        soln = minimize(lambda params: self.cost(params), guess, method=method, options=dict(maxiter=200))
        if not soln.success:
            raise ValueError('Gaussian process fit to quiescence did not converge. Perhaps try a different minimize '
                             'method or different initial parameters.')
        self.fit_params = soln.x
        self.set_to_best_fit()

    def _get_params_boiler(self, params):
        if params is None:
            params = self.get_parameter_vector()
        return np.reshape(params, [-1,3])

    def tau(self, params=None):
        params = self._get_params_boiler(params)
        return np.squeeze(np.exp(-params[:,1]))

    def sigma(self, params=None):
        params = self._get_params_boiler(params)
        return np.squeeze(np.exp(params[:,0]/2))

    def mu(self, params=None):
        params = self._get_params_boiler(params)
        return np.squeeze(params[:,2])

    def sigma_rel(self, params=None):
        return np.squeeze(self.sigma(params)/self.mu(params))

    def sigma_relative_at_tbin(self, tbin, params=None):
        params = self._get_params_boiler(params)
        loga, logc, mu = params.T
        sig, c = np.exp(loga/2), np.exp(logc)
        with np.errstate(invalid='ignore', over='ignore'):
            x = c*tbin
            sig_dt = np.sqrt(2*(sig/x)**2 * (x + np.exp(-x) - 1))
            uncorrelated = (x == np.inf)
            if np.any(uncorrelated):
                dt = np.median(np.diff(self.t))
                sig_dt[uncorrelated] = sig[uncorrelated]/np.sqrt(tbin/dt)
        return np.squeeze(sig_dt/mu)

    def set_to_best_fit(self):
        self.set_parameter_vector(self.fit_params)
        self.quick_compute()

    def curve(self, t):
        return self.predict(self.f[self.mask], t, return_var=True)