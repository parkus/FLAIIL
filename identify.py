import celerite
import numpy as np
import plots
import ranges
from numbers import gappy_runs, exposure_gaps
from matplotlib import pyplot as plt
from scipy.optimize import minimize


def identify_flares(t0, t1, f, e, options={}, plot_steps=False):
    # parse options
    maxiter = options.get('maxiter', len(f))
    sigma_clip_factor = options.get('sigma_clip_factor', 2.)
    sigma_suspect = options.get('sigma_flare', 3.)
    sigma_flare = options.get('sigma_flare', 5.)
    preclean = options.get('preclean', None)
    extend_factor = options.get('extend_factor', 1.5)

    #region setup to handle gaps and other unchanging arrays
    t = (t0 + t1)/2.0
    i_gaps = exposure_gaps(t0, t1)
    t_gap_beg, t_gap_end = t1[i_gaps - 1], t0[i_gaps]
    t_ends = np.sort(np.concatenate([t0[:1], t_gap_beg, t_gap_end, t1[-1:]]))
    t_groups = np.array([t_ends[:-1], t_ends[1:]]).T[::2]
    t_gap_mid = (t_ends[1:-1:2] + t_ends[2::2]) / 2.
    dt_gaps = t_gap_end - t_gap_beg
    dt = t1 - t0
    t_edges = np.unique(np.concatenate([t0, t1]))
    #endregion

    qmodel = QuiescenceModel(t, f, e)
    if preclean is None:
        # first pass using sigma-clipped points
        clean = np.abs(f - np.nanmedian(f)) < sigma_clip_factor*np.nanstd(f)
        qmodel.fit(clean)
        lo, lo_var = qmodel.curve(t)
    else:
        clean = preclean
        qmodel.fit(clean)
        lo, lo_var = qmodel.curve(t)
    count = 0
    oldclean = []
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

        # flag runs that are anomalous
        anom = abs(fluences/np.sqrt(fluence_vars)) > sigma_suspect

        # update the flare point mask, extending the mask from the end of flares to be a bit to be conservative
        anom_ranges = np.array([begs[anom], ends[anom]]).T
        oldclean.append(clean)
        if len(oldclean) > 2:
            oldclean.pop(0)
        spans = anom_ranges[:,1] - anom_ranges[:,0]
        anom_ranges[:,1] = anom_ranges[:,1] + extend_factor*spans
        in_flare = [np.searchsorted(rng, t) == 1 for rng in anom_ranges]
        clean = ~np.any(in_flare, 0) if len(anom_ranges) > 0 else np.ones(len(t), bool)

        # plot step if desired
        if plot_steps:
            tnan = np.insert(t, i_gaps, np.nan)
            fnan = np.insert(f, i_gaps, np.nan)
            plots.standard_flareplot(tnan, fnan, anom_ranges, anom_ranges, qmodel)
            plt.title('Iteration {}'.format(count))
            raw_input('Enter to close figure and continue.')

        # check for convergence (or oscillating convergence)
        if np.all(clean == oldclean[-1]):
            break
        if count > 10 and np.all(clean == oldclean[-2]):
            break
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
    anom_ranges = ranges.rangeset_intersect(anom_ranges, t_groups)
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
    def __init__(self, t, f, e, params=None, mask=None):
        terms = celerite.terms
        kernel = terms.RealTerm(log_a=np.log(np.var(f)), log_c=-10.)
        super(QuiescenceModel, self).__init__(kernel, mean=np.median(f), fit_mean=True)
        if params is not None:
            self.set_parameter_vector(params)
        if mask is not None:
            self.mask = mask
        else:
            self.mask = np.ones(len(t), bool)
        self.t, self.f, self.e = t, f, e
        self.n = len(self.t)
        self.fit_params = None
        self.quick_compute()

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
        return super(QuiescenceModel, self).log_likelihood(self.f[self.mask])

    def fit(self, mask=None, method='Nedler-Mead'):
        mask = self._get_set_mask(mask)
        self.quick_compute()
        guess = self.get_parameter_vector()
        soln = minimize(lambda params: -self.log_likelihood(params), guess, method='nelder-mead')
        if not soln.success or np.allclose(soln.x, guess):
            raise ValueError('Gaussian process fit to quiescence did not converge. Perhaps try a different minimize '
                             'method or different initial parameters.')
        self.fit_params = soln.x
        self.set_to_best_fit()

    def set_to_best_fit(self):
        self.set_parameter_vector(self.fit_params)
        self.quick_compute()

    def curve(self, t):
        return self.predict(self.f[self.mask], t, return_var=True)