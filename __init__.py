import numpy as np
import celerite
from range_utils import rangeset_union, inranges
from scipy.optimize import minimize
from matplotlib import pyplot as plt


# FLAIIL: FLAre Identification in Interrupted Lightcurves

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
        clean = ~np.any(in_flare, 0)

        # plot step if desired
        if plot_steps:
            tnan = np.insert(t, i_gaps, np.nan)
            fnan = np.insert(f, i_gaps, np.nan)
            standard_flareplot(tnan, fnan, anom_ranges, anom_ranges, qmodel)
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
    anom_ranges = reduce(rangeset_union, anom_ranges[1:], anom_ranges[:1])
    sigmas = []
    for rng in anom_ranges:
        inrng = inranges(begs, rng, inclusive=[True, False])
        sigma = fluences[inrng]/np.sqrt(fluence_vars[inrng])
        imx = np.argmax(np.abs(sigma))
        sigma = sigma[imx]
        sigmas.append(sigma)
    sigmas = np.array(sigmas)
    print sigmas
    flare = sigmas > sigma_flare
    flare_ranges = anom_ranges[flare]
    suspect_ranges = anom_ranges[~flare]

    return flare_ranges, suspect_ranges, qmodel


def standard_flareplot(t, f, flare_ranges, suspect_ranges, qmodel):
    plt.figure()
    plt.plot(t, f, 'b.-')
    suspect = inranges(t, suspect_ranges)
    flare = inranges(t, flare_ranges)
    plt.plot(t[suspect], f[suspect], '.', color='orange')
    plt.plot(t[flare], f[flare], 'r.')
    tt = np.linspace(t[0], t[-1], 1000)
    lolo, lolo_var = qmodel.curve(tt)
    lolo_std = np.sqrt(lolo_var)
    plt.plot(tt, lolo, 'k-')
    plt.fill_between(tt, lolo - lolo_std, lolo + lolo_std, color='k', alpha=0.4, edgecolor='none')
    plt.xlabel('Time')
    plt.xlabel('Flux')


class QuiescenceModel(celerite.GP):
    def __init__(self, t, f, e):
        terms = celerite.terms
        kernel = terms.RealTerm(log_a=np.log(np.var(f)), log_c=-10.)
        super(QuiescenceModel, self).__init__(kernel)
        self.t, self.f, self.e = t, f, e
        self.n = len(self.t)
        self.mask = np.ones(len(self.t), bool)

    def fit(self, mask=None):
        if mask is None:
            mask = self.mask
        else:
            self.mask = mask
        self.compute(self.t[mask], self.e[mask])
        def neglike(params):
            self.set_parameter_vector(params)
            loglike = self.log_likelihood(self.f[mask])
            return -loglike
        guess = self.get_parameter_vector()
        soln = minimize(neglike, guess)
        assert soln.status in [0, 2]
        self.set_parameter_vector(soln.x)
        self.compute(self.t[mask], self.e[mask])

    def curve(self, t):
        return self.predict(self.f[self.mask], t, return_var=True)


def run_slices(x, endpts=True):
    """
    Return the slice indices that separate runs (sequences of points above or below zero) in x.
    """

    # first all indices where value crosses from positive to negative or vice versa
    pospts = (x > 0)
    negpts = (x < 0)
    crossings = (pospts[:-1] & negpts[1:]) | (negpts[:-1] & pospts[1:])
    arg_cross = np.nonzero(crossings)[0] + 1

    # now all indices of the middle zero in all series of zeros
    zeropts = (x == 0)
    zero_beg, zero_end = block_edges(zeropts)
    arg_zero = (zero_beg + zero_end + 1) // 2

    # combine
    splits = np.sort(np.concatenate((arg_cross, arg_zero)))

    # return with or without (0, len(x)) endpts added
    if endpts:
        return np.insert(splits, [0, len(splits), [0, len(x)]])
    else:
        return splits


def block_edges(x):
    """
    Returns the beginning and end slice index of each block of true values.
    """
    a = np.insert(x, [0, len(x)], [False, False])
    a = a.astype('i1')
    chng = a[1:] - a[:-1]
    beg, = np.nonzero(chng == 1)
    end, = np.nonzero(chng == -1)
    return beg, end


def gappy_runs(t, y, t_ends, t_gap_mid):
    # find the times of zero crossings
    t_roots = roots(t, y)
    gaproots = inranges(t_roots, t_ends[1:-1].reshape((-1, 2)))
    t_roots =t_roots[~gaproots]

    # consider the start and end of each exposure a zero crossing, so add those in
    i_ends = np.searchsorted(t_roots, t_ends)
    t_roots = np.insert(t_roots, i_ends, t_ends)

    # get beginning and end times of runs
    irootgaps = np.searchsorted(t_roots, t_gap_mid)
    begs = np.delete(t_roots, irootgaps - 1)[:-1]
    ends = np.delete(t_roots, irootgaps)[1:]

    return begs, ends


def roots(x, y):
    """
    Find the roots of some data by linear interpolation where the y values cross the x-axis.

    For series of zeros, the midpoint of the zero values is given.

    Parameters
    ----------
    x : array
    y : array

    Returns
    -------
    x0 : array
        x values of the roots
    """
    sign = np.sign(y)

    # zero values where data crosses x axis
    c = crossings = np.nonzero(abs(sign[1:] - sign[:-1]) == 2)[0] + 1
    a = np.abs(y)
    x0_crossings = (x[c]*a[c-1] + x[c-1]*a[c])/(a[c] + a[c-1])

    # zero values where data falls on x axis
    zero_start, zero_end = block_edges(y == 0)
    x0_zero = (x[zero_start] + x[zero_end-1]) / 2.0

    return np.unique(np.hstack([x0_crossings, x0_zero]))


def exposure_gaps(t0, t1):
    gaps = (t0[1:] - t1[:-1]) > ((t1[0] - t0[0])/1e3)
    igaps, = np.nonzero(gaps)
    igaps += 1
    return igaps

