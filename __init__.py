import numpy as np
import celerite
from range_utils import rangeset_union, inranges
from scipy.optimize import minimize
from matplotlib import pyplot as plt


# FLAIIL: FLAre Identification in Interrupted Lightcurves


def identify_flares(t0, t1, f, e, options={}, plot_steps=False):
    # parse options
    maxiter = options.get('maxiter', 10*len(f))
    gap_factor = options.get('gap_factor', 0.5)
    sigma_clip_factor = options.get('sigma_clip_factor', 2.)
    sigma_flare = options.get('sigma_flare', 3.)
    preclean = options.get('preclean', None)
    spread_factor = options.get('spread_factor', 1.0)
    end_window = options.get('end_window', 100.)

    #region setup to handle gaps and other unchanging arrays
    t = (t0 + t1)/2.0
    i_gaps = exposure_gaps(t0, t1)
    tnan = np.insert(t, i_gaps, np.nan)
    fnan = np.insert(f, i_gaps, np.nan)
    t_gap_beg, t_gap_end = t1[i_gaps - 1], t0[i_gaps]
    t_ends = np.sort(np.concatenate([t0[:1], t_gap_beg, t_gap_end, t1[-1:]]))
    t_gap_mid = (t_ends[1:-1:2] + t_ends[2::2]) / 2.
    dt_gaps = t_gap_end - t_gap_beg
    dt = t1 - t0
    t_edges = np.unique(np.concatenate([t0, t1]))

    # prep for smoothing
    dt_smooth = np.mean(dt)/10.
    t_smooth = np.arange(t_edges[0],  t_edges[-1] + dt_smooth, dt_smooth)
    exposure_ranges = np.array([t_ends[:-1:2], t_ends[1::2]]).T
    exposure_ranges[:,0] -= end_window
    exposure_ranges[:,1] += end_window
    exposure_ranges = reduce(rangeset_union, exposure_ranges[1:], exposure_ranges[:1])
    t_smooth = t_smooth[inranges(t_smooth, exposure_ranges)]
    t_smooth0 = t_smooth - end_window/2.
    t_smooth1 = t_smooth + end_window/2.
    #endregion

    q = QuiescenceModel(t, f, e)
    if preclean is None:
        # first pass using sigma-clipped points
        clean = np.abs(f - np.nanmedian(f)) < sigma_clip_factor*np.nanstd(f)
        q.fit(clean)
        lo, lo_var = q.curve(t)
    else:
        clean = preclean
        q.fit(clean)
        lo, lo_var = q.curve(t)
    count = 0
    while True:
        # not exactly doing this in the most readable way ever in the hopes of some speed increases since monte-carlo
        # simulations of false alarm probabilities and completeness are likely

        # get quiescence-subtracted values
        hi = f - lo
        hi_var = e**2 + lo_var

        #region compute fluences of runs above and below quiescence
        # compute cumulative integral
        F = dt * hi
        V = dt**2 * hi_var
        F = np.insert(F, i_gaps, 0)
        V = np.insert(V, i_gaps, dt_gaps**2*q.curve(t_gap_mid)[1])
        Iedges = np.insert(np.cumsum(F), 0, 0)
        Vedges = np.insert(np.cumsum(V), 0, 0)
        Iinterp = lambda t: np.interp(t, t_edges, Iedges)
        Vinterp = lambda t: np.interp(t, t_edges, Vedges)

        # interpolate fluences of runs above and below quiescence from cumulative integral
        begs, ends = gappy_runs(t, hi, t_ends, t_gap_mid)
        fluences = Iinterp(ends) - Iinterp(begs)
        fluence_vars = Vinterp(ends) - Vinterp(begs)

        # flag runs that are anomalous as flares
        flare = (fluences > 0) & (fluences/np.sqrt(fluence_vars) > sigma_flare)

        # redefine edges of flares according to where a moving average becomes consistent with quiescence
        hi_smooth = (Iinterp(t_smooth1) - Iinterp(t_smooth0))/end_window
        flare_ranges = []
        for i in np.nonzero(flare)[0]:
            zero = (hi_smooth <= 0)
            zero_before = zero & (t_smooth <= begs[i])
            zero_after = zero & (t_smooth >= ends[i])
            i_beg = np.max(np.nonzero(zero_before)[0])
            i_end = np.min(np.nonzero(zero_after)[0])
            beg = t_smooth[i_beg] + end_window/2.
            end = t_smooth[i_end] - end_window/2.
            flare_ranges.append([beg, end])
        flare_ranges = np.array(flare_ranges)

        # combine overlapping flares and recompute fluences
        flare_ranges = reduce(rangeset_union, flare_ranges[1:], flare_ranges[:1])

        # update the flare point mask, "spreading" out the mask from the flares a bit to be conservative
        oldclean = clean
        # spans = flare_ranges[:,1] - flare_ranges[:,0]
        # flare_ranges[:,1] = flare_ranges[:,1] + spread_factor*spans
        in_flare = [np.searchsorted(rng, t) == 1 for rng in flare_ranges]
        clean = ~np.any(in_flare, 0)

        # plot step if desired
        if plot_steps:
            plt.figure()
            plt.plot(tnan, fnan,'b.-')
            plt.plot(t[~clean], f[~clean], 'r.')
            tt = np.linspace(t_edges[0], t_edges[-1], 1000)
            lolo, lolo_var = q.curve(tt)
            lolo_std = np.sqrt(lolo_var)
            plt.plot(tt, lolo, 'k-')
            plt.fill_between(tt, lolo-lolo_std, lolo+lolo_std, color='k', alpha=0.4, edgecolor='none')
            plt.xlabel('time')
            plt.xlabel('flux')
            plt.title('Iteration {}'.format(count))
            raw_input('Enter to close figure and continue.')

        # check for convergence
        if np.all(clean == oldclean):
            break
        if count > maxiter:
            raise StopIteration('Iteration limit of {} exceeded.'.format(maxiter))

        if plot_steps:
            plt.close()

        # fit new quiescence points
        q.fit(clean)
        lo, lo_var = q.curve(t)
        count += 1

    return flare_ranges, q


class QuiescenceModel(celerite.GP):
    def __init__(self, t, f, e):
        terms = celerite.terms
        kernel = terms.RealTerm(log_a=np.log(np.var(f)), log_c=-10.) \
                 + terms.JitterTerm(log_sigma=np.log(np.std(f)))
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


