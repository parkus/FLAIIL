import numpy as np
import celerite
from scipy.optimize import minimize
from matplotlib import pyplot as plt
from scipy.stats import normaltest


# FLAIIL: FLAre Identification in Interrupted Lightcurves


def identify_flares(t0, t1, f, e, options={}, plot_steps=False, return_details=False):
    # parse options
    maxiter = options.get('maxiter', 10*len(f))
    gap_factor = options.get('gap_factor', 0.5)
    sigma_clip_factor = options.get('sigma_clip_factor', 3.)
    sigma_flare = options.get('sigma_flare', 5.)
    preclean = options.get('preclean', None)

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
    t_edges = np.append(t0, t1[-1])
    #endregion

    gp = quiescence_gaussian_process(t, f, e)
    if preclean is None:
        # first pass using sigma-clipped points
        clean = np.abs(f - np.nanmedian(f)) < sigma_clip_factor*np.nanstd(f)
        gp.fit(clean)
        lo, lo_var = gp.predict(f[clean], t, return_var=True)
    else:
        clean = preclean
        gp.fit(clean)
        lo, lo_var = gp.predict(f[clean], t, return_var=True)
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
        Iedges = np.insert(np.cumsum(F), 0, 0)
        Vedges = np.insert(np.cumsum(V), 0, 0)

        # interpolate fluences of runs above and below quiescence from cumulative integral
        begs, ends = gappy_runs(t, hi, t_ends, t_gap_mid)
        fluences = np.interp(ends, t_edges, Iedges) - np.interp(begs, t_edges, Iedges)
        fluence_vars = np.interp(ends, t_edges, Vedges) - np.interp(begs, t_edges, Vedges)

        # combine runs that are likely to span a gap
        f_gap = (hi[i_gaps] + hi[i_gaps + 1]) / 2.0
        fluence_gap = f_gap * dt_gaps
        i_left = np.searchsorted(ends, t_gap_beg, side='left')
        i_right = np.searchsorted(begs, t_gap_end, side='left')
        fluence_left = fluences[i_left]
        fluence_right = fluences[i_right]
        fluence_bracket = fluence_left + fluence_right
        combine = (fluence_gap > 0) & (fluence_left > 0) & (fluence_right > 0) & \
                  (fluence_gap < gap_factor*fluence_bracket)
        if np.any(combine):
            j_runs, j_gaps = i_left[combine], np.nonzero(combine)[0]
            j_runs, j_gaps = [a.tolist() for a in [j_runs, j_gaps]]
            var_gaps = gp.predict(f[clean], t_gap_mid[j_gaps], return_var=True)[1] * dt_gaps[j_gaps] ** 2
            # while loop is my solution to problem of adjacent gaps (i.e. 3+ runs  and 2+ gaps need to all be comined)
            while len(j_gaps) > 0:
                j_gap = j_gaps.pop(0)
                j_run = j_runs.pop(0)
                j0, j1 = j_run, j_run+1
                fluence_combined = fluence_bracket[j_gap] + fluence_gap[j_gap]
                fluence_var_combined = fluence_vars[j0] + fluence_vars[j1] + var_gaps[j_gap]
                fluences[j_run] = fluence_combined
                fluence_vars[j_run] = fluence_var_combined
                fluences = np.delete(fluences, j1)
                fluence_vars = np.delete(fluence_vars, j1)
                begs = np.delete(begs, j1)
                ends = np.delete(ends, j0)
        #endregion

        # flag runs that are anomalous as flares
        flare = (fluences > 0) & (fluences/np.sqrt(fluence_vars) > sigma_flare)

        # update the flare point mask
        oldclean = clean
        ranges = np.array([begs, ends]).T
        clean = inranges(t, ranges[~flare])

        # plot step if desired
        if plot_steps:
            plt.figure()
            plt.plot(tnan, fnan,'b.-')
            plt.plot(t[~clean], f[~clean], 'r.')
            tt = np.linspace(t_edges[0], t_edges[-1], 1000)
            lolo, lolo_var = gp.predict(f[oldclean], tt, return_var=True)
            lolo_std = np.sqrt(lolo_var)
            plt.plot(tt, lolo, 'k-')
            plt.fill_between(tt, lolo-lolo_std, lolo+lolo_std, color='k', alpha=0.4, edgecolor='none')
            plt.xlabel('time')
            plt.xlabel('flux')
            plt.title('Iteration {}'.format(count))
            raw_input('Enter to close figure and continue.')
            plt.close()

        # check for convergence
        if np.all(clean == oldclean):
            break
        if count > maxiter:
            raise StopIteration('Iteration limit of {} exceeded.'.format(maxiter))

        # fit new quiescence points
        gp.fit(clean)
        lo, lo_var = gp.predict(f[clean], t, return_var=True)
        count += 1

    if return_details:
        results = dict(begs=begs, ends=ends, flare=flare,
                       fluences=fluences, fluence_errs=np.sqrt(fluence_vars),
                       gp=gp)
        return results
    else:
        return begs, ends, flare


def quiescence_gaussian_process(t, f, e):
    terms = celerite.terms
    kernel = terms.RealTerm(log_a=np.log(np.var(f)), log_c=-10.) \
             + terms.JitterTerm(log_sigma=np.log(np.std(f)))
    gp = celerite.GP(kernel, mean=np.median(f))

    def fit(clean):
        gp.compute(t[clean], e[clean])
        def neglike(params):
            gp.set_parameter_vector(params)
            loglike = gp.log_likelihood(f[clean])
            # residuals = f[clean] - gp.predict(f[clean], t[clean], False, False)
            # z = normaltest(residuals)[0]
            # v = np.var(residuals/e[clean])
            # penalty = z + (v + 1/v)
            # return -(loglike - penalty)
            return -loglike
        guess = gp.get_parameter_vector()
        soln = minimize(neglike, guess)
        assert soln.status in [0, 2]
        gp.set_parameter_vector(soln.x)
    gp.fit = fit

    return gp


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


def inranges(values, ranges, inclusive=[False, True]):
    """Determines whether values are in the supplied list of sorted ranges.

    Parameters
    ----------
    values : 1-D array-like
        The values to be checked.
    ranges : 1-D or 2-D array-like
        The ranges used to check whether values are in or out.
        If 2-D, ranges should have dimensions Nx2, where N is the number of
        ranges. If 1-D, it should have length 2N. A 2xN array may be used, but
        note that it will be assumed to be Nx2 if N == 2.
    inclusive : length 2 list of booleans
        Whether to treat bounds as inclusive. Because it is the default
        behavior of numpy.searchsorted, [False, True] is the default here as
        well. Using [False, False] or [True, True] will require roughly triple
        computation time.

    Returns a boolean array indexing the values that are in the ranges.
    """
    ranges = np.asarray(ranges)
    if ranges.ndim == 2:
        if ranges.shape[1] != 2:
            ranges = ranges.T
        ranges = ranges.ravel()

    if inclusive == [0, 1]:
        return (np.searchsorted(ranges, values) % 2 == 1)
    if inclusive == [1, 0]:
        return (np.searchsorted(ranges, values, side='right') % 2 == 1)
    if inclusive == [1, 1]:
        a = (np.searchsorted(ranges, values) % 2 == 1)
        b = (np.searchsorted(ranges, values, side='right') % 2 == 1)
        return (a | b)
    if inclusive == [0, 0]:
        a = (np.searchsorted(ranges, values) % 2 == 1)
        b = (np.searchsorted(ranges, values, side='right') % 2 == 1)
        return (a & b)


def gappy_runs(t, y, t_ends, t_gap_mid):
    # find the times of zero crossings
    t_roots = roots(t, y)
    gaproots = inranges(t_roots, t_ends[1:-1].reshape((-1,2)))
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