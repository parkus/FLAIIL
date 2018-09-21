import numpy as np
import plots
import ranges
from flaiil import QuiescenceModel
from numbers import gappy_runs, exposure_gaps, get_repeats
from matplotlib import pyplot as plt


def identify_flares(t0, t1, f, e, options={}, plot_steps=False):
    """
    Use an iterative algorithm to identify flares in a lightcurve.

    The algorithm works by first applying a sigma clip, then fitting a model to lightcurve variations and identifying
    "runs" of points above quiescence that are statistically unlikely. The anomalous runs are masked out, the
    quiescent variations refit, and new statistically anomalous runs identified. This process is iterated until
    the points marked as within anomalous runs begins to repeat.

    Parameters
    ----------
    t0 : array
        starting edges of lightcurve time bins
    t1 : array
        ending edges of lightcurve time bins
    f : array
        flux
    e : array
        error
    options : dict
        Dictionary of knobs to turn for the identification algorithm.
        Options include:
            maxiter : Maximum number of iterations allowed.
            Default is len(f)/5.
            sigma_clip_factor : Initial sigma clip factor. Recommend a
                conservative value as large deviations can cause wonky
                quiescence fits. Default is 2.
            sigma_suspect : Cutoff for a run of points above or
                below quiescence to be flagged as anomalous and
                excluded from further fitting, taken as the area of
                the run divided by the uncertainty in that area.
            sigma_flare : Similar to sigma_suspect, but for a run
                of points above quiescence to be flagged as a flare.
            preclean : A boolean array giving a first guess at which
                points should be masked out. False implies point will
                not be fit with quiescence model. Optional. I can't
                remember why I added this, but I think  I had a good
                reason... Default is None.
            extend_factor : Simple factor by which to extend flare
                runs. This is necessary to mitigate confusion between
                the tail of a flare and quiescent variations. Default
                is 2.
            prepend_time : Similar to extend factor, but a simple
                length of time to prepend to the flare runs. This is
                less important than extend_factor since flux generally
                increases very rapidly during flares.
            tau_min : Minimum autocorrelation timescale to consider
                when fitting quiescent variations with Gaussian Process
                model. Helps keep the solution from getting stuck on
                tau = 0 and not exploring other options.
            tau_logprior : A function serving as a prior on the
                autocorrelation timescale, tau. This should accept a
                single value as input and return the ln(likelihood) of
                that value. Optional. The original intent was to use
                this for encouraging smoothness in the quiescence model,
                but ultimately I built this into the QuisecenceModel
                class by default.
    plot_steps : bool
        If True, plot each step of the identification process. Good for
        making sure nothing silly is happening.

    Returns
    -------
    flare_ranges : 2xN array
        Start and end of ranges identified as belonging to flares.
    suspect_ranges : 2xN array
        Start and end of ranges identified as statistically anomalous,
        but not belonging  to flares.
    qmodel : QuiescenceModel
        Object specifying a model for quiescent variations. (More details
        in the documentation for the qmodel.QuisecenceModel class.)
    """

    # parse options
    maxiter = options.get('maxiter', len(f)/5 if len(f)/5 > 25 else 25)
    sigma_clip_factor = options.get('sigma_clip_factor', 2.)
    sigma_suspect = options.get('sigma_suspect', 3.)
    sigma_flare = options.get('sigma_flare', 5.)
    preclean = options.get('preclean', None)
    extend_factor = options.get('extend_factor', 2.)
    white_limit = options.get('white_limit', 2.)
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

    qmodel = QuiescenceModel(t, f, e, tau_min=tau_min, tau_logprior=tau_logprior, white_limit=white_limit)
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
        significance = abs(fluences/np.sqrt(fluence_vars))
        anom = significance > sigma_suspect

        # update the flare point mask, extending the mask from the end of flares to be conservative
        oldclean.append(clean)
        anom_ranges = np.array([begs[anom], ends[anom]]).T
        flares = significance[anom] > sigma_flare
        spans = anom_ranges[flares,1] - anom_ranges[flares,0]
        anom_ranges[flares,1] = anom_ranges[flares,1] + extend_factor*spans
        anom_ranges[flares,0] = anom_ranges[flares,0] - prepend_time

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