"""Numerical utilities."""

from __future__ import division, print_function, absolute_import

import numpy as np
from . import ranges

def conditional_qmodel_draw(qmodel, t_known, f_known, t_new):
    """
    Draw correlated noise from a GP model using known values as a prior.

    Intended for filling in gaps in a lightcurve with simulated noisy data in a way that makes sense. Not including the
    actual data as a prior results in nonsense.

    Parameters
    ----------
    qmodel : qmodel.QuiescenceModel
    t_known : array
        time points of known data
    f_known : array
        flux of known data
    t_new : array
        time points where new data should be generated

    Returns
    -------
    f_new : array
        fluxes at the new points

    """
    # if white noise, just generate from a normal distribution
    if qmodel.white:
        mu, std = qmodel.mu(), qmodel.sigma()
        return np.random.normal(mu, std, len(t_new))
    else:
        # the following is based on https://newonlinecourses.science.psu.edu/stat505/node/43/
        # you will need to go look at that for the code below to make sense

        # get necessary covariance matrices
        covar11 = qmodel.get_matrix(t_new)
        covar22 = qmodel.get_matrix(t_known)
        covar12 = qmodel.get_matrix(t_new, t_known)
        covar21 = qmodel.get_matrix(t_known, t_new)

        # get vectors of means
        mu1, _ = qmodel.curve(t_new)
        mu2, _ = qmodel.curve(t_known)

        # known fluxes
        f2 = f_known

        # make everything a matrix (for more readable math)
        matrices = list(map(np.matrix, [f2, mu1, mu2, covar11, covar22, covar12, covar21]))
        f2, mu1, mu2, covar11, covar22, covar12, covar21 = matrices

        # compute conditional covariance matrix and mean vector to specificy multivariate normal distribution
        inv22 = covar22 ** -1
        f2, mu1, mu2 = [a.T for a in [f2, mu1, mu2]]
        conditional_mean = mu1 + covar12 * inv22 * (f2 - mu2)
        conditional_covar = covar11 - covar12 * inv22 * covar21
        conditional_mean = np.array(conditional_mean).squeeze()

        # generate rvs from multivariate normal
        return np.random.multivariate_normal(conditional_mean, conditional_covar)


def run_slices(x, endpts=True):
    """
    Return the slice indices that separate runs (sequences of points above or below zero) in x.

    Parameters
    ----------
    x : array
    endpts : True|False
        If True, include 0, len(x) as endpoints of the runs

    Returns
    -------
    run_indices : array
        Slice indices dividing the runs.
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

    Parameters
    ----------
    x : array

    Returns
    -------
    beg, end : arrays

    """
    a = np.insert(x, [0, len(x)], [False, False])
    a = a.astype('i1')
    chng = a[1:] - a[:-1]
    beg, = np.nonzero(chng == 1)
    end, = np.nonzero(chng == -1)
    return beg, end


def gappy_runs(t, y, t_ends, t_gap_mid):
    """
    Return the start and end times of runs in data with gaps. Runs are not allowed to extend into the data gaps.

    Parameters
    ----------
    t, y : arrays
        Data points.
    t_ends : array
        Start and ends of each span of data (e.g., each exposure).
    t_gap_mid : array
        Midpoints of the gaps between data spans.

    Returns
    -------
    begs, ends : arrays
        The value (not index!) of the start and end of each gap.

    """
    # find the times of zero crossings
    t_roots = roots(t, y)
    gaproots = ranges.inranges(t_roots, t_ends[1:-1].reshape((-1, 2)))
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
    """
    Infer the index of gaps between continguous spans of binned data.

    Parameters
    ----------
    t0 : array
        starts of time bins
    t1 : array
        ends of time bins

    Returns
    -------
    igaps : array
        Slice index of the gaps.
    """
    gaps = (t0[1:] - t1[:-1]) > ((t1[0] - t0[0])/1e3)
    igaps, = np.nonzero(gaps)
    igaps += 1
    return igaps


def get_repeats(ary_list):
    """
    Find vectors that repeat.

    Intended to identify when sequences of data points flagged as within a flare begin to recur in successive iterations
    of the flare-finding algorithm and then return those sequences.

    Parameters
    ----------
    ary_list : list
        A list of data vectors, each of the same length.

    Returns
    -------
    repeated_arys : list
        A list of the data vectors that repeat. If no repeats, None is returned.
    """
    # for all "chunk sizes" from 1 to half the length of the list
    for n in range(1, int(len(ary_list)/2)):
        chunk1 = ary_list[-n:] # last chunk of data vectors
        chunk2 = ary_list[-2*n:-n] # second to last chunk of data vectors
        # if all data vectors in the last and second to last chunk match, we have a repeat. return it.
        if all([np.all(r1 == r2) for r1, r2 in zip(chunk1, chunk2)]):
            return ary_list[-n:]

    # if no repeats, return none.
    return None


