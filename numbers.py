import numpy as np
import ranges


def conditional_qmodel_draw(qmodel, t_known, f_known, t_new):
    # from https://newonlinecourses.science.psu.edu/stat505/node/43/
    qmodel = qmodel
    covar11 = qmodel.get_matrix(t_new)
    covar22 = qmodel.get_matrix(t_known)
    covar12 = qmodel.get_matrix(t_new, t_known)
    covar21 = qmodel.get_matrix(t_known, t_new)
    mu1, _ = qmodel.curve(t_new)
    mu2, _ = qmodel.curve(t_known)
    f2 = f_known
    matrices = map(np.matrix, [f2, mu1, mu2, covar11, covar22, covar12, covar21])
    f2, mu1, mu2, covar11, covar22, covar12, covar21 = matrices
    inv22 = covar22 ** -1
    f2, mu1, mu2 = [a.T for a in [f2, mu1, mu2]]
    conditional_mean = mu1 + covar12 * inv22 * (f2 - mu2)
    conditional_covar = covar11 - covar12 * inv22 * covar21
    conditional_mean = np.array(conditional_mean).squeeze()
    return np.random.multivariate_normal(conditional_mean, conditional_covar)


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
    gaps = (t0[1:] - t1[:-1]) > ((t1[0] - t0[0])/1e3)
    igaps, = np.nonzero(gaps)
    igaps += 1
    return igaps