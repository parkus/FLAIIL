import numpy as np


def rangeset_union(ranges0, ranges1):
    invrng0, invrng1 = map(rangeset_invert, [ranges0, ranges1])
    xinv = rangeset_intersect(invrng0, invrng1)
    return rangeset_invert(xinv)


def rangeset_intersect(ranges0, ranges1, presorted=False):
    """
    Return the intersection of two sets of sorted ranges, given as Nx2 array-like.
    """

    if len(ranges0) == 0 or len(ranges1) == 0:
        return np.empty([0, 2])
    rng0, rng1 = map(np.asarray, [ranges0, ranges1])
    rng0, rng1 = [np.reshape(a, [-1, 2]) for a in [rng0, rng1]]

    if not presorted:
        rng0, rng1 = [r[np.argsort(r[:,0])] for r in [rng0, rng1]]
    for rng in [rng0, rng1]:
        assert np.all(rng[1:] > rng[:-1])

    l0, r0 = rng0.T
    l1, r1 = rng1.T
    f0, f1 = [rng.flatten() for rng in [rng0, rng1]]

    lin0 = inranges(l0, f1, [1, 0])
    rin0 = inranges(r0, f1, [0, 1])
    lin1 = inranges(l1, f0, [0, 0])
    rin1 = inranges(r1, f0, [0, 0])

    #keep only those edges that are within a good area of the other range
    l = weave(l0[lin0], l1[lin1])
    r = weave(r0[rin0], r1[rin1])
    return np.array([l, r]).T


def weave(a, b):
    """
    Insert values from b into a in a way that maintains their order. Both must
    be sorted.
    """
    mapba = np.searchsorted(a, b)
    return np.insert(a, mapba, b)


def rangeset_invert(ranges):
    if len(ranges) == 0:
        return np.array([[-np.inf, np.inf]])
    edges = ranges.ravel()
    rnglist = [edges[1:-1].reshape([-1, 2])]
    if edges[0] != -np.inf:
        firstrng = [[-np.inf, edges[0]]]
        rnglist.insert(0, firstrng)
    if edges[-1] != np.inf:
        lastrng = [[edges[-1], np.inf]]
        rnglist.append(lastrng)
    return np.vstack(rnglist)


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