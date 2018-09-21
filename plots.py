import numpy as np
import ranges
from matplotlib import pyplot as plt


def standard_flareplot(t, f, flare_ranges, suspect_ranges, qmodel):
    """
    Plot a lightcurve showing the identified flares and suspect areas.

    Intended for observing the behavior of the flare identification code.

    Parameters
    ----------
    t : array
    f : array
    flare_ranges : Nx2 array
    suspect_ranges : Nx2 array
    qmodel : qmodel.QuiescenceModel

    Returns
    -------
    None

    """
    plt.figure()

    # plot all data
    plt.plot(t, f, 'b.-')

    # overplot flare and suspect data as red and orange points
    suspect = ranges.inranges(t, suspect_ranges)
    flare = ranges.inranges(t, flare_ranges)
    plt.plot(t[suspect], f[suspect], '.', color='orange')
    plt.plot(t[flare], f[flare], 'r.')

    # overplot qmodel
    tt = np.linspace(t[0], t[-1], 1000)
    lolo, lolo_var = qmodel.curve(tt)
    lolo_std = np.sqrt(lolo_var)
    plt.plot(tt, lolo, 'k-')

    # overplot qmodel error region
    plt.fill_between(tt, lolo - lolo_std, lolo + lolo_std, color='k', alpha=0.4, edgecolor='none')

    # label axes
    plt.xlabel('Time')
    plt.xlabel('Flux')