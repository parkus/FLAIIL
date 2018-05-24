import numpy as np
import ranges
from matplotlib import pyplot as plt


def standard_flareplot(t, f, flare_ranges, suspect_ranges, qmodel):
    plt.figure()
    plt.plot(t, f, 'b.-')
    suspect = ranges.inranges(t, suspect_ranges)
    flare = ranges.inranges(t, flare_ranges)
    plt.plot(t[suspect], f[suspect], '.', color='orange')
    plt.plot(t[flare], f[flare], 'r.')
    tt = np.linspace(t[0], t[-1], 1000)
    lolo, lolo_var = qmodel.curve(tt)
    lolo_std = np.sqrt(lolo_var)
    plt.plot(tt, lolo, 'k-')
    plt.fill_between(tt, lolo - lolo_std, lolo + lolo_std, color='k', alpha=0.4, edgecolor='none')
    plt.xlabel('Time')
    plt.xlabel('Flux')