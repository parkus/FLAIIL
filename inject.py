import numpy as np
import numbers as nb
import ranges
import identify
import emcee
from matplotlib import pyplot as plt

def inject_recover(t0, t1, f, e, flare_ranges, qmodel, E, shape_function, trials_per_E=100, options={}):

    # replace flare times with appropriately correlated noise. this invovles computing a covariance matrix conditional
    # upon the known (i.e. non-flare) data, which is kinda nasty
    t = (t0 + t1)/2.
    flare = ranges.inranges(t, flare_ranges)
    # from https://newonlinecourses.science.psu.edu/stat505/node/43/
    covar11 = qmodel.get_matrix(t[flare])
    covar22 = qmodel.get_matrix(t[~flare])
    covar12 = qmodel.get_matrix(t[flare], t[~flare])
    covar21 = qmodel.get_matrix(t[~flare], t[flare])
    mean, _ = qmodel.curve(t)
    mu1 = mean[flare]
    mu2 = mean[~flare]
    f2 = f[~flare]
    matrices = map(np.matrix, [f2, mu1, mu2, covar11, covar22, covar12, covar21])
    f2, mu1, mu2, covar11, covar22, covar12, covar21 = matrices
    inv22 = covar22**-1
    f2, mu1, mu2 = [a.T for a in [f2, mu1, mu2]]
    conditional_mean = mu1 + covar12 * inv22 * (f2 - mu2)
    conditional_covar = covar11 - covar12 * inv22 * covar21
    conditional_mean = np.array(conditional_mean).squeeze()
    f_fill = np.random.multivariate_normal(conditional_mean, conditional_covar)
    isort = np.argsort(f)
    e_fill = np.interp(mean[flare], f[isort], e[isort])
    f_fill += np.random.randn(np.sum(flare))*e_fill

    # pick a series of random times within the dataset
    T = np.sum(t1 - t0)
    t_flares = np.random.uniform(0, T, trials_per_E)
    i_gaps = nb.exposure_gaps(t0, t1)
    for i in i_gaps:
        t_flares[t_flares > t1[i]] += t0[i+1] - t1[i]

    # just for brevity
    id = lambda f: identify.identify_flares(t0, t1, f, e, options=options)

    # make sure no flares are detected in the null case
    fr, _, _ = id(f)
    if len(fr):
        raise ValueError("Flares identified even when there are no flares. Make sure to supply flare_ranges from a"
                         "run of identify_flares on the same dataset with the same options provided here.")

    completeness = []
    for EE in E:
        n_detected = 0
        for t_flare in t_flares:
            f_flare = shape_function(EE, t - t_flare)
            f_test = f + f_flare
            fr, _, _ = id(f_test)
            if len(fr) > 1:
                n_detected += 1
        completeness.append(float(n_detected)/trials_per_E)

    return np.array(completeness)
