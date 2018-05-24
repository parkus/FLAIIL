import numpy as np
import numbers as nb
import ranges
import identify
from scipy.stats import multivariate_normal
import emcee
from matplotlib import pyplot as plt

def inject_recover(t0, t1, f, e, flare_ranges, qmodel, E, shape_function, trials_per_E=100, options={}):

    # replace flare times with appropriately correlated noise. the emcee business how I handle drawing samples given
    # that the non-flare samples should inform the flare samples. There is probably a cleaner way, but, hey, this works.
    t = (t0 + t1)/2.
    flare = ranges.inranges(t, flare_ranges)
    covar = qmodel.get_matrix(t, include_diagonal=True)
    mean, _ = qmodel.curve(t)
    f_filled = np.copy(f)
    def conditional_loglike(f_fill):
        f_filled[flare] = f_fill
        return multivariate_normal.logpdf(f_filled, mean=mean, cov=covar)
    nflare = int(np.sum(flare))
    nwalkers = 4*nflare
    pos0 = np.random.randn(nwalkers, nflare)*e[flare] + mean[flare][None,:]
    sampler = emcee.EnsembleSampler(nwalkers, nflare, conditional_loglike)
    sampler.run_mcmc(pos0, 100)
    f_sim = sampler.chain[]


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
