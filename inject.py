import numpy as np
import numbers as nb
import ranges
import identify
from scipy.integrate import quad
from matplotlib import pyplot as plt

def inject_recover(t0, t1, f, e, energy0, shape_function, trials_per_E=100, options={}, silent=False,
                   sampling_factor=5., Nmax=50):

    # just for brevity
    t = (t0 + t1)/2.
    get_flares = lambda f, e: identify.identify_flares(t0, t1, f, e, options=options)
    flare_ranges, _, qmodel = get_flares(f, e)

    # pick a series of random times within the dataset
    T = np.sum(t1 - t0)
    t_flares = np.random.uniform(0, T, trials_per_E)
    i_gaps = nb.exposure_gaps(t0, t1)
    for i in i_gaps:
        t_flares[t_flares > t1[i-1]] += t0[i] - t1[i-1]

    # replace flare times with appropriately correlated noise. this invovles computing a covariance matrix conditional
    # upon the known (i.e. non-flare) data, which is kinda nasty
    if len(flare_ranges) > 0:
        # pull random draws to fill where flares were until no false positives occur
        flare = ranges.inranges(t, flare_ranges)

        # estimate white noise error in flare range
        isort = np.argsort(f)
        mean_flare, _ = qmodel.curve(t[flare])
        e_sim = np.interp(mean_flare, f[isort], e[isort])

        f_filled, e_filled = map(np.copy, [f, e])
        count = 1
        while True:
            # draw random fluxes and estimate what the uncertainty estimate would have been for those points
            f_fill = nb.conditional_qmodel_draw(qmodel, t[~flare], f[~flare], t[flare])
            f_fill += np.random.randn(np.sum(flare))*e_sim
            e_fill = np.interp(f_fill, f[isort], e[isort])
            e_filled[flare] = e_fill
            f_filled[flare] = f_fill

            # see if the simulated data result in any false positive flares
            fr, _, _ = get_flares(f_filled, e_filled)
            if len(fr) == 0:
                break
            if count > 10:
                raise ValueError("Flares are repeatedly identified even when there are no flares. This doesn't make sense."
                                 "You'll have to dig into the code for this one, I'm afraid. ")
            count += 1

    # now inject flares and see if they are detected
    Etrials, completeness = [], []

    def get_completeness(logE):
        E = 10**logE
        if not silent:
            print '    Sampling E = {:.2g}'.format(E)
        n_detected = 0
        n_trials = 0
        for t_flare in t_flares:
            f_flare = shape_function(E, t - t_flare)
            f_test = f_filled + f_flare
            e_test = np.sqrt(f_test/f_filled)*e
            try:
                fr, _, _ = get_flares(f_test, e_test)
                if len(fr) >= 1:
                    n_detected += 1
                n_trials += 1
            except StopIteration:
                continue
        cmplt = float(n_detected)/n_trials
        if not silent:
            print '        completeness {:.2g}'.format(cmplt)
        Etrials.append(E)
        completeness.append(cmplt)
        return cmplt

    # find where completeness is 0 and 1
    try:
        if not silent:
            print 'Searching for completeness of 0...'
        logE = np.log10(energy0)
        while True:
            cmplt = get_completeness(logE)
            if cmplt > 0:
                logE -= 0.3
            else:
                break
        logE = np.log10(max(Etrials)*2)
        if not silent:
            print 'Searching for completeness of 1...'
        while True:
            cmplt = get_completeness(logE)
            if cmplt < 1:
                logE += 0.3
            else:
                break
        Eary = np.array(Etrials)
        Cary = np.array(completeness)
        Ea = np.max(Eary[Cary == 0])
        Eb = np.min(Eary[Cary == 1])
        i_mid = np.argmin(np.abs(Cary - 0.5))
        Emid = Eary[i_mid]

        # use a numerical integrator to appropriately smaple between 0 and 1 completeness
        if not silent:
            print ('Using numerical integrator to sample between completeness of 0 and 1 (E = {:.2g} -- {:.2g}.'
                   ''.format(Ea, Eb))
            _ = quad(get_completeness, np.log10(Ea), np.log10(Eb), epsrel=1./sampling_factor, limit=Nmax,
                     points=[Emid])
    except:
        print '!!! Exception occurred. Process terminated early.'
        pass

    return map(np.sort, (Etrials, completeness))
