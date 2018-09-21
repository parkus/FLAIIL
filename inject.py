import numpy as np
import numbers as nb
import ranges
import identify
from warnings import warn
import plots
from matplotlib import pyplot as plt


def inject_recover(t0, t1, f, e, multiplicative_profile, additive_profile=None):
    pass # for single energy


def explore_completeness(t0, t1, f, e, energy0, shape_function, sample_dex=0.05,
                                 trials_per_E=100, options={}, silent=False, return_on_exception=False,
                                 sample_energies=None):

    # just for brevity
    t = (t0 + t1)/2.
    get_flares = lambda f, e: identify.identify_flares(t0, t1, f, e, options=options)
    flare_ranges, suspect_ranges, qmodel = get_flares(f, e)

    # pick a series of random times within the dataset
    T = np.sum(t1 - t0)
    t_flares = np.random.uniform(t0[0], t0[0] + T, trials_per_E)
    t_flares = np.sort(t_flares) # makes debugging a little easier
    i_gaps = nb.exposure_gaps(t0, t1)
    for i in i_gaps:
        t_flares[t_flares > t1[i-1]] += t0[i] - t1[i-1]

    count = 1
    while True:
        if count > 10:
            raise StopIteration("Having trouble getting a clean lightcurve filled with correlated noise where "
                                "flares were. I'm afraid this will require some digging.")

        f_filled, e_filled = lightcurve_fill(t, f, e, qmodel, flare_ranges)

        # see if the simulated data result in any false positive flares
        # if so, mask those and proceed
        fr, sr, qm = get_flares(f_filled, e_filled)
        if len(fr) > 0:
            flare_ranges = ranges.rangeset_union(flare_ranges, fr)
        else:
            break

        count += 1


    def get_completeness(logE):
        E = 10**logE
        if silent != False:
            print '    Sampling E = {:.2g}'.format(E)
        n_detected = 0
        n_trials = 0
        for t_flare in t_flares:
            f_quiescent = qm.curve(t)
            f_flare = shape_function(E, t_flare, f_quiescent)
            f_test = f_filled + f_flare
            e_test = np.sqrt(f_test/f_filled)*e_filled
            e_extra = np.sqrt(e_test**2 - e_filled**2)
            f_test += np.random.randn(len(f_test))*e_extra
            if silent == 'verbose':
                print '    injecting at {:.1f}, '.format(t_flare),
            try:
                fr, sr, q = get_flares(f_test, e_test)
                if len(fr) >= 1:
                    # plots.standard_flareplot(t, f_test, fr, sr, q)
                    # raw_input('Enter to close and continue.')
                    # plt.close()
                    if silent == 'verbose':
                        print 'detected flare(s) in ' + ', '.join(['[{:.1f}--{:.1f}]'.format(*_fr) for _fr in fr])
                    if ranges.inranges(t_flare, fr):
                        n_detected += 1
                else:
                    if silent == 'verbose':
                        print 'no flares detected'
                n_trials += 1
            except (StopIteration, ValueError):
                if silent == 'verbose':
                    print 'exception occurred'
                continue
        if n_trials < trials_per_E/2.:
            raise ValueError("An error occurred in flare detection for more than half of the injections E = {:.2g} "
                             "(probably due to quiescence fit not converging). You should probably investigate."
                             "".format(E))
        cmplt = float(n_detected)/n_trials
        if silent != False:
            print '        completeness {:.3f}'.format(cmplt)
        return cmplt

    # now inject flares and see if they are detected
    try:
        if sample_energies:
            Etrials = sample_energies
            completeness = map(get_completeness, np.log10(sample_energies))
        else:
            # find where completeness is 0 and 1, sampling at the dex desired
            Etrials, completeness = [], []
            if not silent:
                print 'Searching for completeness of 0...'
            logE = np.log10(energy0)
            while True:
                cmplt = get_completeness(logE)
                Etrials.append(10**logE)
                completeness.append(cmplt)
                if cmplt > 0:
                    logE -= sample_dex
                else:
                    break
            logE = np.log10(max(Etrials)) + sample_dex
            if not silent:
                print 'Searching for completeness of 1...'
            while True:
                cmplt = get_completeness(logE)
                Etrials.append(10 ** logE)
                completeness.append(cmplt)
                if cmplt < 1:
                    logE += sample_dex
                else:
                    break
    except:
        if return_on_exception:
            print '!!! Exception occurred. Process terminated early.'
            pass
        else:
            raise

    isort = np.argsort(Etrials)
    # need to return error on energy
    #, completeness, ))


def lightcurve_fill(t, f, e, qmodel, flare_ranges):
    # replace flare times with appropriately correlated noise. this involves computing a covariance matrix conditional
    # upon the known (i.e. non-flare) data, which is kinda nasty
    f_filled, e_filled = map(np.copy, [f, e])
    if len(flare_ranges) > 0:
        # pull random draws to fill where flares were until no false positives occur
        flare = ranges.inranges(t, flare_ranges)
        if not np.any(flare):
            raise warn("Flare ranges were supplied, yet no points were within these ranges.")
            return f, e

        isort = np.argsort(f) # comes in handy in a sec

        # estimate white noise error in flare range
        mean_flare, _ = qmodel.curve(t[flare])
        e_sim = np.interp(mean_flare, f[isort], e[isort])

        # draw random fluxes and estimate what the uncertainty estimate would have been for those points
        f_fill = nb.conditional_qmodel_draw(qmodel, t[~flare], f[~flare], t[flare])
        f_fill += np.random.randn(np.sum(flare))*e_sim
        e_fill = np.interp(f_fill, f[isort], e[isort])
        e_filled[flare] = e_fill
        f_filled[flare] = f_fill
        return f_filled, e_filled
    else:
        return f, e