import numpy as np
import numbers as nb
import ranges
import identify

def inject_recover(t0, t1, f, e, flare_ranges, E, shape_function, trials_per_E=100, options={}):

    # trim to quiescent times
    t = (t0 + t1)/2.
    q = ranges.inranges(t, flare_ranges)
    t0, t1, t, f, e = [a[q] for a in  [t0, t1, t, f, e]]

    # pick a series of random times within the dataset
    T = np.sum(t1 - t0)
    t_flares = np.random.uniform(0, T, trials_per_E)
    i_gaps = nb.exposure_gaps(t0, t1)
    for i in i_gaps:
        t_flares[t_flares > t1[i]] += t0[i+1] - t1[i]

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
            fr, _, _ = id(f)
            if len(fr) > 1:
                n_detected += 1
        completeness.append(float(n_detected)/trials_per_E)

    return np.array(completeness)
