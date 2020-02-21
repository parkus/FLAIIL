# FLAIIL

FLAre Identification in Interrupted Lightcurves,  the algorithm used for identifying flares in the FUV lightcurves from the MUSCLES program -- see Loyd et al. 2018 (http://arxiv.org/abs/1809.07322).

The meat of the code is the flaiil.identify.identify_flares function. See the docstring for usage. Also useful are the qmodel.QuiescenceModel class for modeling quiescent variations with a Gaussian Process (an extension of Dan Foreman-Mackey's GP class from the Gaussian Process code `celerite`, https://celerite.readthedocs.io/en/stable/ -- I aspire to one day make code as clean, elegant, and well-documented as his!) and the qmodel.lightcurve_fill function for using the generating realistic quiescent data to fill areas where flares are removed  for the purposes of injecting fake flares to see how well you can recover them.

## Quick Start

```
from astropy import table
import os
import flaiil
ex_file = os.path.join(flaiil.__path__[0], 
                       'gj876_example_fuv_lightcurve.ecsv')
lc = table.Table.read(ex_file)
```

Have a look at the lightcurve.
```
plt.figure()
plt.plot(lc['t'], lc['f'], '.')
```

You will see a giant gap in the data. You can zoom in on the first epoch of data, which has one small gap.
```
plt.xlim(-200, 6000)
```

The second epoch of data has four gaps.
```
offset = 1.1055e8
rng = np.array((-5000, 22000))
plt.xlim(rng + offset)
```

You should see a few very clear flares in the data, some smaller flares, 
and some questionable flares.

Let's see what `FLAIIL` decides are flares. 
```
from flaiil.identify import identify_flares as find
results = find(lc['t0'], lc['t1'], lc['f'], lc['e'])
```

Oops, **GOTCHA**! It seems `celerite` will give you a nan value if you try to predict data way beyond the decay timescale of the kernel. There are a variety of ways you might choose to deal with this. The easiest is to just analyze different epochs of data separately. However, this means the quiescence model will differ between the two. 

Let's just analyze the later epoch of this dataset.

```
keep = lc['t'] > 6000
lc = lc[keep]

# just for nicer time values...
for s in ['t', 't0', 't1']:
    lc[s] -= offset
```

Now let's try again. Use `plot_steps=True` to see the iterative flare identification. 
```
results = find(lc['t0'], lc['t1'], lc['f'], lc['e'], plot_steps=True)
flare_ranges, suspect_ranges, qmodel = results
```

You will see the algorithm identify the really obvious events on the first pass. Then on successive passes it will identify larger ranges as belonging to these events, as well as a lot of smaller events as the quisecence model gets more and more refined. 

Note that `plot_steps` shows all ranges that aren't being used to fit the quiescence model in red. But actually the model has identified many of these as only "suspect" -- too minor to be considered flares but also statistically a little to anomalous to be safely considered quiescence. `FLAIIL` includes a utility function to plot up the results.

The results you get are the time ranges of the flare and suspect ranges along with the final quiescence model.

```
from flaiil import plots
plt.figure()
plots.standard_flareplot(lc['t'], lc['f'], *results) 
```

This shows what `FLAIIL` decided was suspect in orange and what it decided was a flare in red. (Yeah, the lines between the data are annoying in the gaps. If this really bothers you, you can put some nan points in the gaps before you plot.)

You will see that the default quiescence model is a straight line in the gaps. I believe this is because the kernel has no periodic component to it, so the most likely prediction is a line. However, if you use the kernel to generate fake random data, you will see autocorrelations like might have initially expected. Here, let's fill in those gaps with some fake random data "drawn" from the quiescence model. This is actually one of the best features of `FLAIIL`, as it allows you to fill in places where you identified flares with realistic, noisy data (incorporating both photometric and astrophysical noise!) for the purposes of injection and retrieval. Let's try that out. There is a utility function for it.

```
from flaiil.qmodel import lightcurve_fill as fill
fill_ranges = np.vstack((flare_ranges, suspect_ranges))
isort = np.argsort(fill_ranges[:,0])
fill_ranges = fill_ranges[isort,:]
f, e = fill(lc['t'], lc['f'], lc['e'], qmodel, fill_ranges)
```

It does take a second to do this. Some gnarly matrix math is involved that draws from the queiscence model *conditional upon the existing data!* Without that conditioning, you get unphysical jumps between the actual data and the filled data drawn from the model. 

Potential GOTCHA: If you don't fill the suspect ranges as well as the flare ranges, you will see odd jumps in the filled data because the quiescence model was fit omitting the suspect data.

```
plt.plot(lc['t'], f, 'k.', alpha=0.5)
```

Note especially the filled region of the second to last exposure. At the start of the exposure, the quiescent data are steadily declining. Hence, when drawing from the quiescence model conditional upon that data, you see the simulated data continue to decline before the data start to return back towards the mean expected value from the quiescence model. 

What you decide to call a flare or not -- or, more accurately, what you decide to tell `FLAIIL` to call a flare or not is highly subjective. As a field, we don't seem to have a consensus on how one should define a flare. You can play with what you want to make the dividing line with `FLAIIL` through an options dictionary that you supply to `identify_flares`. For example, if we want to be more choosy about what we call suspect and what we call flares we could do

```
opts = {'sigma_suspect': 5,
        'sigma_flare': 10}
results = find(lc['t0'], lc['t1'], lc['f'], lc['e'], options=opts,
               plot_steps=True)
```

In this case, we get no suspect ranges and the ranges of the big flares are a little smaller. See the docstring of `identify_flares` for all the options you can define. 

One last caution: sometimes `identify_flares` gets stuck in an oscillating solution. It will identify a point as a flare on one iteration and make a new quiescence model. Then on the next iteration it will decide that point wasn't a flare. Then it is again. And so on. You might have to write some code to watch out for this and react to it (e.g., with a maximum iterations limit or such) if you are automating the use of `FLAIIL`. 