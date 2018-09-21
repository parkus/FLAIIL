# FLAIIL

FLAre Identification in Interrupted Lightcurves,  the algorithm used for identifying flares in the FUV lightcurves from the MUSCLES program -- see Loyd et al. 2018 (http://arxiv.org/abs/1809.07322).

The meat of the code is the flaiil.identify.identify_flares function. See the docstring for usage. Also useful are the qmodel.QuiescenceModel class for modeling quiescent variations with a Gaussian Process (an extension of Dan Foreman-Mackey's GP class from the Gaussian Process code `celerite`, https://celerite.readthedocs.io/en/stable/ -- I aspire to one day make code as clean, elegant, and well-documented as his!) and the qmodel.lightcurve_fill function for using the generating realistic quiescent data to fill areas where flares are removed  for the purposes of injecting fake flares to see how well you can recover them.

If you would  like for me to write  up an usage example, please open an issue to let me  know! I merely don't want to  put forth the effort until it is needed.
