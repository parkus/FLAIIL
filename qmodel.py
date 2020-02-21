from __future__ import division, print_function, absolute_import

from warnings import warn

import celerite
from . import numbers as nb
import numpy as np
from . import ranges
from scipy.optimize import minimize


class QuiescenceModel(celerite.GP):
    """
    An extension of the `celerite` Gassian Process model object for use in modeling the quiescence variations of a
    stellar lightcurve as correlated noise.
    """

    def __init__(self, t, f, e, tau_min=100., tau_logprior=None, params=None, mask=None, white_limit=2.0):
        """
        Create a QuiscenceModel object that models lightcurve fluctuations as correlated noise with a Gaussian Process
        (GP) where correlations decay as exp(-t/tau) where tau is a constant. Specifically, elements of the covariance
        matrix are parameterized as a*exp(-c*t), where ln(a), ln(c), and the mean data value, mu, are parameters of
        the model.

        An important deviation from a typical GP is that the QuiescenceModel enforces a penalty for "jagged" models
        to promote  smoothness. Specifically, the penalty is based on the power of the model at frequencies of 0.1 Hz.

        Parameters
        ----------
        t, f, e : arrays
            Time, flux, and error of data points.
        tau_min : float
            Minimum correlation time to consider in GP fit. Default
            is 100 s. For no limit, simply set to 0.
        tau_logprior : function, optional
            Prior on the correlation time. Should accept a single
            value for tau as input an return the natural
            logarithm of the likelihood of that value.
        params : list
            Parameters of the GP covariance model, a * exp(-c*t). 
            These are log(a), log(c), and the (constant) mean, mu. 
        mask : bool array, optional
            mask of data points where a value of True indicates a
            point that should be excluded from fitting (e.g., because
            it is in a flare or suspect region). If None, assume
            all data should be included.
        white_limit : float
            Likelihood ratio below which a white noise model should
            be used instead of a correlated noise model. Default is
            2.0 to give preference for the simpler model even if a
            GP gives a slightly better fit.
        """
        # set up celerite GP model as a
        terms = celerite.terms
        kernel = terms.RealTerm(log_a=np.log(np.var(f)), log_c=-7.)
        super(QuiescenceModel, self).__init__(kernel, mean=np.median(f), fit_mean=True)

        # initialize parameters of GP if desired
        if params is not None:
            self.set_parameter_vector(params)

        # set data mask
        if mask is not None:
            self.mask = mask
        else:
            self.mask = np.ones(len(t), bool)

        # store other attributes
        self.tau_min = tau_min
        self.tau_logprior = tau_logprior
        self.t, self.f, self.e = t, f, e
        self.n = len(self.t)
        self.fit_params = None
        self.white_limit = white_limit
        self.white = (self.get_parameter('kernel:log_c') == np.inf)
        self.quick_compute()

    def tau_loglike(self, tau):
        """
        Log likeliood of tau values based on lower limit and any user-defined prio.
        """
        if tau < self.tau_min:
            return -np.inf
        if self.tau_logprior is None:
            return 0.0
        else:
            return self.tau_logprior(tau)

    def smoothness_penalty(self, params):
        """
        Cost function for power at 0.1 Hz frequencies. Exact form was set by trial and error.
        
        Parameters
        ----------
        params : list
            Parameters of the GP covariance model, a * exp(-c*t). 
            These are log(a), log(c), and the (constant) mean, mu. 
        """
        self.set_parameter_vector(params)
        hi_freq_power = self.kernel.get_psd(2*np.pi/10)
        if hi_freq_power == 0:
            return 0.0
        if hi_freq_power == np.inf or np.isnan(hi_freq_power):
            return -np.inf
        else:
            return -5*np.sum(self.mask)*np.sqrt(hi_freq_power)/np.std(self.f[self.mask])

    def quick_compute(self):
        """
        Quickly recompute GP intermediate values (i.e. following a change in the parameters).

        """
        super(QuiescenceModel, self).compute(self.t[self.mask], self.e[self.mask])

    def _get_set_mask(self, mask=None):
        if mask is None:
            return self.mask
        else:
            self.mask = mask
            return mask

    def log_likelihood(self, params):
        """
        Log likelihood of the GP model based on the data likelihood and prior on tau. *Smoothness penalty is not applied
        here* so this can be used for sampling the posterior of parameters without that associated bias.

        Parameters
        ----------
        params : list
            Parameters of the GP covariance model, a * exp(-c*t). 
            These are log(a), log(c), and the (constant) mean, mu. 
        
        Returns
        -------
        Natural log of the likelihood of the data given the model and priors on parameters.
        """
        self.set_parameter_vector(params)
        if self.white:
            return self.log_likelihood_white_noise(params[[0,2]])
        else:
            data_loglike = super(QuiescenceModel, self).log_likelihood(self.f[self.mask])
            return data_loglike + self.tau_loglike(self.tau())

    def cost(self, params):
        """
        Log likelihood of the GP model including data likelihood, prior on tau, and smoothness penalty. This is
        what is actually used for the fitting.

        Parameters
        ----------
        params : list
            Parameters of the GP covariance model, a * exp(-c*t). 
            These are log(a), log(c), and the (constant) mean, mu.
            
        Returns
        -------
        Natural log of the likelihood of the data given the model, priors on parameters, and smoothness penalty.
        """
        return -(self.log_likelihood(params) + self.smoothness_penalty(params))
        # return -self.log_likelihood(params)

    def log_likelihood_white_noise(self, log_sig2_and_mu):
        """
        Log likelihood of a constant mean + white noise model.
        
        
        Parameters
        ----------
        log_sig2_and_mu : list
            Natural log of the "extra" variance and the constant mean.

        Returns
        -------
        Natural log of the likelihood of the data given the model.
        """
        self.set_parameter_vector([log_sig2_and_mu[0], np.inf, log_sig2_and_mu[1]])
        return super(QuiescenceModel, self).log_likelihood(self.f[self.mask])

    def log_likelihood_no_noise(self, mu):
        """
        Log likelihood of a model  with no noise and constant mean.
        
        Parameters
        ----------
        mu : float
            value of constant mean

        Returns
        -------
        Natural log of the likelihood of the data given the model.
        """
        self.set_parameter_vector([-np.inf, np.inf, mu])
        return super(QuiescenceModel, self).log_likelihood(self.f[self.mask])

    def fit(self, mask=None, method='Nelder-Mead'):
        """
        Perform a max-likelihood fit of the QuiescenceModel to the data.
        
        Parameters
        ----------
        mask : bool array
            Mask identifying data to be used in the fit.
        method : str
            numerical method to use in minimizing the likelihood function 
            (of those allowable by scipy.optimize.minimize)

        Returns
        -------
        None. Fit is performed in-place and parameters of the QuiescenceModel object set to the best-fit values.
        """
        self._get_set_mask(mask)
        self.quick_compute()
        guess = self.get_parameter_vector()
        if self.white:
            guess[1] = -7
        soln = minimize(self.cost, guess, method=method)
        soln_white = minimize(lambda params: -self.log_likelihood_white_noise(params), guess[[0,2]], method=method)
        if not (soln.success and soln_white.success):
            raise ValueError('Gaussian process fit to quiescence did not converge. Perhaps try a different minimize '
                             'method or different initial parameters.')
        if soln_white.fun + self.log_likelihood(soln.x) < np.log(self.white_limit):
            self.fit_params = np.array([soln_white.x[0], np.inf, soln_white.x[1]])
            self.white = True
        else:
            self.fit_params = soln.x
            self.white = False
        self.set_to_best_fit()

    def fit_with_frozen_tau(self, mask=None, method='Nelder-Mead'):
        """
        Perform a max-likelihood fit of the QuiescenceModel to the data holding the time constant, tau, constant.

        Parameters
        ----------
        mask : bool array
            Mask identifying data to be used in the fit.
        method : str
            numerical method to use in minimizing the likelihood function
            (of those allowable by scipy.optimize.minimize)

        Returns
        -------
        None. Fit is performed in-place and parameters of the QuiescenceModel object set to the best-fit values.
        """
        self._get_set_mask(mask)
        self.quick_compute()
        logc = self.get_parameter('kernel:log_c')
        if self.white:
            def cost(params):
                return -self.log_likelihood_white_noise([params[0], params[1]])
        else:
            def cost(params):
                return self.cost([params[0], logc, params[1]])
        soln = minimize(cost, [np.log(np.var(self.f)), np.median(self.f)], method=method)
        assert soln.success
        self.fit_params = np.array([soln.x[0], logc, soln.x[1]])
        self.set_to_best_fit()

    def _get_params_boiler(self, params):
        """Boilerplate code for getting the current model parameters as an array."""
        if params is None:
            params = self.get_parameter_vector()
        return np.reshape(params, [-1,3])

    def tau(self, params=None):
        """
        Value of the decay constant tau in the covariance model sigma**2 * exp(-t/tau). If the model is set
        to white noise, this  will be 0.

        Parameters
        ----------
        params : list
            Parameters of the GP covariance model, a * exp(-c*t).
            These are log(a), log(c), and the (constant) mean, mu.
            If None, the parameters currently set in the QuiescenceModel
            are used.
        """
        params = self._get_params_boiler(params)
        return np.squeeze(np.exp(-params[:,1]))

    def sigma(self, params=None):
        """
        Value of the standard deviation in the covariance model sigma**2 * exp(-t/tau). If the model is set
        to white noise, then technically this is sigma in sigma**2 * delta(t) where delta is the Dirac delta function.

        Parameters
        ----------
        params : list
            Parameters of the GP covariance model, a * exp(-c*t).
            These are log(a), log(c), and the (constant) mean, mu.
            If None, the parameters currently set in the QuiescenceModel
            are used.
        """
        params = self._get_params_boiler(params)
        return np.squeeze(np.exp(params[:,0]/2))

    def mu(self, params=None):
        """
        Value of model mean value.

        Parameters
        ----------
        params : list
            Parameters of the GP covariance model, a * exp(-c*t).
            These are log(a), log(c), and the (constant) mean, mu.
            If None, the parameters currently set in the QuiescenceModel
            are used.
        """
        params = self._get_params_boiler(params)
        return np.squeeze(params[:,2])

    def sigma_rel(self, params=None):
        """
        Value of the standard deviation in the covariance model sigma**2 * exp(-t/tau), normalized by the
        mean.

        Parameters
        ----------
        params : list
            Parameters of the GP covariance model, a * exp(-c*t).
            These are log(a), log(c), and the (constant) mean, mu.
            If None, the parameters currently set in the QuiescenceModel
            are used.
        """
        return np.squeeze(self.sigma(params)/self.mu(params))

    def sigma_relative_at_tbin(self, tbin, params=None):
        """
        Current value of the expected standard deviation of the model if values were binned to tbin.

        Parameters
        ----------
        tbin : float
            Width of time bins over which model values are assumed to be averaged.
        params : list
            Parameters of the GP covariance model, a * exp(-c*t).
            These are log(a), log(c), and the (constant) mean, mu.
            If None, the parameters currently set in the QuiescenceModel
            are used.
        """
        params = self._get_params_boiler(params)
        loga, logc, mu = params.T
        sig, c = np.exp(loga/2), np.exp(logc)
        with np.errstate(invalid='ignore', over='ignore'):
            x = c*tbin
            sig_dt = np.sqrt(2*(sig/x)**2 * (x + np.exp(-x) - 1))
            uncorrelated = (x == np.inf)
            if np.any(uncorrelated):
                dt = np.median(np.diff(self.t))
                sig_dt[uncorrelated] = sig[uncorrelated]/np.sqrt(tbin/dt)
        return np.squeeze(sig_dt/mu)

    def set_to_best_fit(self):
        """Return the model parameters to their best-fit values, if they have been computed.
        Else you will get an error."""
        self.set_parameter_vector(self.fit_params)
        self.quick_compute()

    def curve(self, t):
        """Compute the lightcurve of the quiescence  model at t."""
        if self.white:
            return self.mu()*np.ones_like(t), self.sigma()**2*np.ones_like(t)
        else:
            return self.predict(self.f[self.mask], t, return_var=True)


def lightcurve_fill(t, f, e, qmodel, flare_ranges):
    """
    Replace flare times with simulated data based on the qmodel with appropriately correlated noise.

    Parameters
    ----------
    t, f, e : arrays
        Lightcurve points -- time, flux, energy.
    qmodel : QuiescenceModel
        Gassian Process model for quiescent variations in lightcurve.
    flare_ranges : Nx2 array
        Start and end time of each flare.

    Returns
    -------
    f_filled, e_filled : arrays
        Flux and error arrays where regions within flares  have been filled with simulated data.
    """

    f_filled, e_filled = list(map(np.copy, [f, e]))
    if len(flare_ranges) > 0:
        # pull random draws to fill where flares were until no false positives occur
        flare = ranges.inranges(t, flare_ranges)
        if not np.any(flare):
            warn("Flare ranges were supplied, yet no points were within these ranges.")
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