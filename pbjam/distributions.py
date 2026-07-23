"""
The distributions module contains a selection of probability densities and methods, 
which are primarily used to build prior probability densities in PBjam. These classes
to some extent mimic those of the scipy.stats package in terms of their inputs and 
functionality.
"""

import jax.numpy as jnp
from functools import partial
import jax
import numpy as np
import jax.scipy.special as jsp
from pbjam import jar
import statsmodels.api as sm

def makeDistObject(data, **kwargs):
    """
    Construct distribution wrappers from an empirical sample.
    
    Parameters
    ----------
    data : array-like
        Two-dimensional sample with shape ``(n_samples, n_dimensions)``.
    **kwargs
        Additional arguments passed to :func:`getQuantileFuncs`.
    
    Returns
    -------
    list of distribution
        One distribution wrapper for each column of ``data``.
    """

    ppfs, pdfs, logpdfs, cdfs = getQuantileFuncs(data, **kwargs)

    D = []
    for i in range(len(ppfs)):
        D.append(distribution(ppfs[i], pdfs[i], logpdfs[i], cdfs[i]))

    return D

def getQuantileFuncs(data, cut=5, densityScale=30, **kwargs):
    """
    Construct marginal distribution functions from an empirical sample.
    
    Each column is treated independently and represented by a univariate kernel
    density estimate.
    
    Parameters
    ----------
    data : array-like
        Two-dimensional sample with shape ``(n_samples, n_dimensions)``.
    cut : float, optional
        Number of kernel bandwidths by which the KDE support extends beyond the
        data range.
    densityScale : int, optional
        Multiplier controlling the resolution of the interpolated quantile grid.
    **kwargs
        Reserved for compatibility.
    
    Returns
    -------
    ppfs : list of callable
        Percent-point functions for each dimension.
    pdfs : list of callable
        Probability density functions for each dimension.
    logpdfs : list of callable
        Log-probability density functions for each dimension.
    cdfs : list of array-like
        Cumulative-density values supplied by the fitted KDE objects.
    """

    ppfs = []

    pdfs = []

    cdfs = []

    logpdfs = []

    for i in range(data.shape[1]):

        kde = sm.nonparametric.KDEUnivariate(np.array(data[:, i]).real)

        kde.fit(cut=cut)

        # TODO currently sampling the unit interval at 5120 points, is this 
        # enough? Increasing doesn't seem to impact evaluation time of the ppf.
        A = jnp.linspace(0, 1, densityScale*len(kde.cdf))

        cdfs.append(kde.cdf)
        
        # The icdf from statsmodels is only evaluated on the input values,
        # not the complete support of the kde-pdf which may be wider because
        # of the kernel bandwidth. 
        x = np.linspace(kde.support[0], kde.support[-1], len(A))
        Q = jar.getCurvePercentiles(x, 
                                    kde.evaluate(x),
                                    percentiles=A)
        
        ppfs.append(jar.jaxInterp1D(A, Q))
        
        # TODO should increase resolution on pdf like on the ppf
        pdfs.append(jar.jaxInterp1D(kde.support, kde.evaluate(kde.support)))

        logpdfs.append(jar.jaxInterp1D(kde.support, jnp.log(kde.evaluate(kde.support))))

    return ppfs, pdfs, logpdfs, cdfs

class beta():
    """
    Beta distribution on a finite interval.
    
    Parameters
    ----------
    a, b : float, optional
        Positive shape parameters.
    loc : float, optional
        Lower bound of the support.
    scale : float, optional
        Width of the support.
    """
    def __init__(self, a=1, b=1, loc=0, scale=1):
        """
        Beta distribution on a finite interval.
        
        Parameters
        ----------
        a, b : float, optional
            Positive shape parameters.
        loc : float, optional
            Lower bound of the support.
        scale : float, optional
            Width of the support.
        """

        # Turn init args into attributes
        self.__dict__.update((k, v) for k, v in locals().items() if k not in ['self'])

        self.logfac = jsp.gammaln(self.a + self.b) - jsp.gammaln(self.a) - jsp.gammaln(self.b) - jnp.log(self.scale) 

        self.fac = jnp.exp(self.logfac)

        self.am1 = self.a - 1

        self.bm1 = self.b - 1

        self._set_stdatt()

    def rv(self):
        """
        Draw one random variate.
        
        Returns
        -------
        float or int
            Random value drawn from the distribution.
        """

        u = np.random.uniform(0, 1)
        
        x = self.ppf(u)
        
        return x
    
    def _set_stdatt(self):
        """
        Set standard summary attributes such as the mean and median.
        """
        self.mean = self.loc + self.scale * self.a / (self.a + self.b)
        self.median = self.ppf(0.5)

    @partial(jax.jit, static_argnums=(0,))
    def _transformx(self, x):
        """
        Map values from the distribution support to the unit interval.
        
        Parameters
        ----------
        x : array-like
            Values on the original support.
        
        Returns
        -------
        array-like
            Transformed values.
        """
        return (x - self.loc) / self.scale

    @partial(jax.jit, static_argnums=(0,))
    def _inverse_transform(self, x):
        """
        Map values from the unit interval to the distribution support.
        
        Parameters
        ----------
        x : array-like
            Values on the unit interval.
        
        Returns
        -------
        array-like
            Values on the original support.
        """
        
        return x * self.scale + self.loc
 
    @partial(jax.jit, static_argnums=(0,))
    def pdf(self, x, norm=True):
        """
        Evaluate the probability density function.
        
        Parameters
        ----------
        x : array-like
            Point or points at which to evaluate the distribution.
        norm : bool, optional
            Include the normalization constant when ``True``.
        
        Returns
        -------
        array-like
            Probability density at ``x``.
        """
 
        _x = self._transformx(x)
   
        T = jax.lax.lt(_x, 0.) | jax.lax.lt(1., _x)  
         
        y = jax.lax.cond(T, lambda : 0., lambda : _x**self.am1 * (1 - _x)**self.bm1)
                
        if norm:
            return y * self.fac

        else:
            return y

        
    @partial(jax.jit, static_argnums=(0,))
    def logpdf(self, x, norm=True):
        """
        Evaluate the log-probability density function.
        
        Parameters
        ----------
        x : array-like
            Point or points at which to evaluate the distribution.
        norm : bool, optional
            Include the normalization constant when ``True``.
        
        Returns
        -------
        array-like
            Log-probability density at ``x``.
        """

        x = jnp.array(x)

        _x = self._transformx(x)
            
        T = jax.lax.lt(_x, 0.) | jax.lax.lt(1., _x)  

        y = jax.lax.cond(T, lambda : -jnp.inf, lambda : self.am1 * jnp.log(_x) + self.bm1 * jnp.log(1-_x))
        
        if norm:
            return y + self.logfac
        else:
            return y

        
    def cdf(self, x):
        """
        Evaluate the cumulative distribution function.
        
        Parameters
        ----------
        x : array-like
            Point or points at which to evaluate the distribution.
        
        Returns
        -------
        array-like
            Cumulative probability at ``x``.
        """

        _x = self._transformx(x)

        y = jsp.betainc(self.a, self.b, _x)

        y = y.at[_x<=0].set(0)

        y = y.at[_x>=1].set(1)

        return y

    @partial(jax.jit, static_argnums=(0,))
    def ppf(self, y):
        """
        Evaluate the percent-point (quantile) function.
        
        Parameters
        ----------
        y : array-like
            Cumulative probability in the interval [0, 1].
        
        Returns
        -------
        array-like
            Quantile corresponding to ``y``.
        """

        _x = self.betaincinv(self.a, self.b, y)

        x = self._inverse_transform(_x)

        return x


    @partial(jax.jit, static_argnums=(0,))
    def update_x(self, x, a, b, p, a1, b1, afac):
        """
        Perform one root-finding update for the inverse incomplete beta function.
        """
        err = jsp.betainc(a, b, x) - p
        t = jnp.exp(a1 * jnp.log(x) + b1 * jnp.log(1.0 - x) + afac)
        u = err/t
        tmp = u * (a1 / x - b1 / (1.0 - x))
        t = u/(1.0 - 0.5 * jnp.clip(tmp, None, 1.0))
        x -= t
        x = jnp.where(x <= 0., 0.5 * (x + t), x)
        x = jnp.where(x >= 1., 0.5 * (x + t + 1.), x)

        return x, t

    @partial(jax.jit, static_argnums=(0,))
    def func_1(sefl, a, b, p):
        """
        Compute an initial inverse-beta estimate when both shapes are at least one.
        """
        pp = jnp.where(p < .5, p, 1. - p)
        t = jnp.sqrt(-2. * jnp.log(pp))
        x = (2.30753 + t * 0.27061) / (1.0 + t * (0.99229 + t * 0.04481)) - t
        x = jnp.where(p < .5, -x, x)
        al = (jnp.power(x, 2) - 3.0) / 6.0
        h = 2.0 / (1.0 / (2.0 * a - 1.0) + 1.0 / (2.0 * b - 1.0))
        w = (x * jnp.sqrt(al + h) / h)-(1.0 / (2.0 * b - 1) - 1.0/(2.0 * a - 1.0)) * (al + 5.0 / 6.0 - 2.0 / (3.0 * h))
        return a / (a + b * jnp.exp(2.0 * w))

    @partial(jax.jit, static_argnums=(0,))
    def func_2(sefl, a, b, p):
        """
        Compute an initial inverse-beta estimate when either shape is below one.
        """
        lna = jnp.log(a / (a + b))
        lnb = jnp.log(b / (a + b))
        t = jnp.exp(a * lna) / a
        u = jnp.exp(b * lnb) / b
        w = t + u

        return jnp.where(p < t/w, jnp.power(a * w * p, 1.0 / a), 1. - jnp.power(b *w * (1.0 - p), 1.0/b))

    @partial(jax.jit, static_argnums=(0,))
    def compute_x(self, p, a, b):
        """
        Select an initial estimate for the inverse incomplete beta function.
        """
        return jnp.where(jnp.logical_and(a >= 1.0, b >= 1.0), self.func_1(a, b, p), self.func_2(a, b, p))

    @partial(jax.jit, static_argnums=(0,))
    def betaincinv(self, a, b, p):
        """
        Approximate the inverse regularized incomplete beta function.
        """
        a1 = a - 1.0
        b1 = b - 1.0

        ERROR = 1e-8

        p = jnp.clip(p, 0., 1.)

        x = jnp.where(jnp.logical_or(p <= 0.0, p >= 1.), p, self.compute_x(p, a, b))

        afac = - jsp.betaln(a, b)
        stop  = jnp.logical_or(x == 0.0, x == 1.0)
        for i in range(10):
            x_new, t = self.update_x(x, a, b, p, a1, b1, afac)
            x = jnp.where(stop, x, x_new)
            stop = jnp.where(jnp.logical_or(jnp.abs(t) < ERROR * x, stop), True, False)

        return x

class distribution():
    """
    Wrap callable PDF, log-PDF, CDF, and PPF functions.
    
    Parameters
    ----------
    ppf, pdf, logpdf, cdf : callable
        Functions implementing the corresponding distribution methods.
    """
    def __init__(self, ppf, pdf, logpdf, cdf):
        """
        Wrap callable PDF, log-PDF, CDF, and PPF functions.
        
        Parameters
        ----------
        ppf, pdf, logpdf, cdf : callable
            Functions implementing the corresponding distribution methods.
        """

        self.pdf = pdf

        self.ppf = ppf

        self.logpdf = logpdf

        self.cdf = cdf

        self._set_stdatt()

    def rv(self):
        """
        Draw one random variate.
        
        Returns
        -------
        float or int
            Random value drawn from the distribution.
        """

        u = np.random.uniform(0, 1)
        
        x = self.ppf(u)
        
        return x
    
    def _set_stdatt(self):       
        """
        Set standard summary attributes such as the mean and median.
        """
        x = jnp.linspace(self.ppf(1e-6), self.ppf(1-1e-6), 1000)

        self.mean = jnp.trapezoid(x * jnp.array([self.pdf(_x) for _x in x]), x)

        self.median = self.ppf(0.5)

class uniform():
    """
    Continuous uniform distribution.
    
    Parameters
    ----------
    loc : float, optional
        Lower bound.
    scale : float, optional
        Width of the support.
    """
    def __init__(self, loc=0, scale=1):
        """
        Continuous uniform distribution.
        
        Parameters
        ----------
        loc : float, optional
            Lower bound.
        scale : float, optional
            Width of the support.
        """

        # Turn init args into attributes
        self.__dict__.update((k, v) for k, v in locals().items() if k not in ['self'])
                
        self.a = self.loc

        self.b = self.loc + self.scale

        self.mean = 0.5 * (self.a + self.b)

        self._set_stdatt()

    
    def rv(self):
        """
        Draw one random variate.
        
        Returns
        -------
        float or int
            Random value drawn from the distribution.
        """

        u = np.random.uniform(0, 1)
        
        x = self.ppf(u)
        
        return x
    
    def _set_stdatt(self):
        """
        Set standard summary attributes such as the mean and median.
        """
        self.mean = self.a + 0.5 * self.scale
        self.median = self.ppf(0.5)

    @partial(jax.jit, static_argnums=(0,))
    def pdf(self, x):
        """
        Evaluate the probability density or mass function.
        
        Parameters
        ----------
        x : array-like
            Point or points at which to evaluate the distribution.
        
        Returns
        -------
        array-like
            Probability density at ``x``.
        """
            
        T = jax.lax.lt(x, self.a) | jax.lax.lt(self.b, x)  
             
        y = jax.lax.cond(T, lambda : 0., lambda : 1./self.scale)
            
        return y

    @partial(jax.jit, static_argnums=(0,))
    def logpdf(self, x):
        """
        Evaluate the log-probability density function.
        
        Parameters
        ----------
        x : array-like
            Point or points at which to evaluate the distribution.
        
        Returns
        -------
        array-like
            Log-probability density at ``x``.
        """
 
        T = jax.lax.lt(x, self.a) | jax.lax.lt(self.b, x)  
            
        y = jax.lax.cond(T, lambda : -jnp.inf, lambda : -jnp.log(self.scale))
        
        return y

    @partial(jax.jit, static_argnums=(0,))
    def cdf(self, x):
        """
        Evaluate the cumulative distribution function.
        
        Parameters
        ----------
        x : array-like
            Point or points at which to evaluate the distribution.
        
        Returns
        -------
        array-like
            Cumulative probability at ``x``.
        """
 
        y = (x - self.a) / (self.b - self.a)
                
        return y

    @partial(jax.jit, static_argnums=(0,))
    def ppf(self, y):
        """
        Evaluate the percent-point (quantile) function.
        
        Parameters
        ----------
        y : array-like
            Cumulative probability in the interval [0, 1].
        
        Returns
        -------
        array-like
            Quantile corresponding to ``y``.
        """

        y = jnp.array(y)

        x = y * (self.b - self.a) + self.a

        return x

class normal():
    """
    Normal distribution.
    
    Parameters
    ----------
    loc : float, optional
        Mean.
    scale : float, optional
        Standard deviation.
    """
    def __init__(self, loc=0, scale=1):
        """
        Normal distribution.
        
        Parameters
        ----------
        loc : float, optional
            Mean.
        scale : float, optional
            Standard deviation.
        """
        # Turn init args into attributes
        self.__dict__.update((k, v) for k, v in locals().items() if k not in ['self'])

        self.fac = -0.5 / self.scale**2      

        self.norm = 1 / (jnp.sqrt(2*jnp.pi) * self.scale)

        self.lognorm = jnp.log(self.norm)

        self._set_stdatt()


    def rv(self):
        """
        Draw one random variate.
        
        Returns
        -------
        float or int
            Random value drawn from the distribution.
        """

        u = np.random.uniform(0, 1)
        
        x = self.ppf(u)
        
        return x
    
    def _set_stdatt(self):
        """
        Set standard summary attributes such as the mean and median.
        """
        self.mean = self.loc
        self.median = self.ppf(0.5)
    
    @partial(jax.jit, static_argnums=(0,))
    def pdf(self, x, norm=True):
        """
        Evaluate the probability density function.
        
        Parameters
        ----------
        x : array-like
            Point or points at which to evaluate the distribution.
        norm : bool, optional
            Include the normalization constant when ``True``.
        
        Returns
        -------
        array-like
            Probability density at ``x``.
        """
        y = jnp.exp( self.fac * (x - self.loc)**2)

        Y = jax.lax.cond(norm,
                         lambda y: y * self.norm,
                         lambda y: y ,
                         y)

        return Y

    @partial(jax.jit, static_argnums=(0,))
    def logpdf(self, x, norm=True):
        """
        Evaluate the log-probability density function.
        
        Parameters
        ----------
        x : array-like
            Point or points at which to evaluate the distribution.
        norm : bool, optional
            Include the normalization constant when ``True``.
        
        Returns
        -------
        array-like
            Log-probability density at ``x``.
        """

        y = self.fac * (x - self.loc)**2

        Y = jax.lax.cond(norm,
                         lambda y: y + self.lognorm,
                         lambda y: y,
                         y)

        return Y

    @partial(jax.jit, static_argnums=(0,))
    def cdf(self, x):
        """
        Evaluate the cumulative distribution function.
        
        Parameters
        ----------
        x : array-like
            Point or points at which to evaluate the distribution.
        
        Returns
        -------
        array-like
            Cumulative probability at ``x``.
        """

        y = 0.5 * (1 + jsp.erf((x-self.loc)/(jnp.sqrt(2)*self.scale)))

        return y

    @partial(jax.jit, static_argnums=(0,))
    def ppf(self, y):
        """
        Evaluate the percent-point (quantile) function.
        
        Parameters
        ----------
        y : array-like
            Cumulative probability in the interval [0, 1].
        
        Returns
        -------
        array-like
            Quantile corresponding to ``y``.
        """

        x = self.loc + self.scale*jnp.sqrt(2)*jsp.erfinv(2*y-1)

        return x

class truncsine():
    """
    Sine distribution truncated to the interval [0, pi/2].
    """
    def __init__(self,):
        """
        Sine distribution truncated to the interval [0, pi/2].
        """

        self._set_stdatt()
        
    def rv(self):
        """
        Draw one random variate.
        
        Returns
        -------
        float or int
            Random value drawn from the distribution.
        """

        u = np.random.uniform(0, 1)
        
        x = self.ppf(u)
        
        return x

    def _set_stdatt(self):
        """
        Set standard summary attributes such as the mean and median.
        """
        self.mean = 1.0
        self.median = self.ppf(0.5)
 
    @partial(jax.jit, static_argnums=(0,))
    def pdf(self, x):
        """
        Evaluate the probability density or mass function.
        
        Parameters
        ----------
        x : array-like
            Point or points at which to evaluate the distribution.
        
        Returns
        -------
        array-like
            Probability density or mass at ``x``.
        """

        T = jax.lax.lt(x, 0.) | jax.lax.lt(jnp.pi/2, x)  
             
        y = jax.lax.cond(T, lambda : 0., lambda : jnp.sin(x))
            
        return y

    @partial(jax.jit, static_argnums=(0,))
    def logpdf(self, x):
        """
        Evaluate the log-probability density or mass function.
        
        Parameters
        ----------
        x : array-like
            Point or points at which to evaluate the distribution.
        
        Returns
        -------
        array-like
            Log-probability density or mass at ``x``.
        """

        T = jax.lax.lt(x, 0.) | jax.lax.lt(jnp.pi/2., x)  
             
        y = jax.lax.cond(T, lambda : -jnp.inf, lambda : jnp.log(jnp.sin(x)))

        return y

    @partial(jax.jit, static_argnums=(0,))
    def cdf(self, x):
        """
        Evaluate the cumulative distribution function.
        
        Parameters
        ----------
        x : array-like
            Point or points at which to evaluate the distribution.
        
        Returns
        -------
        array-like
            Cumulative probability at ``x``.
        """

        y = 1 + jnp.cos(x-jnp.pi) 
 
        return y

    @partial(jax.jit, static_argnums=(0,))
    def ppf(self, y):
        """
        Evaluate the percent-point (quantile) function.
        
        Parameters
        ----------
        y : array-like
            Cumulative probability in the interval [0, 1].
        
        Returns
        -------
        array-like
            Quantile corresponding to ``y``.
        """

        x = jnp.arccos(1-y)

        return x
    
class randint():
    """
    Discrete uniform distribution over consecutive integers.
    
    Parameters
    ----------
    low : int
        Inclusive lower bound.
    high : int
        Exclusive upper bound.
    """
    def __init__(self, low, high):
        """
        Discrete uniform distribution over consecutive integers.
        
        Parameters
        ----------
        low : int
            Inclusive lower bound.
        high : int
            Exclusive upper bound.
        """
        self.low = low
        
        self.high = high
        
        self.diff = self.high - self.low

        self.ints = jnp.arange(low, high, 1)

        self._set_stdatt()

    def rv(self):
        """
        Draw one random variate.
        
        Returns
        -------
        float or int
            Random value drawn from the distribution.
        """

        u = np.random.uniform(0, 1)
        
        x = self.ppf(u)
        
        return x

    def _set_stdatt(self):
        """
        Set standard summary attributes such as the mean and median.
        """
        x = jnp.linspace(self.ppf(1e-6), self.ppf(1-1e-6), 1000)

        self.mean = jnp.trapezoid(x * jnp.array([self.pdf(_x) for _x in x]), x)

        self.median = self.ppf(0.5)

    @partial(jax.jit, static_argnums=(0,))    
    def pdf(self, x):
        """
        Evaluate the probability density or mass function.
        
        Parameters
        ----------
        x : array-like
            Point or points at which to evaluate the distribution.
        
        Returns
        -------
        array-like
            Probability density or mass at ``x``.
        """
        T = jnp.sum(self.ints == x).astype(bool)
        
        y = jax.lax.cond(T, lambda : 1/self.diff, lambda : 0.)

        return y
    
    @partial(jax.jit, static_argnums=(0,))    
    def logpdf(self, x):
        """
        Evaluate the log-probability density or mass function.
        
        Parameters
        ----------
        x : array-like
            Point or points at which to evaluate the distribution.
        
        Returns
        -------
        array-like
            Log-probability density or mass at ``x``.
        """
        T = jnp.sum(self.ints == x).astype(bool)
        
        y = jax.lax.cond(T, lambda : -self.diff, lambda : -jnp.inf)

        return y

    @partial(jax.jit, static_argnums=(0,))    
    def cdf(self, x):
        """
        Evaluate the cumulative distribution function.
        
        Parameters
        ----------
        x : array-like
            Point or points at which to evaluate the distribution.
        
        Returns
        -------
        array-like
            Cumulative probability at ``x``.
        """
        
        k = jnp.floor(x)
        
        return (k - self.low + 1.) / self.diff
        
    @partial(jax.jit, static_argnums=(0,))
    def ppf(self, q):
        """
        Evaluate the percent-point (quantile) function.
        
        Parameters
        ----------
        q : array-like
            Cumulative probability in the interval [0, 1].
        
        Returns
        -------
        array-like
            Quantile corresponding to ``q``.
        """
        vals = jnp.ceil(q * self.diff + self.low) - 1
        
        vals1 = (vals - 1).clip(self.low, self.high)
        
        temp = self.cdf(vals1)
         
        return jnp.floor(jnp.where(temp >= q, vals1, vals))
