"""
The DR (for dimensionality reduction) module contains the methods used by PBjam to compute the principle 
components of the prior sample. This is used by PBjam to compute the prior density used during the sampling.
"""
import numpy as np
import pandas as pd
import jax, warnings
import jax.numpy as jnp
from functools import partial
jax.config.update('jax_enable_x64', True)

class PCA():
    """
    Perform local, optionally weighted principal-component analysis.
    
    The class selects a neighbourhood from a prior-sample table, standardizes the
    chosen variables, computes a weighted covariance matrix, and provides
    transformations between model space and a reduced latent space.
    
    Parameters
    ----------
    obs : dict
        Observational constraints used to select the local prior sample.
    varLabels : list of str
        Columns included in the PCA.
    fName : str or pathlib.Path
        CSV file containing the prior sample.
    nSamples : int
        Requested number of neighbouring prior samples.
    selectLabels : list of str
        Columns used to identify nearby samples.
    weights : array-like or callable, optional
        Sample weights or a callable that computes them.
    weightArgs : dict, optional
        Keyword arguments passed to a callable ``weights`` object.
    dropNansIn : {"all", "select"}, optional
        Controls whether rows containing missing PCA variables are removed, or only
        rows missing selection variables.
    """
    def __init__(self, obs, varLabels, fName, nSamples, selectLabels, weights=None, weightArgs={}, dropNansIn='all'):
        """
        Perform local, optionally weighted principal-component analysis.
        
        The class selects a neighbourhood from a prior-sample table, standardizes the
        chosen variables, computes a weighted covariance matrix, and provides
        transformations between model space and a reduced latent space.
        
        Parameters
        ----------
        obs : dict
            Observational constraints used to select the local prior sample.
        varLabels : list of str
            Columns included in the PCA.
        fName : str or pathlib.Path
            CSV file containing the prior sample.
        nSamples : int
            Requested number of neighbouring prior samples.
        selectLabels : list of str
            Columns used to identify nearby samples.
        weights : array-like or callable, optional
            Sample weights or a callable that computes them.
        weightArgs : dict, optional
            Keyword arguments passed to a callable ``weights`` object.
        dropNansIn : {"all", "select"}, optional
            Controls whether rows containing missing PCA variables are removed, or only
            rows missing selection variables.
        """
        self.__dict__.update((k, v) for k, v in locals().items() if k not in ['self'])
         
        if self.nSamples > 5000:
            warnings.warn('The requested PCA sample is very large, you may run in to memory issues.')
        
        self.dataF, self.dimsF, self.nSamples = self.getSample(self.fName, self.nSamples)
         
        self.setWeights(self.weights, self.weightArgs)

        self.mu  = jnp.average(self.dataF, axis=0, weights=self.weights)

        self.var = jnp.average((self.dataF - self.mu)**2, axis=0, weights=self.weights)

        self.std = jnp.sqrt(self.var)
 

    def setWeights(self, w, kwargs):
        """
        Set the sample weights.
        
        Parameters
        ----------
        w : array-like, callable, or None
            Explicit weights, a callable that computes weights, or ``None`` for uniform
            weights.
        kwargs : dict
            Keyword arguments supplied to a callable ``w``.
        """
         
        if w is None:
            self.weights = jnp.ones(self.nSamples)

        elif callable(w):
            self.weights = w(self, **kwargs)

        else:
            self.weights = w


    def readPriorData(self, fName, labels):
        """
        Read and preprocess the prior-sample table.
        
        Parameters
        ----------
        fName : str or pathlib.Path
            CSV file containing the prior sample.
        labels : list of str
            Columns to read.
        
        Returns
        -------
        pandas.DataFrame
            Selected columns after replacing infinities and dropping required missing
            values.
        """

        self.priorData = pd.read_csv(fName, usecols=labels)
         
        self.priorData.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        if self.dropNansIn == 'all':
            self.dropNanLabels = self.varLabels + self.selectLabels
        else:
            self.dropNanLabels = self.selectLabels

        self.priorData.dropna(axis=0, how="any", inplace=True, subset=self.dropNanLabels)
         
        self.priorData.reset_index(drop=True, inplace=True)
         
        return self.priorData

    def getSample(self, fName, nSamples):
        """
        Load and select a local prior sample.
        
        Parameters
        ----------
        fName : str or pathlib.Path
            CSV file containing the prior sample.
        nSamples : int
            Requested number of neighbouring samples.
        
        Returns
        -------
        data : jax.Array
            Selected PCA variables.
        n_dimensions : int
            Number of PCA variables.
        n_samples : int
            Number of selected rows after filtering.
        """
         
        readlabels = self.varLabels + [key for key in self.selectLabels if key not in self.varLabels]
         
        priorData = self.readPriorData(fName, readlabels)
        
        self.selectedSubset = self.findNearest(priorData, nSamples)
        
        _nSamples, _ndim = self.selectedSubset[self.varLabels].shape

        # TODO put in something that tries a bit harder if no nearby targets are found
        
        return jnp.array(self.selectedSubset[self.varLabels].to_numpy()), _ndim, _nSamples

    def findNearest(self, fullPriorData, N):
        """
        Select the nearest prior samples in observable space.
        
        Parameters
        ----------
        fullPriorData : pandas.DataFrame
            Complete, preprocessed prior table.
        N : int
            Maximum number of neighbours to return.
        
        Returns
        -------
        pandas.DataFrame
            Up to ``N`` nearest viable samples.
        """

        limits = {'numax': 0.2,
                  'dnu': 0.2,
                  'teff': 9999,
                  'bp_rp': 9999}
 
        # Tgts within limits
        idx = np.prod(np.array([abs(fullPriorData[key].values - self.obs[key][0]) < limits[key] for key in self.selectLabels], dtype=bool), axis=0).astype(bool)
         
        priorData = fullPriorData.loc[idx, :].reset_index(drop=True)
         
        priorSampleMean = np.mean(priorData[self.selectLabels].values, axis=0)

        priorSampleStd = np.std(priorData[self.selectLabels].values, axis=0)
 
        normedObs = {key: (self.obs[key][0] - priorSampleMean[i]) / priorSampleStd[i] for i, key in enumerate(self.selectLabels)}
 
        normedPrior = {key: (priorData[key].values - priorSampleMean[i]) / priorSampleStd[i] for i, key in enumerate(self.selectLabels)}

        delta_i = np.array([normedPrior[key] - normedObs[key] for i, key in enumerate(self.selectLabels)])
            
        euclDist = np.sqrt(np.average(delta_i**2, axis=0))
         
        sortidx = np.argsort(euclDist)
        
        selectedSubset = priorData.loc[sortidx, :][:N]

        self.nanFraction = np.max(np.sum(np.isnan(selectedSubset).values, axis=0)/len(selectedSubset))

        self.viableFraction = 1 - self.nanFraction

        selectedSubset.dropna(axis=0, how="any", inplace=True)
         
        return selectedSubset.reset_index(drop=True)

    @partial(jax.jit, static_argnums=(0,))
    def scale(self, data):
        """
        Standardize model-space samples.
        
        Parameters
        ----------
        data : array-like
            Samples in model space.
        
        Returns
        -------
        jax.Array
            Samples shifted by the weighted mean and divided by the weighted standard
            deviation.
        """

        scaledData = (data - self.mu) / self.std

        return scaledData

    @partial(jax.jit, static_argnums=(0,))
    def inverse_scale(self, scaledData):
        """
        Undo model-space standardization.
        
        Parameters
        ----------
        scaledData : array-like
            Standardized samples.
        
        Returns
        -------
        jax.Array
            Samples restored to their original scale.
        """

        unscaled = scaledData * self.std + self.mu

        return unscaled

    @partial(jax.jit, static_argnums=(0,))
    def transform(self, X):
        """
        Project model-space samples into latent PCA coordinates.
        
        Parameters
        ----------
        X : array-like
            Samples in model space.
        
        Returns
        -------
        jax.Array
            Coordinates in the retained latent dimensions.
        """
        _X = self.scale(X)
         
        Y = self.eigvectors[:, self.sortidx].T.dot(_X.T)

        return Y.T.real

    @partial(jax.jit, static_argnums=(0,))
    def inverse_transform(self, Y):
        """
        Project latent PCA coordinates back into model space.
        
        Parameters
        ----------
        Y : array-like
            Samples in latent space.
        
        Returns
        -------
        jax.Array
            Reconstructed model-space samples.
        """

        _X = jnp.dot(Y, self.eigvectors[:, self.sortidx].T)

        return self.inverse_scale(_X).real

    def fit_weightedPCA(self, dim):
        """ Compute PCAs and transform
        
        Wrapper for various functions to compute the covariance matrix of a 
        sample, a few metrics for the PCAs and then project the sample of model
        parameters into the the latent space.

        Parameters
        ----------
        dim : int
            Maximum number of principal components to retain.
        
        Returns
        -------
        None
        """

        
        self.dimsR = min([dim, len(self.varLabels)])

        _X = self.scale(self.dataF)
         
        self.covariance = self.covarianceMatrix(_X)
        
        self.eigvals, self.eigvectors = jnp.linalg.eigh(self.covariance)

        self.sortidx = sorted(range(len(self.eigvals)), key=lambda i: self.eigvals[i], reverse=True)[:self.dimsR]

        self.explained_variance_ratio = sorted(self.eigvals / jnp.sum(self.eigvals), reverse=True)

        self.erank = jnp.exp(-jnp.sum(self.explained_variance_ratio * np.log(self.explained_variance_ratio))).real

        self.dataR = self.transform(self.dataF)

    def setLatentNormalPrior(self, latentSample=None):
        """
        Set independent normal priors on the latent coordinates.
        
        Parameters
        ----------
        latentSample : array-like, optional
            Sample used to estimate each latent mean and standard deviation. The fitted
            PCA sample is used by default.
        """

        from pbjam import distributions as dist

        if latentSample is None:
            latentSample = self.dataR

        latentSample = np.asarray(latentSample)
        loc = np.mean(latentSample, axis=0)
        scale = np.std(latentSample, axis=0)
        scale = np.where(np.isfinite(scale) & (scale > 0), scale, 1.0)

        self.latentPriorLoc = jnp.array(loc)
        self.latentPriorScale = jnp.array(scale)
        self.latentPriors = [dist.normal(loc=self.latentPriorLoc[i],
                                         scale=self.latentPriorScale[i])
                             for i in range(self.dimsR)]
        self.ppf = [prior.ppf for prior in self.latentPriors]
        self.pdf = [prior.pdf for prior in self.latentPriors]
        self.logpdf = [prior.logpdf for prior in self.latentPriors]
        self.cdf = [prior.cdf for prior in self.latentPriors]

    def covarianceMatrix(self, _X):
        """
        Compute the unbiased weighted covariance matrix.
        
        Parameters
        ----------
        _X : array-like
            Standardized samples with rows representing observations.
        
        Returns
        -------
        jax.Array
            Weighted covariance matrix.
        """

        W = jnp.diag(self.weights)
        
        C = _X.T@W@_X * jnp.sum(self.weights) / (jnp.sum(self.weights)**2 - jnp.sum(self.weights**2))

        return C

    def refinePriorByObservables(self, N=10000, minAccepted=100, sigmaInflation=1,
                                 rng=None):
        """
        Refine latent normal priors using observational consistency.
        
        Parameters
        ----------
        N : int, optional
            Number of latent samples to draw.
        minAccepted : int, optional
            Minimum number of observationally consistent samples required.
        sigmaInflation : float, optional
            Multiplicative factor applied to observational uncertainties.
        rng : numpy.random.Generator, optional
            Random-number generator.
        
        Returns
        -------
        dict
            Diagnostic information describing the accepted sample and refined priors.
        """

        if not hasattr(self, 'ppf'):
            raise AttributeError('Set the initial latent prior before refining the prior.')

        if not hasattr(self, 'dimsR') or self.dimsR == 0:
            return None

        obsLabels = [key for key in self.selectLabels
                     if (key in self.varLabels) and (key in self.obs)]

        obsLabels = [key for key in obsLabels if self.obs[key][1] > 0]

        if len(obsLabels) == 0:
            warnings.warn('Selective prior refinement skipped: no observed selection labels are in the PCA variables.',
                          stacklevel=2)
            return None

        if rng is None:
            rng = np.random.default_rng()
        elif not hasattr(rng, 'uniform'):
            rng = np.random.default_rng(rng)

        obsIdx = np.array([self.varLabels.index(key) for key in obsLabels])
        obsVals = np.array([self.obs[key][0] for key in obsLabels])
        baseObsErrs = np.array([self.obs[key][1] for key in obsLabels])

        nDraws = max(1, int(N))
        currentSigmaInflation = sigmaInflation
        while True:
            latentDraws = rng.normal(loc=np.asarray(self.latentPriorLoc),
                                     scale=np.asarray(self.latentPriorScale),
                                     size=(nDraws, self.dimsR))

            physicalDraws = np.asarray(self.inverse_transform(jnp.array(latentDraws)))

            finite = np.all(np.isfinite(physicalDraws), axis=1)
            obsErrs = baseObsErrs * currentSigmaInflation
            delta = (physicalDraws[:, obsIdx] - obsVals) / obsErrs
            logLike = -0.5 * np.sum(delta**2, axis=1)
            finite &= np.isfinite(logLike)

            if not np.any(finite):
                raise ValueError('Selective prior refinement found no finite prior draws.')

            acceptProb = np.zeros_like(logLike)
            acceptProb[finite] = np.exp(logLike[finite] - np.max(logLike[finite]))
            accepted = rng.uniform(0, 1, size=len(logLike)) < acceptProb

            M = int(np.sum(accepted))
            self.selectivePriorInfo = {'draws': nDraws,
                                       'accepted': M,
                                       'minAccepted': int(minAccepted),
                                       'sigmaInflation': currentSigmaInflation,
                                       'labels': obsLabels}

            if M >= minAccepted:
                break

            nextNDraws = 2 * nDraws
            nextSigmaInflation = 2 * currentSigmaInflation
            warnings.warn(f'Selective prior refinement accepted {M} points, fewer than minAccepted={minAccepted}. '
                          f'Retrying with {nextNDraws} draws and sigmaInflation={nextSigmaInflation}.',
                          stacklevel=2)
            nDraws = nextNDraws
            currentSigmaInflation = nextSigmaInflation

        self.selectivePhysicalSample = physicalDraws[accepted, :]
        self.selectiveSubset = pd.DataFrame(self.selectivePhysicalSample, columns=self.varLabels)

        selectiveLatentSample = np.asarray(self.transform(jnp.array(self.selectivePhysicalSample)))
        self.selectiveLatentSample = selectiveLatentSample

        self.dataR = jnp.array(selectiveLatentSample)
        self.setLatentNormalPrior(selectiveLatentSample)

        return selectiveLatentSample
