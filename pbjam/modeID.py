"""Mode identification for solar-like oscillators.

This module provides :class:`modeID`, the PBjam interface for estimating the
locations and properties of oscillation modes before detailed peakbagging. The
mode-identification stage fits the background plus the ``l=2,0`` modes first and
then fits the ``l=1`` modes on the residual spectrum. Splitting the fit this way
keeps the more difficult mixed-mode calculation separate from the radial and
quadrupole-mode model.

The merged output from this stage is intended as the input to the detailed
peakbagging classes. Since :class:`modeID` inherits from
:class:`pbjam.plotting.plotting`, the resulting object can also use plotting
helpers such as ``echelle``, ``spectrum`` and ``corner`` after a model has been
run.
"""

import warnings, os, pickle
import jax.numpy as jnp
import numpy as np
from pbjam.l1models import Asyl1model, Mixl1model, RGBl1model
from pbjam.l20models import Asyl20model
from pbjam.plotting import plotting
from pbjam import IO
import pandas as pd

class modeID(plotting, ):  
    """Identify oscillation modes in a power density spectrum.

    The class stores the input spectrum, prepares a frequency window around
    ``numax``, runs the ``l=2,0`` and ``l=1`` mode-identification models, and
    merges their results into ``self.result``. Observational constraints are
    supplied through ``obs`` and must include at least ``numax``, ``dnu`` and
    ``teff`` as ``(value, uncertainty)`` pairs. Optional constraints such as
    ``bp_rp`` can be included when they are available.

    Parameters
    ----------
    f : array-like
        Frequency bins of the power density spectrum, in microhertz.
    s : array-like
        Power density values corresponding to ``f``.
    obs : dict
        Observational constraints. Required keys are ``'numax'``, ``'dnu'`` and
        ``'teff'``. Values are two-element sequences containing the measured
        value and its uncertainty, for example ``{'numax': (100.0, 5.0)}``.
    addPriors : dict, optional
        Additional or overriding priors passed to the underlying model classes.
        The expected structure depends on the selected model parameter.
    N_p : int, optional
        Number of radial orders to identify. Default is 7.
    freqLimits : list, optional
        Two-element ``[lower, upper]`` frequency window in microhertz. If
        ``None``, PBjam builds a window around ``numax`` using ``dnu`` and
        ``N_p``.
    priorPath : str, optional
        Path to the prior sample CSV file. If ``None``, PBjam uses the packaged
        default prior table.
    **kwargs
        Additional attributes stored on the instance for downstream plotting or
        model configuration.

    Attributes
    ----------
    f, s : jax.numpy.ndarray
        Input spectrum converted to JAX arrays.
    sel : ndarray of bool
        Mask selecting the spectrum inside ``freqLimits``.
    l20result, l1result : dict
        Parsed results from the individual model stages, created after
        ``runl20model`` and ``runl1model`` respectively.
    result : dict
        Merged result dictionary created after either ``runl20model`` or
        ``runl1model``. It contains top-level mode labels and nested ``summary``
        and ``samples`` dictionaries.
    """

    def __init__(self, f, s, obs, addPriors={}, N_p=7, freqLimits=None, priorPath=None, **kwargs):

        self.__dict__.update((k, v) for k, v in locals().items() if k not in ['self'])

        self.f = jnp.array(self.f)

        self.s = jnp.array(self.s)
      
        self.Nyquist = self.f[-1]

        # Set frequency range to compute the model on. Default is one radial order above/below the requested number.
        if self.freqLimits is None:
            self.freqLimits = [self.obs['numax'][0] - self.obs['dnu'][0]*(self.N_p//2+5), 
                               self.obs['numax'][0] + self.obs['dnu'][0]*(self.N_p//2+5),]
            
        self.sel = (np.array(self.freqLimits).min() < self.f) & (self.f < np.array(self.freqLimits).max())   

        if self.priorPath is None:
            self.priorPath = IO._getPriorPath()
 
    def runl20model(self, progress=True, dynamic=False, minSamples=5000, sampler_kwargs={}, logl_kwargs={},
                    loglikelihoodMultiplier=1.0, PCAsamples=50, PCAdims=6, selectivePrior=True,
                    selectivePriorN=10000, selectivePriorMin=100, selectivePriorSigma=1,
                    selectivePriorSeed=None, **kwargs):
        """Fit the background plus ``l=2,0`` modes.

        This is the first mode-identification stage. It fits the selected part
        of the spectrum with :class:`pbjam.l20models.Asyl20model`, stores the
        model instance on ``self.l20model``, stores raw posterior samples on
        ``self.l20Samples``, stores parsed results on ``self.l20result``, and
        initializes ``self.result`` with the merged ``l=2,0`` output.

        Parameters
        ----------
        progress : bool, optional
            Whether dynesty should show sampler progress. Default is ``True``.
        dynamic : bool, optional
            Whether to use dynamic nested sampling. Default is ``False``.
        minSamples : int, optional
            Minimum number of posterior samples requested from the sampler.
            Default is 5000.
        sampler_kwargs : dict, optional
            Extra keyword arguments passed to the dynesty sampler.
        logl_kwargs : dict, optional
            Extra keyword arguments passed to the log-likelihood function.
        loglikelihoodMultiplier : float, optional
            Multiplier applied to the final log-likelihood value passed to the
            nested sampler. Default is 1.0.
        PCAsamples : int, optional
            Number of prior samples used in the PCA-based prior construction.
            Default is 50.
        PCAdims : int, optional
            Number of PCA dimensions retained for the ``l=2,0`` prior model.
            Default is 6.
        selectivePrior : bool, optional
            Whether to refine the prior sample around the observed constraints.
            Default is ``True``.
        selectivePriorN : int, optional
            Number of candidate prior samples considered during selective prior
            refinement.
        selectivePriorMin : int, optional
            Minimum number of accepted selective-prior samples before warning.
        selectivePriorSigma : float, optional
            Width, in observational standard deviations, of the selective-prior
            acceptance region.
        selectivePriorSeed : int, optional
            Seed used by the selective-prior sampler. ``None`` leaves the draw
            unseeded.
        **kwargs
            Accepted for API compatibility; currently not used by this method.

        Returns
        -------
        result : dict
            Parsed ``l=2,0`` result dictionary.
        """

        f = self.f[self.sel]

        s = self.s[self.sel]

        selectiveKwargs = {'selectivePrior': selectivePrior,
                           'selectivePriorN': selectivePriorN,
                           'selectivePriorMin': selectivePriorMin,
                           'selectivePriorSigma': selectivePriorSigma,
                           'selectivePriorSeed': selectivePriorSeed}

        self.l20model = Asyl20model(f, s, 
                                    self.obs, 
                                    self.addPriors, 
                                    self.N_p, 
                                    PCAsamples, 
                                    PCAdims,
                                    priorPath=self.priorPath,
                                    **selectiveKwargs)

        self.l20model.likelihoodScale = float(loglikelihoodMultiplier)
        
        self.l20Samples = self.l20model.runSampler(progress=progress,
                                                   dynamic=dynamic,
                                                   minSamples=minSamples, 
                                                   logl_kwargs=logl_kwargs, 
                                                   sampler_kwargs=sampler_kwargs)

        l20samples_u = self.l20model.unpackSamples(self.l20Samples)

        self.l20result = self.l20model.parseSamples(l20samples_u)

        self.result = self.mergeResults(l20result=self.l20result)
 
        return self.l20result

    def runl1model(self, progress=True, dynamic=False, minSamples=5000, sampler_kwargs={}, logl_kwargs={},
                   model='auto', loglikelihoodMultiplier=1.0, PCAsamples=500, PCAdims=7,
                   selectivePrior=True, selectivePriorN=10000, selectivePriorMin=100,
                   selectivePriorSigma=1, selectivePriorSeed=None, **kwargs):
        """Fit the ``l=1`` modes on the ``l=2,0`` residual spectrum.

        This method should be called after :meth:`runl20model`, because it uses
        the median ``l=2,0`` model to divide the selected spectrum and fit the
        dipole modes in the residual. The parsed results are stored on
        ``self.l1result`` and merged with ``self.l20result`` into
        ``self.result``.

        Parameters
        ----------
        progress : bool, optional
            Whether dynesty should show sampler progress. Default is ``True``.
        dynamic : bool, optional
            Whether to use dynamic nested sampling. Default is ``False``.
        minSamples : int, optional
            Minimum number of posterior samples requested from the sampler.
            Default is 5000.
        sampler_kwargs : dict, optional
            Extra keyword arguments passed to the dynesty sampler.
        logl_kwargs : dict, optional
            Extra keyword arguments passed to the log-likelihood function.
        model : {'auto', 'ms', 'sg', 'rgb'}, optional
            Model family used for the dipole modes. ``'auto'`` selects from the
            observed ``dnu`` and ``teff`` values. ``'ms'`` uses the asymptotic
            main-sequence model, ``'sg'`` uses the mixed-mode subgiant model,
            and ``'rgb'`` uses the red-giant branch model.
        loglikelihoodMultiplier : float, optional
            Multiplier applied to the final log-likelihood value passed to the
            nested sampler. Default is 1.0.
        PCAsamples : int, optional
            Number of prior samples used in the PCA-based prior construction
            for models that use PCA priors. Default is 500.
        PCAdims : int, optional
            Number of PCA dimensions retained for the subgiant prior model.
            Default is 7.
        selectivePrior : bool, optional
            Whether to refine the prior sample around the observed constraints
            for model families that support selective priors.
        selectivePriorN : int, optional
            Number of candidate prior samples considered during selective prior
            refinement.
        selectivePriorMin : int, optional
            Minimum number of accepted selective-prior samples before warning.
        selectivePriorSigma : float, optional
            Width, in observational standard deviations, of the selective-prior
            acceptance region.
        selectivePriorSeed : int, optional
            Seed used by the selective-prior sampler. ``None`` leaves the draw
            unseeded.
        **kwargs
            Accepted for API compatibility; currently not used by this method.

        Returns
        -------
        result : dict
            Parsed ``l=1`` result dictionary.

        Raises
        ------
        ValueError
            If ``model`` is not one of ``'auto'``, ``'ms'``, ``'sg'`` or
            ``'rgb'``.
        """

        # Compute the l=2,0 model residual. 
        self.l20residual = self.s[self.sel] / self.l20model.getMedianModel()
      
        f = self.f[self.sel]

        s = self.l20residual

        selectiveKwargs = {'selectivePrior': selectivePrior,
                           'selectivePriorN': selectivePriorN,
                           'selectivePriorMin': selectivePriorMin,
                           'selectivePriorSigma': selectivePriorSigma,
                           'selectivePriorSeed': selectivePriorSeed}

        summary = {'n_p': self.l20result['enn'][self.l20result['ell']==0],
                   'nu0_p': self.l20result['summary']['freq'][0, self.l20result['ell']==0]}
 
        for key in ['numax', 'dnu', 'env_height', 'env_width', 'mode_width', 'teff', 'bp_rp']:
            summary[key] = self.l20result['summary'][key]

        if model.lower() =='auto':
            
            model = self.selectModel()

            print(f'Input Teff={self.obs["teff"][0]}K and dnu={self.obs["dnu"][0]}muHz suggests the appropriate l=1 model is: {model}')

        if model.lower() == 'ms':
            self.l1model = Asyl1model(f, s, 
                                      summary, 
                                      self.addPriors,
                                      PCAsamples, 
                                      priorPath=self.priorPath)

        elif model.lower() == 'sg':    
            self.l1model = Mixl1model(f, s, 
                                      summary, 
                                      self.addPriors,
                                      PCAsamples, 
                                      PCAdims,
                                      priorPath=self.priorPath,
                                      **selectiveKwargs)
            
        elif model.lower() == 'rgb':
            self.l1model = RGBl1model(f, s,  
                                      summary, 
                                      self.addPriors, 
                                      PCAsamples,
                                      rootiter=15,
                                      priorPath=self.priorPath,
                                      modelChoice='simple')
        else:
            raise ValueError(f'Model {model} is invalid. Please use either MS, SG or RGB.')

        self.l1model.likelihoodScale = float(loglikelihoodMultiplier)
         
        self.l1Samples  = self.l1model.runSampler(progress=progress,
                                                  dynamic=dynamic,
                                                  minSamples=minSamples, 
                                                  logl_kwargs=logl_kwargs, 
                                                  sampler_kwargs=sampler_kwargs)
        
        l1SamplesU = self.l1model.unpackSamples(self.l1Samples)

        self.l1result = self.l1model.parseSamples(l1SamplesU)

        self.result = self.mergeResults(l20result=self.l20result, l1result=self.l1result)

        return self.l1result

    def __call__(self, model='auto', progress=True, dynamic=False, sampler_kwargs={}, logl_kwargs={},
                 loglikelihoodMultiplier=1.0, **kwargs):
        """Run the full mode-identification workflow.

        Calling a :class:`modeID` instance runs :meth:`runl20model` followed by
        :meth:`runl1model`. The merged output is available as ``self.result``.

        Parameters
        ----------
        model : {'auto', 'ms', 'sg', 'rgb'}, optional
            Dipole-mode model selection passed to :meth:`runl1model`.
        progress : bool, optional
            Whether dynesty should show sampler progress.
        dynamic : bool, optional
            Whether both stages should use dynamic nested sampling.
        sampler_kwargs : dict, optional
            Extra keyword arguments passed to both samplers.
        logl_kwargs : dict, optional
            Extra keyword arguments passed to both likelihood functions.
        loglikelihoodMultiplier : float, optional
            Multiplier applied to the final model log-likelihood values
            passed to the nested sampler. Default is 1.0.
        **kwargs
            Accepted for API compatibility; currently not used by this method.
        """
         
        self.runl20model(progress, dynamic, sampler_kwargs=sampler_kwargs, logl_kwargs=logl_kwargs,
                         loglikelihoodMultiplier=loglikelihoodMultiplier)
        
        self.runl1model(progress, dynamic, model=model, sampler_kwargs=sampler_kwargs, logl_kwargs=logl_kwargs,
                        loglikelihoodMultiplier=loglikelihoodMultiplier)
 
    def mergeResults(self, l20result=None, l1result=None, N=5000):
        """Merge ``l=2,0`` and ``l=1`` model outputs.

        Attempts to include N samples from both models, but will use the lowest common
        value in case one of the models returns less than N samples.

        If both stages provide the same scalar parameter, such as ``numax`` or
        ``dnu``, the ``l=2,0`` value is retained because that stage generally
        provides the more stable estimate.

        Parameters
        ----------
        l20result : dict, optional
            Result dictionary from :meth:`runl20model`. If ``None``, the method
            uses ``self.l20result`` when available.
        l1result : dict, optional
            Result dictionary from :meth:`runl1model`. If ``None``, the method
            uses ``self.l1result`` when available.
        N : int, optional
            Maximum number of samples to include in the merged result. The
            returned sample count is the minimum of ``N`` and the available
            sample counts from the supplied model results.

        Returns
        -------
        R : dict
            Merged result dictionary with mode labels ``ell``, ``enn``, ``emm``
            and ``zeta``, plus nested ``summary`` and ``samples`` dictionaries.
        """

        # Initialize an empty result dictionary
        R = {'ell': np.array([]),
             'enn': np.array([]),
             'emm': np.array([]),
             'zeta': np.array([]),
             'summary': {'freq'  : np.array([]).reshape((2, 0)), 
                         'height': np.array([]).reshape((2, 0)), 
                         'width' : np.array([]).reshape((2, 0)),
                         'rotAsym': np.array([]).reshape((2, 0)),
                        },
             'samples': {'freq'  : np.array([]).reshape((N, 0)),
                         'height': np.array([]).reshape((N, 0)), 
                         'width' : np.array([]).reshape((N, 0)),
                         'rotAsym' : np.array([]).reshape((N, 0))
                        },
            }
        
        # Use self.l20result if l20result is not provided
        if l20result is None and hasattr(self, 'l20result'):
            l20result = self.l20result

        _N = np.append(N, l20result['samples']['freq'].shape[0])
         
        # Use self.l1result if l1result is not provided
        if l1result is None and hasattr(self, 'l1model'):
            l1result = self.l1result
        
        if l1result is not None:
            _N = np.append(_N, l1result['samples']['freq'].shape[0])
 
        # Use the minimum number of samples.
        N = np.min(_N)
        
        # This order overrides the numax, dnu and teff from l1result in the output since the l20result is more reliable. 
        resList = [l1result, l20result] 

        # Merge top level dictionary keys.
        for rootkey in ['ell', 'enn', 'emm', 'zeta']:
            for D in resList:
                if D is not None:
                    R[rootkey] = np.append(R[rootkey], D[rootkey])
        
        # Merge summary and samples level keys.
        for rootkey in ['summary', 'samples']:
            for D in resList: 
                if D is not None:
                    for subkey in list(D[rootkey].keys()):
                        if subkey not in ['freq', 'height', 'width', 'rotAsym']:
                            R[rootkey][subkey] = D[rootkey][subkey]

                    for subkey in ['freq', 'height', 'width', 'rotAsym']:
                        R[rootkey][subkey] = np.hstack((R[rootkey][subkey][:N, :], 
                                                        D[rootkey][subkey][:N, :]))

                else:
                    continue

        return R

    def storeResult(self, resultDict, path=None, ID=None):
        """Write mode-identification results to disk.

        The method writes two files named ``<ID>_modeIDresult.pkl`` and
        ``<ID>_modeIDresult.csv`` inside ``path``. The pickle contains the full
        result dictionary. The CSV contains scalar summary parameters from
        ``self.result['summary']`` and omits per-mode arrays such as frequency,
        height and width.

        Parameters
        ----------
        resultDict : dict
            Result dictionary to serialize to the pickle file, typically
            ``self.result``.
        path : str, optional
            Directory where output files should be written. If ``None``, the
            current working directory is used.
        ID : str, optional
            Identifier used in the output filenames. If ``None``, a random
            ``unknown_tgt_<integer>`` identifier is generated and a warning is
            emitted.
        """

        # If no path is specified use cwd.
        if path is None:
            path = os.getcwd()
        else:
            path = str(path)
        
        # Make the dir path if it doesn't exist
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(path)

        # Assign random number identifier to the save
        if ID is not None:
            _ID = str(ID) 
        else:   
            _ID = f'unknown_tgt_{np.random.randint(0, 1e10)}'

            warnings.warn(f'Output stored under {_ID}. You should probably specify a target ID.')
        
        basefilename = os.path.join(*[path, f'{_ID}_modeIDresult'])
        
        # Store everything in pickled dict
        pickle.dump(resultDict, open(basefilename+'.pkl', 'wb'))

        # Grab just model parameters and save
        _tmp = {key: self.result['summary'][key] for key in self.result['summary'].keys() if key not in ['freq', 'height', 'width']}

        df_data = [{'name': key, 'mean': value[0], 'error': value[1]} for key, value in _tmp.items()]

        # Create a DataFrame
        df = pd.DataFrame(df_data)
        
        df.to_csv(basefilename+'.csv', index=False)
 

    def selectModel(self, ):
        """Select the dipole-mode model from observed ``dnu`` and ``teff``.

        The method classifies the star as ``'ms'`` (main sequence), ``'sg'``
        (subgiant), or ``'rgb'`` (red giant branch) using linear thresholds in
        the large frequency separation and effective temperature plane.

        Returns
        -------
        model : str
            Selected model name: ``'ms'``, ``'sg'`` or ``'rgb'``.

        Notes
        -----
        The classification uses linear relations:

        - Main sequence: ``dnu > -0.016 * teff + 157``
        - Subgiant: ``dnu > -0.010 * teff + 74``
        - Red giant branch: used if neither condition is satisfied

        Examples
        --------
        >>> obj.obs = {'dnu': (100.0, 1.0), 'teff': (5800.0, 50.0)}
        >>> obj.selectModel()
        'sg'

        >>> obj.obs = {'dnu': (120.0, 1.0), 'teff': (6000.0, 50.0)}
        >>> obj.selectModel()
        'ms'

        >>> obj.obs = {'dnu': (50.0, 1.0), 'teff': (5000.0, 50.0)}
        >>> obj.selectModel()
        'rgb'
        """
            
        MSclassify = lambda Teff: -0.016 * Teff + 157

        SGclassify = lambda Teff: -0.010 * Teff + 74

        if self.obs['dnu'][0] > MSclassify(self.obs['teff'][0]):
            model = 'ms'

        elif self.obs['dnu'][0] > SGclassify(self.obs['teff'][0]):
            model = 'sg'

        else:
            model = 'rgb'

        return model
    
