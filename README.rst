
PBjam 2
============================

**Peakbagging made easy**

.. image:: https://img.shields.io/badge/GitHub-PBjam-green.svg
    :target: https://github.com/PBjam-projects/PBjam
.. image:: https://readthedocs.org/projects/pbjam/badge/?version=latest
    :target: https://pbjam.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status
.. image:: https://img.shields.io/badge/license-MIT-blue.svg?style=flat
    :target: https://github.com/PBjam-projects/PBjam/blob/master/LICENSE
.. image:: https://img.shields.io/github/issues-closed/PBjam-projects/PBjam.svg
    :target: https://github.com/PBjam-projects/PBjam/issues
.. image:: https://badge.fury.io/py/pbjam.svg
    :target: https://badge.fury.io/py/pbjam
.. image:: https://img.shields.io/badge/arXiv-2012.00580-B31B1B.svg
    :target: https://arxiv.org/abs/2012.00580

PBjam is a Bayesian toolkit for analysing the oscillation spectra of
solar-like oscillators. It first identifies modes in a power-density spectrum,
then fits the individual peaks to measure their frequencies and other
properties.

For main-sequence stars, mode identification fits the asymptotic
``l=0,1,2`` p-mode pattern jointly. For subgiants and red giants, PBjam first
fits the background and ``l=2,0`` pairs, then selects an evolutionary-stage
model for the ``l=1`` modes. Priors are informed by previous Kepler, K2 and
TESS observations distributed with the package.

Detailed peakbagging uses nested sampling or MCMC to fit Lorentzian profiles
with fewer constraints than the mode-identification models.

Installation
------------

PBjam requires Python 3.10 or newer::

    python -m pip install pbjam

Quick start
-----------

The high-level :class:`pbjam.session` interface accepts a power-density
spectrum as a two-row array. Frequencies must be in microhertz, and each
observable is supplied as a ``(value, uncertainty)`` pair::

    import numpy as np
    import pbjam

    obs = {
        "numax": (2204.0, 100.0),
        "dnu": (103.2, 0.54),
        "teff": (6140.0, 77.0),
        # Optional, but improves prior selection when available:
        "bp_rp": (0.700, 0.050),
    }

    run = pbjam.session(
        "my-star",
        obs,
        spectrum=np.vstack((frequency, power_density)),
    )
    run()

    star = run.stars[0]
    mode_id_result = star.modeID.result
    peakbag_result = star.peakbag.result

PBjam can also download light curves through Lightkurve or compute a spectrum
from a supplied time series. See the `documentation
<https://pbjam.readthedocs.io/>`_ and example notebooks for those workflows.

.. inclusion_marker0


Contributing
------------
If you want to raise an issue or contribute code to PBjam, see the
`contribution guidelines
<https://github.com/PBjam-projects/PBjam/blob/master/CONTRIBUTING.rst>`_.

Authors
-------
There are different ways to contribute to PBjam, the Scientific Influencers help guide the scientific aspects of PBjam, the Chaos Engineers try to break the code or simply report bugs, while the Main Contributors submit Pull Requests with somewhat bigger additions to the code or documentation. 

===================================================== ================================================ ====================================================
Main Contributors                                     Chaos Engineers                                  Scientific Influencers
===================================================== ================================================ ====================================================
`Lindsey Carboneau <https://github.com/lmcarboneau>`_ `Warrick Ball <https://github.com/warrickball>`_ `Othman Benomar <https://github.com/OthmanB>`_
`Guy Davies <https://github.com/grd349>`_             `Rafa Garcia <https://github.com/rgarcibus>`_    Bill Chaplin 
`Oliver Hall <https://github.com/ojhall94>`_          `Tanda Li <https://github.com/litanda>`_	       `Enrico Corsaro <https://github.com/EnricoCorsaro>`_
`Alex Lyttle <https://github.com/alexlyttle>`_        Angharad Weeks                                   `Patrick Gaulme <https://github.com/gaulme>`_  
`Martin Nielsen <https://github.com/nielsenmb>`_      Jens Rersted Larsen                              `Mikkel Lund <https://github.com/Miklnl>`_
`Joel Ong <https://github.com/darthoctopus>`_         |                                                Benoit Mosser 
`George Hookway <https://github.com/George-Hookway>`_ |                                                Andy Moya
|                                                     |                                                Ian Roxburgh
===================================================== ================================================ ====================================================


Acknowledgements
----------------
If you use PBjam in your work, please cite one of the PBjam papers
(`Paper I: Nielsen et al. 2021
<https://ui.adsabs.harvard.edu/abs/2021AJ....161...62N/abstract>`_,
`Paper II: Nielsen et al. 2023
<https://ui.adsabs.harvard.edu/abs/2023A%26A...676A.117N/abstract>`_) and,
if possible, link to the `GitHub repository
<https://github.com/PBjam-projects/PBjam>`_.

We encourage users to also cite the packages and publications that PBjam makes use of.  
