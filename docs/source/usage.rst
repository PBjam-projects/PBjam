User Guide
==========

PBjam is intended to be a user-friendly peakbagging tool. The most
straightforward interface is :class:`~pbjam.core.session`, which organises the
inputs and runs mode identification followed by peakbagging. Results are
available on each session's :class:`~pbjam.core.star` objects.

Inputs and units
----------------

The required observables are ``numax`` and ``dnu`` in microhertz and ``teff``
in kelvin. Each value is supplied with its uncertainty as a two-element
sequence. ``bp_rp`` is optional, but improves selection of the empirical prior
when available. A supplied spectrum must contain frequency in microhertz and
power density; a supplied time series normally uses time in days and flux in
parts per million.

Session
-------
The :class:`~pbjam.core.session` class can analyse one or more stars. It accepts
a supplied spectrum or time series, or can download light curves through
Lightkurve before computing the power-density spectrum. The `Session notebook
<Examples/example-session.ipynb>`_ demonstrates these forms.
 

Star
----
The :class:`~pbjam.core.star` class runs the same workflow for one supplied
power-density spectrum and is useful in custom scripts.

The :class:`~pbjam.core.star` class is meant for more detailed control of the inputs for each star. The `Star notebook <Examples/example-star.ipynb>`_ shows a simple example of this. 
    

Advanced
--------
It is not strictly necessary to use either the :class:`~pbjam.core.session` or :class:`~pbjam.core.star` classes. The `mode ID <Examples/example-modeID.ipynb>`_ and `peakbag <Examples/example-peakbag.ipynb>`_ notebooks show a lower-level walkthrough of the steps that PBjam goes through for peakbagging.

.. note:: 
    For additional useful examples see the `Examples <examples.html>`_ page.

Papers
------
The `first paper
<https://ui.adsabs.harvard.edu/abs/2021AJ....161...62N/abstract>`_ describes
the initial version of PBjam. The `second paper
<https://ui.adsabs.harvard.edu/abs/2023A%26A...676A.117N/abstract>`_ describes
the empirical prior construction used by current releases.
