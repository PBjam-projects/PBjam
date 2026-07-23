API reference
=============

PBjam has three main public entry points:

* :class:`pbjam.core.session` coordinates analyses for one or more targets.
* :class:`pbjam.core.star` coordinates the analysis of a single target.
* :class:`pbjam.modeID.modeID` and the classes in
  :mod:`pbjam.peakbagging` provide lower-level interfaces for custom pipelines.

The remaining modules contain model implementations, probability distributions,
input/output utilities, plotting helpers, and samplers used by those interfaces.

Modules
-------

.. toctree::
   :maxdepth: 1

   background
   core
   distributions
   DR
   IO
   jar
   l1models
   l20models
   MSmodels
   modeID
   peakbagging
   plotting
   samplers
   validation
