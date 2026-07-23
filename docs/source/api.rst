API reference
=============

PBjam has a few high-level interfaces for complete analyses and lower-level
components for custom workflows.

Main interfaces
---------------

* :class:`pbjam.core.session` coordinates mode identification and peakbagging
  for one or more targets.
* :class:`pbjam.core.star` runs the same workflow for a single target when a
  power density spectrum is already available.
* :class:`pbjam.modeID.modeID` performs mode identification independently.
* :class:`pbjam.peakbagging.peakbag` performs detailed peakbagging from an
  input mode list.

Supporting modules provide model implementations, probability distributions,
dimensionality reduction, input/output handling, plotting utilities and
sampler integrations.

Module reference
----------------

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
