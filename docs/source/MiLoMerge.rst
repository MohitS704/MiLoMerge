.. _MiLoMerge:

=======================
Package Documentation
=======================

The Minimal-Loss Merge (MiLoMerge) package enables one to reduce the number of bins 
in such a way that the separability loss between N hypotheses
is minimized.

The general framework is that:

* Comparative metrics between distributions are available in the functions within the :ref:`Metrics` section.
* The bins to merge are returned in the two classes of mergers available in the :ref:`Mergers` section. 
* After generating the bins, the functions available in :ref:`Placement` are there to utilize the mapping generated from data to the new binning as an analogue for normal histogram functions. 

.. toctree::
   :maxdepth: 2
   :caption: Contents:
   :hidden:
   
   Mergers
   Metrics
   Placement


