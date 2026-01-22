.. MiLoMerge documentation master file, created by
   sphinx-quickstart on Tue Jan  6 18:08:39 2026.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

MiLoMerge Documentation
=======================

.. Add your content using ``reStructuredText`` syntax. See the
.. `reStructuredText <https://www.sphinx-doc.org/en/master/usage/restructuredtext/index.html>`_
.. documentation for details.

.. toctree::
   :maxdepth: 3
   :caption: Contents:
   :hidden:

   MiLoMerge
   Tutorials
   Changelog

Overview
----------

This documentation is provided to assist in the usage of the MiLoMerge
package for "Minimal Loss Merging". The MiLoMerge package is designed to provide an interface
for merging bins together such that the loss in separability
between N distributions is minimized.

MiLoMerge is based off of `Maximizing Returns: Optimizing Experimental Observables at the LHC <https://arxiv.org/abs/2601.10822>`_,
which is currently available on ArXiv.

Relevant URLs
---------------

The homepage for MiLoMerge is located at 
the `JHUGen website <https://spin.pha.jhu.edu/>`_,
with the `GitHub <https://github.com/JHUGen/MiLoMerge>`_ available as well.
Both are linked in the footer of each page.

There is also a `PyPi page <https://pypi.org/project/MiLoMerge/>`_ available.

Installation
-------------

You can install MiLoMerge with pip, conda, or directly from the homepage.

.. code-block:: console

   pip install MiLoMerge

.. code-block:: console

   wget https://spin.pha.jhu.edu/Merger/<version>.tar.gz
   tar -xzf <version>.tar.gz
   cd MiLoMerge
   pip install .

Papers and Talks
------------------

Papers
++++++++


- (Jan. 15, 2026): `ArXiv:2601.10822 <Maximizing Returns: Optimizing Experimental Observables at the LHC>`_

Talks
++++++

- JHU HEP-Ex Seminar (Nov. 19, 2025): `Maximizing Impact: The Quest for Optimal Observables in Experiments <https://physics-astronomy.jhu.edu/event/hep-ex-amo-seminar-mohit-srivastav-michalis-panagiotou-jhu/>`_

Citing MiLoMerge
------------------

If using the MiLoMerge package, please cite the papers listed in the section above.



Frequently Asked Questions
------------------------------


Where should I go if I encounter a bug?
++++++++++++++++++++++++++++++++++++++++++

Please raise an issue on `Github <https://github.com/JHUGen/MiLoMerge/issues>`_ and/or contact
msrivas6@jh.edu with your bug.


Is there a paper about this stuff?
++++++++++++++++++++++++++++++++++++++++++

Yes! The initial paper, which includes
examples for how the package can be used,
as well as the basis for the metric used,
can be found on `ArXiv <https://arxiv.org/abs/2601.10822>`_.


Where has this package been used?
++++++++++++++++++++++++++++++++++++++++++

So far, the MiLoMerge package has
been used in some particle physics analyses on CMS
for the Higgs Boson and B-Physics.
