.. _Tests:

==============================
Test Cases
==============================

The :py:func:`test_2_case` tests in :ref:`Localtest` and :ref:`NonlocalTest`
showcase the fact that the brute force case and the MiLoMerge package produce the same
result. Selected tests are shown below, but one can also
look at the source code on `Git <https://github.com/MohitS704/MiLoMerge/tree/master/tests>`_
if desired.

.. _LocalTest:

Local Usage
++++++++++++

.. literalinclude :: ../../tests/test_merge_local.py
   :language: python

.. _NonlocalTest:

Nonlocal Usage
+++++++++++++++

.. literalinclude :: ../../tests/test_merge_nonlocal.py
   :language: python

Post-merging Bin Placement
++++++++++++++++++++++++++++

.. literalinclude :: ../../tests/test_placement.py
   :language: python
