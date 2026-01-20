.. _Placement:

==============================
Placing Events After Merging
==============================

These are functions that are designed to place
unbinned data into the bins generated
by the :ref:`Mergers`. 

.. note::
    If the parameter `map_at` is given to a merger class
    from :ref:`Mergers`, then there is a mapping deposited into
    a tracker file and, if the merging was nonlocal,
    a bins file. The tracker file will be a `.hdf5` file,
    and the bins file a `.npy` file. 


.. autofunction:: MiLoMerge.place_event_nonlocal

.. autofunction:: MiLoMerge.place_array_nonlocal

.. autofunction:: MiLoMerge.place_local

