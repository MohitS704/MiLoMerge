.. _Examples:

==============================
Basic Examples
==============================

While not a formal tutorial, this page code snippets that 
one may find useful for certain use cases. 


Generating a Toy Dataset and Plotting Separability as a Function of Bin Number
--------------------------------------------------------------------------------

.. warning:: 
    This code snippet is best used for either local merging
    or for nonlocal merging where the amount of time taken
    is short. Any nonlocal merging done with large N should instead 
    use the example below
    that one utilizes the map_at functionality.

Since both the merger classes in :ref:`Mergers` 
save the number of bins from the run method, one can
keep calling the run method.

.. code-block:: python
    :name: perfPerBinExample

    import matplotlib.pyplot as plt
    import numpy as np
    import MiLoMerge

    signal = np.random.normal(loc=600, scale=20, size=5000) #some random signal centered at 600
    background = 175 + 1000*np.random.exponential(0.2, 100_000) #some random falling background
    binning = np.arange(175, 1001, 25)

    background_counts, _ = np.histogram(background, binning)
    background_counts = background_counts*1000/background_counts.sum()
    background_counts = background_counts.astype(int)

    signal_counts, _ = np.histogram(signal, binning)
    signal_counts = signal_counts*50/signal_counts.sum()
    signal_counts = signal_counts.astype(int)

    # here we are starting from 25 bins
    merger = MiLoMerge.MergerLocal(binning, signal_counts, background_counts)

    n_bins = np.arange(25, 2, -1, dtype=np.int32)
    scores = np.zeros_like(n_bins, dtype=np.float64)

    for i, n in enumerate(n_bins):
        # We can unpack the 2-d bin counts for our own use
        _, (h1_temp, h2_temp) = merger.run(n, return_counts=True)
        scores[i] = MiLoMerge.LOC_curve(h1_temp, h2_temp)[-1]

    plt.plot(n_bins, scores)


Generating a Toy Dataset and Plotting Separability as a Function of Bin Number (Large N)
-------------------------------------------------------------------------------------------

.. code-block:: python
    :name: perfPerBinExamplelargeN

    import matplotlib.pyplot as plt
    import numpy as np
    import MiLoMerge

    signal = np.random.normal(loc=600, scale=20, size=5000) #some random signal centered at 600
    background = 175 + 1000*np.random.exponential(0.2, 100_000) #some random falling background
    binning = np.arange(175, 1001, 25)

    background_counts, _ = np.histogram(background, binning)
    background_counts = background_counts*1000/background_counts.sum()
    background_counts = background_counts.astype(int)

    signal_counts, _ = np.histogram(signal, binning)
    signal_counts = signal_counts*50/signal_counts.sum()
    signal_counts = signal_counts.astype(int)

    # here we are starting from 25 bins
    n_bins = np.arange(25, 2, -1, dtype=np.int32)
    fpath = "./"
    fname = "test"
    fprefix = f"{fpath}{fname}"
    merger = MiLoMerge.MergerNonlocal(
        binning, 
        signal_counts, 
        background_counts, 
        map_at=n_bins, 
        file_path=fpath, 
        file_name=fname
    )

    scores = np.zeros_like(n_bins, dtype=np.float64)
    merger.run(2)

    #To ensure that all data lies within the observable phasespace
    #So that the placement function does not error out
    signal_mask = (signal >= binning[0]) & (signal <= binning[-1])
    background_mask = (background >= binning[0]) & (background <= binning[-1])

    for i, n in enumerate(n_bins):
        # We can unpack the 2-d bin counts for our own use
        h1_temp = np.bincount(MiLoMerge.place_array_nonlocal(n, signal[signal_mask], file_prefix=fprefix), minlength=n).astype(float)
        h2_temp = np.bincount(MiLoMerge.place_array_nonlocal(n, background[background_mask], file_prefix=fprefix), minlength=n).astype(float)

        scores[i] = MiLoMerge.LOC_curve(h1_temp, h2_temp)[-1]

    plt.plot(n_bins, scores)

Using the ROC and LOC curves for N-d distributions
---------------------------------------------------

The ROC and LOC curve functions documented in the :ref:`Metrics` 
section take 1-dimensional arrays. This does not mean that
they have to be 1-dimensional distributions! Since
the curves are ordered by ratio, one can simply
call :py:func:`numpy.ndarray.ravel` to unroll the histogram
into an equivalent 1-dimensional distribution.

.. warning:: 
    The metric functions specifically take in arrays of floats.
    If the histogram being used is an array of integers,
    simply call :py:func:`numpy.ndarray.astype` to convert
    it to a float.


.. code-block:: python
    :name: ROC_LOC_example

    import matplotlib.pyplot as plt
    import numpy as np
    import MiLoMerge

    # Imagine that h1 and h2 are n-dimensional NumPy histograms
    # of some arbitrary variable, which are currently integer arrays

    #Using the LOC curve
    TPR, FPR, score = MiLoMerge.LOC_curve(h1.ravel().astype(float), h2.ravel().astype(float))

    #Using the ROC curve
    TPR, FPR, score = MiLoMerge.ROC_curve(h1.ravel().astype(float), h2.ravel().astype(float))


