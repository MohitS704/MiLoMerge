import warnings
import os
import numpy as np
import h5py

def load_file_local(fname, key):
    """Simple function to load and return an h5py file

    Parameters
    ----------
    fname : str
        The filename to load
    key : str
        The key to load for the h5py file

    Returns
    -------
    numpy.ndarray
        Returns a copy of the array that is being accessed through h5py
    """
    f = h5py.File(fname, 'r')
    return f[key][:]

def load_file_nonlocal(fname_tracker, fname_bins, key):
    """Simple function to load and return an h5py file
    alongside its corresponding "physical bins" file

    Parameters
    ----------
    fname_tracker : str
        The filename of the tracker (h5py file)
    fname_bins : str
        The filename of the physical bins file
    key : str
        The key to load for the h5py file

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Returns a copy of the array that is being accessed through h5py
        alongside a copy of the physical bins
    """
    f = h5py.File(fname_tracker, 'r')
    bin_mapping = f[key][:]

    physical_bins = np.load(fname_bins, allow_pickle=True)

    return bin_mapping, physical_bins

def place_event_nonlocal(N, *observable, file_prefix="", verbose=False):
    """Places a given event within the appropriate bin using a binmap and the 
    original physical bins

    Parameters
    ----------
    N : _type_
        _description_
    file_prefix : str, optional
        _description_, by default ""
    verbose : bool, optional
        _description_, by default False

    Returns
    -------
    _type_
        _description_

    Raises
    ------
    ValueError
        _description_
    ValueError
        _description_
    ValueError
        _description_
    ValueError
        _description_
    """

    fname_tracker = f".{file_prefix}_tracker.hdf5"
    fname_bins = f".{file_prefix}_physical_bins.npy"
    bin_mapping, physical_bins = load_file_nonlocal(fname_tracker, fname_bins, str(N))

    observable = np.array(observable)

    # print(physical_bins.shape, observable)
    subarray_lengths = [len(b) for b in physical_bins]
    if len(physical_bins.shape) > 1 and len(observable) > 1:
        n_observables = physical_bins.shape[0]
        n_physical_bins = physical_bins.shape[1]

        nonzero_rolled = np.zeros(n_observables, dtype=np.uint64)
        for i in np.arange(n_observables):
            nonzero_rolled[i] = np.searchsorted(physical_bins[i], observable[i]) - 1

        if np.any(nonzero_rolled < 0) or np.any(nonzero_rolled > n_physical_bins - 1):
            raise ValueError(f"observables are outside of the provided phase space!")

        if verbose:
            print('original index of:', nonzero_rolled)
            print('This places your point in the range:')
            for i in range(physical_bins.shape[0]):
                print('[', physical_bins[i][nonzero_rolled[i]], ',', physical_bins[i][int(nonzero_rolled[i]+1)], ']')

        unrolled_index = (
            np.power(
                n_physical_bins - 1, 
                np.arange(n_observables - 1,-1,-1, np.int16)
                )*nonzero_rolled
            ).sum()

    elif any([subarray_lengths[0] != b for b in subarray_lengths]) and len(subarray_lengths) == len(observable):
        n_observables = len(observable)
        n_physical_bins = np.array(subarray_lengths) #this is now per dimension

        nonzero_rolled = np.zeros(n_observables, dtype=np.uint64)
        for i in np.arange(n_observables):
            nonzero_rolled[i] = np.searchsorted(physical_bins[i], observable[i]) - 1
            if nonzero_rolled[i] < 0 or nonzero_rolled[i] >= n_physical_bins[i]:
                raise ValueError(f"Observable of index {i} is outside of the provided phase space!")
            
            if verbose:
                print('original index of:', nonzero_rolled[i])
                print('This places your point in the range:')
                print('[', physical_bins[i][nonzero_rolled[i]], ',', physical_bins[i][int(nonzero_rolled[i]+1)], ']')
        
        unrolled_index = 0
        multiplier = 1
        for i in reversed(range(len(n_physical_bins))):
            unrolled_index += nonzero_rolled[i]*multiplier
            if i > 0:
                multiplier *= subarray_lengths[i] - 1

    elif len(physical_bins.shape) > 1 or len(observable) > 1:
        raise ValueError(
            f"Shapes are incompatible\n"
            f" shape of bins is {len(physical_bins.shape)} and"
            f" length of observables is {len(observable)}"
        )
    else:
        if len(observable) > 1:
            raise ValueError

        unrolled_index = np.searchsorted(physical_bins, observable) - 1

    # print(bin_mapping)
    try:
        mapped_index = bin_mapping[unrolled_index]
    except IndexError as e:
        raise ValueError(f"{observable} is outside of the provided phase space!") from e

    return mapped_index

def place_array_nonlocal(N, observables, file_prefix="", verbose=False):

    fname_tracker = f".{file_prefix}_tracker.hdf5"
    fname_bins = f".{file_prefix}_physical_bins.npy"
    bin_mapping, physical_bins = load_file_nonlocal(fname_tracker, fname_bins, str(N))
    observables_stacked = np.array(observables)

    subarray_lengths = np.array([len(b) for b in physical_bins])
    if physical_bins.ndim > 1:
        if len(observables_stacked[0]) != len(physical_bins):
            raise ValueError(
                f"Number of observables {len(observables_stacked[0])} != Number of bin dimensions {len(physical_bins)}"
                )
        n_physical_bins = physical_bins.shape[1]
        
        n_datapoints, n_observables = observables_stacked.shape
        nonzero_rolled = np.zeros((n_datapoints, n_observables), dtype=np.uint64)
        for i in range(n_observables):
            nonzero_rolled[:, i] = np.searchsorted(physical_bins[i], observables_stacked[:, i]) - 1
        if verbose:
            print("Original indices")
            print(nonzero_rolled)

        unrolled_index = (np.power(n_physical_bins - 1, np.arange(n_observables - 1,-1,-1, np.int16))*nonzero_rolled).sum(axis=1)
        unrolled_index = unrolled_index.astype(int)
    elif np.any(subarray_lengths != subarray_lengths[0]):
        if len(observables_stacked[0]) != len(physical_bins):
            raise ValueError(
                f"Number of observables {len(observables_stacked[0])} != Number of bin dimensions {len(physical_bins)}"
            )
        n_physical_bins = subarray_lengths
        n_datapoints, n_observables = observables_stacked.shape
        nonzero_rolled = np.zeros((n_datapoints, n_observables), dtype=np.uint64)
        for i in range(n_observables):
            nonzero_rolled[:,i] = np.searchsorted(physical_bins[i], observables_stacked[:, i]) - 1
        if verbose:
            print("Original indices")
            print(nonzero_rolled)
        
        unrolled_index = np.zeros(n_datapoints, dtype=np.uint64)
        multiplier = 1
        for i in reversed(range(len(n_physical_bins))):
            unrolled_index += nonzero_rolled[:,i]*multiplier
            if i > 0:
                multiplier *= subarray_lengths[i] - 1
    else:
        if observables_stacked.ndim != physical_bins.ndim:
            raise ValueError(f"Number of observables {observables_stacked.ndim} != Number of bin dimensions {physical_bins.ndim}")
        n_physical_bins = len(physical_bins)
        nonzero_rolled = np.searchsorted(physical_bins, observables_stacked) - 1
        unrolled_index = nonzero_rolled

    failed_events = (unrolled_index > len(bin_mapping))
    if np.any(failed_events):
        print("The following events have indices that are too large:")
        for i, j in zip(nonzero_rolled[failed_events], unrolled_index[failed_events]):
            print(f"{i} = {j}")
        raise KeyError("Please check your phasespace to ensure it is within your original binning!")
    
    return bin_mapping[unrolled_index].ravel()

def place_local(N, observable_array, file_prefix="", verbose=False):
    fname = f".{file_prefix}_tracker.hdf5"
    bin_mapping = load_file_local(fname, str(N))

    if verbose:
        print(f"Using file {os.path.abspath(fname)}")
        print(np.array(bin_mapping))

    placements = np.searchsorted(bin_mapping, observable_array) - 1

    if np.any((placements < 0) or (placements == len(bin_mapping)) ):
        warnings.warn("Some items placed out of bounds! Consider having an overflow or underflow bin!")

    return placements