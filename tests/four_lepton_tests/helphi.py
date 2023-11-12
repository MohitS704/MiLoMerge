import numpy as np
import numba as nb
mapping = np.memmap('helphi.mm', dtype=np.int32, mode='r', shape=(100, 100))
def place_entry(N, observables, verbose=False):
	"""**place_entry** will place a set of observables into a given bin index from a previous optimization

	Parameters
	----------
	N : int
		The number of bins that you want total
	observables : float
		An observable quantity that you are placing
	verbose : bool, optional
		Set to True if you want to print verbose options, by default False

	Returns
	-------
	int
		The bin index in range [0, N) that the observables will be placed at

	Raises
	------
	ValueError
		len(observables) must be equal to the shape of bin edges
	ValueError
		You asked for fewer bins than is possible!
	"""
	if len(observables) != 1:
		raise ValueError('Must input 1 observables for this histogram!')
	observables = np.array(observables, dtype=np.float64)
	edges=np.array([-3.142,-3.079,-3.016,-2.953,-2.890,-2.827,-2.765,-2.702,-2.639,-2.576,-2.513,-2.450,-2.388,-2.325,-2.262,-2.199,-2.136,-2.073,-2.011,-1.948,-1.885,-1.822,-1.759,-1.696,-1.634,-1.571,-1.508,-1.445,-1.382,-1.319,-1.257,-1.194,-1.131,-1.068,-1.005,-0.942,-0.880,-0.817,-0.754,-0.691,-0.628,-0.565,-0.503,-0.440,-0.377,-0.314,-0.251,-0.188,-0.126,-0.063,0.000,0.063,0.126,0.188,0.251,0.314,0.377,0.440,0.503,0.565,0.628,0.691,0.754,0.817,0.880,0.942,1.005,1.068,1.131,1.194,1.257,1.319,1.382,1.445,1.508,1.571,1.634,1.696,1.759,1.822,1.885,1.948,2.011,2.073,2.136,2.199,2.262,2.325,2.388,2.450,2.513,2.576,2.639,2.702,2.765,2.827,2.890,2.953,3.016,3.079,3.142])
	nonzero_rolled = np.searchsorted(edges, observables) - 1
	if verbose:
		print('original index of:', nonzero_rolled)
		print('This places your point in the range:')
		for i in nb.prange(edges.shape[0]):
			print('[', edges[i][nonzero_rolled[i]], ',', edges[i][int(nonzero_rolled[i]+1)], ']')
	nonzero = nonzero_rolled
	mapped_val = mapping[N-1][nonzero]
	if mapped_val < 0:
		raise ValueError('Cannot have fewer than 0 bins!')
	return nonzero_rolled, mapped_val
def help():
	text = """**place_entry** will place a set of observables into a given bin index from a previous optimization

	Parameters
	----------
	N : int
		The number of bins that you want total
	observables : float
		An observable quantity that you are placing
	verbose : bool, optional
		Set to True if you want to print verbose options, by default False

	Returns
	-------
	int
		The bin index in range [0, N) that the observables will be placed at

	Raises
	------
	ValueError
		len(observables) must be equal to the shape of bin edges
	ValueError
		You asked for fewer bins than is possible!
	"""
	print(text)
	exit()