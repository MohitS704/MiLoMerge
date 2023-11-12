import numpy as np
import numba as nb
mapping = np.memmap('Analytic_pairwise_INT.mm', dtype=np.int32, mode='r', shape=(3125, 3125))
def place_entry(N, observables, verbose=False):
	"""**place_entry** will place a set of observables into a given bin index from a previous optimization

	Parameters
	----------
	N : int
		The number of bins that you want total
	observables : numpy.ndarray
		A __1-d__ array of observables that you want to to place
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
	if len(observables) != 5:
		raise ValueError('Must input 5 observables for this histogram!')
	observables = np.array(observables, dtype=np.float64)
	edges=np.array([[40.000,54.000,68.000,82.000,96.000,110.000],
 [0.000,13.000,26.000,39.000,52.000,65.000],
 [-1.000,-0.600,-0.200,0.200,0.600,1.000],
 [-1.000,-0.600,-0.200,0.200,0.600,1.000],
 [-3.142,-1.885,-0.628,0.628,1.885,3.142]])
	nonzero_rolled = np.zeros(5, dtype=np.uint8)
	for i in nb.prange(5):
		nonzero_rolled[i] = np.searchsorted(edges[i], observables[i])
	nonzero_rolled -= 1
	if verbose:
		print('original index of:', nonzero_rolled)
		print('This places your point in the range:')
		for i in nb.prange(edges.shape[0]):
			print('[', edges[i][nonzero_rolled[i]], ',', edges[i][int(nonzero_rolled[i]+1)], ']')
	nonzero = (np.power(5, np.arange(4,-1,-1, np.int16))*nonzero_rolled).sum()
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
	observables : numpy.ndarray
		A __1-d__ array of observables that you want to to place
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