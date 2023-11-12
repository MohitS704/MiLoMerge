import numpy as np
import numba as nb
mapping = np.memmap('Z2Mass.mm', dtype=np.int32, mode='r', shape=(100, 100))
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
	edges=np.array([12.001,13.011,14.020,15.030,16.040,17.050,18.059,19.069,20.079,21.088,22.098,23.108,24.118,25.127,26.137,27.147,28.157,29.166,30.176,31.186,32.195,33.205,34.215,35.225,36.234,37.244,38.254,39.264,40.273,41.283,42.293,43.302,44.312,45.322,46.332,47.341,48.351,49.361,50.371,51.380,52.390,53.400,54.409,55.419,56.429,57.439,58.448,59.458,60.468,61.478,62.487,63.497,64.507,65.516,66.526,67.536,68.546,69.555,70.565,71.575,72.585,73.594,74.604,75.614,76.623,77.633,78.643,79.653,80.662,81.672,82.682,83.692,84.701,85.711,86.721,87.730,88.740,89.750,90.760,91.769,92.779,93.789,94.799,95.808,96.818,97.828,98.837,99.847,100.857,101.867,102.876,103.886,104.896,105.906,106.915,107.925,108.935,109.944,110.954,111.964,112.974])
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