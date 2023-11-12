import numpy as np
import numba as nb
mapping = np.memmap('Z1Mass.mm', dtype=np.int32, mode='r', shape=(100, 100))
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
	edges=np.array([40.005,40.803,41.601,42.399,43.197,43.995,44.793,45.591,46.389,47.187,47.985,48.783,49.581,50.379,51.177,51.975,52.773,53.571,54.369,55.167,55.965,56.763,57.561,58.359,59.157,59.956,60.754,61.552,62.350,63.148,63.946,64.744,65.542,66.340,67.138,67.936,68.734,69.532,70.330,71.128,71.926,72.724,73.522,74.320,75.118,75.916,76.714,77.512,78.310,79.108,79.906,80.704,81.503,82.301,83.099,83.897,84.695,85.493,86.291,87.089,87.887,88.685,89.483,90.281,91.079,91.877,92.675,93.473,94.271,95.069,95.867,96.665,97.463,98.261,99.059,99.857,100.655,101.453,102.251,103.050,103.848,104.646,105.444,106.242,107.040,107.838,108.636,109.434,110.232,111.030,111.828,112.626,113.424,114.222,115.020,115.818,116.616,117.414,118.212,119.010,119.808])
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