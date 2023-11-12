import numpy as np
import numba as nb
mapping = np.memmap('helcosthetaZ2.mm', dtype=np.int32, mode='r', shape=(100, 100))
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
	edges=np.array([-1.000,-0.980,-0.960,-0.940,-0.920,-0.900,-0.880,-0.860,-0.840,-0.820,-0.800,-0.780,-0.760,-0.740,-0.720,-0.700,-0.680,-0.660,-0.640,-0.620,-0.600,-0.580,-0.560,-0.540,-0.520,-0.500,-0.480,-0.460,-0.440,-0.420,-0.400,-0.380,-0.360,-0.340,-0.320,-0.300,-0.280,-0.260,-0.240,-0.220,-0.200,-0.180,-0.160,-0.140,-0.120,-0.100,-0.080,-0.060,-0.040,-0.020,0.000,0.020,0.040,0.060,0.080,0.100,0.120,0.140,0.160,0.180,0.200,0.220,0.240,0.260,0.280,0.300,0.320,0.340,0.360,0.380,0.400,0.420,0.440,0.460,0.480,0.500,0.520,0.540,0.560,0.580,0.600,0.620,0.640,0.660,0.680,0.700,0.720,0.740,0.760,0.780,0.800,0.820,0.840,0.860,0.880,0.900,0.920,0.940,0.960,0.980,1.000])
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