import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
import matplotlib as mpl
import histogram_helpers as h
import tqdm
import jit_methods as j
import warnings
import sys
import numba as nb

# plt.style.use(hep.style.ROOT)
# mpl.rcParams['axes.labelsize'] = 40
# mpl.rcParams['xaxis.labellocation'] = 'center'

class Grim_Brunelle_merger(object):#Professor Nathan Brunelle!
    #https://engineering.virginia.edu/faculty/nathan-brunelle
    def __init__(self, bins, *counts, stats_check=True, subtraction_metric=True, weights=None, SM_version=False) -> None:
        """This class is the base object for bin merging. It merges bins locally (by outting adjacent bins together).
        Please only input 1 dimensional histograms! Should you need to operate upon a multidimensional histogram - please unroll them first.

        Parameters
        ----------
        bins : numpy.ndarray
            These are your bin edges
        counts : numpy.ndarray
            These are a set of bin counts put in as args. Place an unlimited number of them - each one corresponds to a different hypothesis. 
            Both bins and counts can be created using numpy.histogram()
        stats_check : bool, optional
            If you want to insert a statistics check enable this option, by default True
        subtraction_metric : bool, optional
            If you want to use the metric that does subtraction. If false, it uses version of the metric that uses division, by default True
        weights : numpy.ndarray, optional
            A list of weights of the same length as the number of counts put in for each hypothesis. If None all weights are 1, by default None
        SM_version : bool, optional
            Whether you are using the "Standard Model" version where samples are compared to the first set of inputted counts instead of to everything else

        Raises
        ------
        ValueError
            If your bin counts are not all the same length, raise an error
        ValueError
            If len(counts) != len(bins) - 1 then raise an error
        ValueError
            If len(weights) != len(counts) then raise an error
        """
        
        it = iter(counts)
        the_len = len(next(it))
        if not all(len(l) == the_len for l in it):
            errortext = [str(len(count)) for count in counts]
            errortext = " ".join(errortext)
            errortext = 'Not all counts have same length! Lengths are: ' + errortext
            errortext = h.print_msg_box(errortext, title="ERROR")
            raise ValueError('\n'+errortext)
        
        if len(counts[0]) != len(bins) - 1:
            errortext = "Invalid lengths! {:.0f} != {:.0f} - 1".format(len(counts[0]), len(bins))
            errortext = h.print_msg_box(errortext, title="ERROR")
            raise ValueError('\n'+errortext)

        if weights != None and len(weights) != len(counts):
            errortext = "If using weights, the number of weight values and the number of hypotheses should be the same!"
            errortext += '\nCurrently there are {:.0f} hypotheses and {:.0f} weight value(s)'.format(len(counts), len(weights))
            errortext = h.print_msg_box(errortext)
            raise ValueError('\n'+errortext)
        
        if weights == None:
            weights = np.ones(len(counts), dtype=float)
        else:
            weights = np.array(weights)
        
        self.weights = np.outer(weights, weights) #generates a matrix from the outer product of two vectors

        self.n = len(counts)
        self.subtraction_metric=subtraction_metric

        self.original_bins = bins.copy()
        self.original_counts = np.vstack(counts)
        self.original_counts = self.original_counts.astype(float)
        self.original_counts = self.original_counts.T
        self.original_counts /= np.abs(self.original_counts).sum(axis=0)
        self.original_counts = self.original_counts.T
        
        stats_for_mean = np.concatenate(counts)
        
        if not stats_check:
            self.merged_counts, self.post_stats_merge_bins = self.original_counts.copy(), self.original_bins.copy()
        else:
            self.merged_counts, self.post_stats_merge_bins = h.merge_bins(0.05*np.mean( stats_for_mean ), 
                                            bins, 
                                            *self.original_counts.copy()
                                            )

        self.n_items = len(self.merged_counts[0])
        
        self.local_edges = self.post_stats_merge_bins.copy()
        
        self.counts_to_merge = self.merged_counts.copy()
        self.counts_to_merge = self.merged_counts.astype(float)
        
        self.SM_version = SM_version
    
    def reset(self):
        """Resets the state
        """
        self.counts_to_merge = self.merged_counts.copy()
        self.counts_to_merge = self.merged_counts.astype(float)
        self.local_edges = self.post_stats_merge_bins.copy()
        self.n_items = len(self.merged_counts[0])
    
    @staticmethod
    @nb.njit
    def __MLM__(n, counts, weights, b , bP, subtraction_metric, SM_version):
        """This is the money function that calculates the metric

        Parameters
        ----------
        n : int
            This is the input for self.n - i.e. the total number of entries in each set of counts
        counts : numpy.ndarray
            This is the input for self.counts_to_merge - i.e. the stacked set of histogram counts
        weights : numpy.ndarray
            This is the input for self.weights - i.e. the weight for each hypothesis
        b : int
            The first bin number being compared
        bP : int
            The second bin number being compared
        subtraction_metric : bool
            This is the input for self.subtraction_metric - i.e. whether to use the subtraction metric or the division metric
        SM_version : bool
            This is the input for self.SM_version - i.e. whether to compare everything to each other or just to the first input count

        Returns
        -------
        float
            A score for merging two histogram count entries
        """
        denomenator = numerator = 0
        
        initial_range = np.arange(1, dtype=np.uint32) if SM_version else np.arange(n, dtype=np.uint32)
        
        for h in initial_range:
            for hP in np.arange(h+1, n):
                mat = np.array([
                    [ counts[h][b], counts[h][bP] ],
                    [ counts[hP][b], counts[hP][bP] ]
                ], dtype=np.float64)
                
                mat *= weights[h][hP]
                # print("tests")
                # print(mat)
                # print(counts[h][b], counts[h][bP], counts[hP][b], counts[hP][bP])
                # print(2*np.prod(mat), 2*np.prod([counts[h][b], counts[h][bP], counts[hP][b], counts[hP][bP]]))
                # print()
                if subtraction_metric:
                    numerator += (mat[0][0]*mat[1][1])**2 + (mat[0][1]*mat[1][0])**2 - 2*np.prod(mat)
                else:
                    numerator += (mat[0][0]*mat[1][1])**2 + (mat[0][1]*mat[1][0])**2
                    denomenator += np.prod(mat)
                    
        if subtraction_metric:
            return numerator
        else:
            return numerator/(2*denomenator)
    
    def __merge__(self, i, j):
        """Merges bins together by bin count index

        Parameters
        ----------
        i : int
            index i
        j : int
            index j

        Returns
        -------
        Tuple([numpy.ndarray, numpy.ndarray])
            Returns a numpy histogram tuple of locally merged bins

        Raises
        ------
        ValueError
            Raises a ValueError if the edges are merged upon
        ValueError
            Raises an error if you try to merge non-adjacent bins
        """
        temp_counts = self.counts_to_merge.copy()
        temp_edges = self.local_edges.copy()
        
        if i == j + 1: #merges count i with the count behind it
            if i > self.n_items - 1 or i < 0:
                raise ValueError("YOU IDIOT WHY WOULD YOU MERGE ON THE EDGES THAT LOSES THE RANGE\nTrying to merge {:.0f} with {:.0f}".format(i,j))
            temp_counts = np.concatenate( (self.counts_to_merge[:,:j], np.array([self.counts_to_merge[:,j] + self.counts_to_merge[:,i]]).T, self.counts_to_merge[:,i+1:]), axis=1 )
            temp_edges = np.concatenate( (self.local_edges[:i], self.local_edges[i+1:]) )
        elif i == j - 1: #merges count i with the count in front of it
            if i > self.n_items - 1 or i < 1:
                raise ValueError("YOU IDIOT WHY WOULD YOU MERGE ON THE EDGES THAT LOSES THE RANGE\nTrying to merge {:.0f} with {:.0f}".format(i,j))  
            temp_counts = np.concatenate( (self.counts_to_merge[:,:i], np.array([self.counts_to_merge[:,i] + self.counts_to_merge[:,j]]).T, self.counts_to_merge[:,j+1:]), axis=1 )
            temp_edges = np.concatenate( (self.local_edges[:i+1], self.local_edges[i+2:]) )
        elif i == j: #does nothing
            pass
        else:
            raise ValueError("This is local binning! Can only merge ahead or behind!")
        
        return (temp_counts, temp_edges)
    
    def run(self, target_bin_number):
        """runs the local bin merging

        Parameters
        ----------
        target_bin_number : int
            the number of bins you want

        Returns
        -------
        Tuple(numpy.ndarray, numpy.ndarray)
            a numpy histogram of the new counts and bins
        """
        if self.n_items <= target_bin_number:
            msg = "Bins requested >= current number of bins!"
            msg += "\n({:.0f} >= {:.0f})".format(target_bin_number, self.n_items)
            msg += "\nNo merging will occur"
            warnings.warn("\n"+h.print_msg_box(msg, title="WARNING"))
            
        while self.n_items > target_bin_number:
            combinations = {}
            scores = {}
            
            temp_counts = np.zeros(shape=(self.counts_to_merge.shape[0], self.counts_to_merge.shape[1] - 1), dtype=float)
            
            for i in range(1, self.n_items - 1): #don't merge edge bins/counts!
                score = self.__MLM__(self.n, self.counts_to_merge.copy(), self.weights.copy(), i, i+1, self.subtraction_metric, self.SM_version)
                combinations[i] = (i, i+1)
                scores[i] = score
            
            score = self.__MLM__(self.n, self.counts_to_merge.copy(), self.weights.copy(), 1, 0, self.subtraction_metric, self.SM_version) #try merging counts 1 with count 0 (backwards)
            combinations[0] = (1, 0)
            scores[0] = score
            i1, i2 = combinations[ min(scores, key=scores.get) ]
            # print(scores)
            # print(combinations, '\n')
            # for k in range(self.n):
            temp_counts, temp_bins = self.__merge__(i1, i2)
            
            self.counts_to_merge, self.local_edges = temp_counts, temp_bins
            
            self.n_items -= 1
            
        return self.counts_to_merge, self.local_edges
    
    def run_local_faster(self, target_bin_number):
        """attempts to run faster - needs to be debugged. WIP

        Parameters
        ----------
        target_bin_number : int
            the number of bins you want

        Returns
        -------
        Tuple(numpy.ndarray, numpy.ndarray)
            a numpy histogram of the new counts and bins
        """
        
        things_to_recalculate = set(list(range(1, self.n_items - 1)))
        combinations = {}
        scores = {}
        while self.n_items > target_bin_number:
            # print(things_to_recalculate)
            # print(scores)
            # print(combinations)
            
            temp_counts = np.zeros(shape=(self.counts_to_merge.shape[0], self.counts_to_merge.shape[1] - 1), dtype=float)
            for i in things_to_recalculate: #don't merge edge bins/counts!
                if i == 1:
                    score = self.__MLM__(1, 0) #try merging counts 1 with count 0 (backwards)
                    combinations[0] = (1, 0)
                    scores[0] = score
                score = self.__MLM__(i, i+1)
                combinations[i] = (i, i+1)
                scores[i] = score
            
            item_to_delete = min(scores, key=scores.get)
            i1, i2 = combinations[item_to_delete]
            
            # print(i1, i2, self.n_items, '\n')
            if i1 == self.n_items - 2:
                things_to_recalculate = set([i-1])
            else:
                things_to_recalculate = set((i1-1, i1))
            # print(scores)
            # print(combinations)
            for index in scores.keys():
                if index > i1:
                    scores[index-1] = scores[index]
                    scores[index] = np.inf
                    
                    old_i1, old_i2 = combinations[index]
                    combinations[index-1] = (old_i1 - 1, old_i2 - 1)
                    combinations[index] = (np.nan, np.nan)
            
            temp_counts, temp_bins = self.__merge__(i1, i2)
            
            self.counts_to_merge, self.local_edges = temp_counts, temp_bins
            
            self.n_items -= 1
            
        return self.counts_to_merge, self.local_edges

class Grim_Brunelle_nonlocal(Grim_Brunelle_merger):
    def __init__(self, bins, *counts, stats_check=True, subtraction_metric=True, weights=None, SM_version=False) -> None:
        """This class is the base object for bin merging. 
        It merges bins nonlocally - so the bins no longer contain any physical meaning afterwards.
        Please only input 1 dimensional histograms! Should you need to operate upon a multidimensional histogram - please unroll them first.

        Parameters
        ----------
        bins : numpy.ndarray
            These are your bin edges
        counts : numpy.ndarray
            These are a set of bin counts put in as args. Place an unlimited number of them - each one corresponds to a different hypothesis. 
            Both bins and counts can be created using numpy.histogram()
        stats_check : bool, optional
            If you want to insert a statistics check enable this option, by default True
        subtraction_metric : bool, optional
            If you want to use the metric that does subtraction. If false, it uses version of the metric that uses division, by default True
        weights : numpy.ndarray, optional
            A list of weights of the same length as the number of counts put in for each hypothesis. If None all weights are 1, by default None
        SM_version : bool, optional
            Whether you are using the "Standard Model" version where samples are compared to the first set of inputted counts instead of to everything else

        Raises
        ------
        ValueError
            If your bin counts are not all the same length, raise an error
        ValueError
            If len(counts) != len(bins) - 1 then raise an error
        ValueError
            If len(weights) != len(counts) then raise an error
        """
        bins = np.array(bins)
        self.physical_bins = bins
        self.n_observables = bins.shape[0] if len(bins.shape) > 1 else 1
        # print(self.n_observables)
        self.original_counts_shape = counts[0].shape
        
        unrolled = list(map(np.ndarray.ravel, counts))
        bins = np.arange(len(unrolled[0]) + 1)
        
        
        super().__init__(bins, *unrolled, stats_check=stats_check, subtraction_metric=subtraction_metric, weights=weights, SM_version=SM_version)
    
        self.tracker = np.full((self.n_items, self.n_items),np.nan, object)
        
        for i in range(self.n_items):
            self.tracker[self.n_items - 1][i] = tuple([i]) #record where it started, and where it ended
            
        self.things_to_recalculate = tuple([i for i in range(self.n_items)])
        
        self.scores = np.zeros((self.n_items, self.n_items))
    
    def dump_edges(self, fname=""):
        """This function dumps everything into files of your choice. 
        The tool for placement is put into the python function file
        It will also dump the tracker into a numpy memorymap .mm file
        
        Parameters
        ----------
        fname : str, optional
            The filename you want to output to. If none is given it will dump to "dumped", by default ""
        """
        
        if not fname:
            fname = "dumped"
            
        bin_map = np.memmap(fname+".mm", dtype=np.int32, shape=self.tracker.shape, mode='w+')
        bin_map[:] = -1
        for n_bins, set_of_bins in enumerate(self.tracker):
            for new_bin, bin_tuple in enumerate(set_of_bins):
                if not isinstance(bin_tuple, tuple):
                    continue
                for index in bin_tuple:
                    bin_map[n_bins][index] = new_bin
        
        np.set_printoptions(threshold=sys.maxsize)
        edges_str = np.array2string(self.physical_bins, separator=',', formatter={'float_kind':lambda x: "%.3f" % x}, max_line_width=np.inf)
        
        docstr = '''"""**place_entry** will place a set of observables into a given bin index from a previous optimization

	Parameters
	----------
	N : int
		The number of bins that you want total
	observables : numpy.ndarray
		A _1-d_ array of observables that you want to to place
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
	"""'''
        
        
        code = ["import numpy as np"]
        code += ["def place_entry(N, observables, verbose=False):"]
        code = "\n".join(code)
        
        func = [docstr]
        func += ["if len(observables) != {:.0f}:".format(self.n_observables)]
        func += ["\traise ValueError('Must input {:.0f} observables for this histogram!')".format(self.n_observables)]
        func += ["observables = np.array(observables, dtype=np.float64)"]
        func += ["edges=np.array("+edges_str + ")"]
        if self.n_observables > 1:
            func += ["observables = observables.reshape(1, edges.shape[0])"]
            func += ["counts, *_ = np.histogramdd(observables, edges)"]
        else:
            func += ["counts, *_ = np.histogram(observables, edges)"]
        func += ["nonzero_rolled = np.fromiter(map( lambda x: x[0], np.nonzero(counts)), dtype=np.uint64)"]
        func += ["if verbose:"]
        func += ["\tprint('original index of:', nonzero_rolled)"]
        func += ["\tprint('This places your point in the range:')"]
        func += ["\tfor i in range(edges.shape[0]):"]
        func += ["\t\tprint('[', edges[i][nonzero_rolled[i]], ',', edges[i][int(nonzero_rolled[i]+1)], ']')"]
        func += ["counts = counts.ravel()"]
        func += ["nonzero = np.nonzero(counts)"]
        func += ["mapping = np.memmap('"+fname+ ".mm', dtype=np.int32, mode='r', shape=" + str(self.tracker.shape) + ")"]
        func += ["mapped_val = mapping[N-1][nonzero]"]
        func += ["if mapped_val < 0:"]
        func += ["\traise ValueError('number of bins: ' + str(N) + ' was not calculated ')"]
        func += ["return nonzero_rolled, mapped_val[0]"]
        
        func = "\n\t".join(func)
        func = "\n\t" + func
        
        code2 = "\ndef help():"
        helper = ["text = " + docstr]
        helper += ["print(text)"]
        helper += ["exit()"]
        helper = "\n\t".join(helper)
        helper = "\n\t" + helper
        
        with open(fname+".py", "w+") as f:
            f.write(code)
            f.write(func)
            f.write(code2)
            f.write(helper)
    
    def __merge__(self, i, j):
        """Merges bins together by bin count index

        Parameters
        ----------
        i : int
            index i
        j : int
            index j

        Returns
        -------
        Tuple([numpy.ndarray, numpy.ndarray])
            Returns a numpy histogram tuple of nonlocally merged bin edges and their counts. The bin edges are nonphysical, so are just a range of indices equally spaced.
        """
        merged_counts = np.zeros(shape=(self.n, self.n_items - 1), dtype=float)
        k = 0
        
        for n in range(self.n_items):
            if n != i and n != j:
                self.tracker[self.n_items - 2][k] = self.tracker[self.n_items - 1][n]
                merged_counts[:,k] = self.counts_to_merge[:,n]
                self.scores[:,k] = self.scores[:,n]
                self.scores[k] = self.scores[n]
                
                k += 1
        
        self.tracker[self.n_items - 2][k] = self.tracker[self.n_items - 1][i] + self.tracker[self.n_items - 1][j]
        self.things_to_recalculate = tuple([k])
        
        merged_counts[:,k] = self.counts_to_merge[:,i] + self.counts_to_merge[:,j] #shove everything into the final bin
        
        # for wipe in range(k, len(self.scores)): #ALSO WIPES THE OLD ROW!
            #wipe away rows and columns that are no longer being considered
            #(i.e. phasespace moves down from 5 bins to 4, wipe scores from old bin 5)
        self.scores[k:] = np.inf
        self.scores[:,k:] = np.inf
            
        self.n_items -= 1
        
        self.counts_to_merge = merged_counts
        
        # self.tracker[self.n_items - 1] = current_iteration_tracker
        
        return self.counts_to_merge

    def __closest_pair__(self):
        """A simple function to find the closest pair of points between any two bins nonlocally

        Returns
        -------
        Tuple([float, int, int])
            A tuple of values containing the smallest distance, and the indices that create that distance

        Raises
        ------
        ValueError
            If there is ever a nan returned throw an error. This is only possible when using the division version of the metric.
        """
        # print("USING INDICES:", indices, "With BRUTE_FORCE=", brute_force)
        smallest_distance = (np.inf, None, None)
        for i in range(self.n_items):
            for j in self.things_to_recalculate:
                if i == j:
                    self.scores[i][j] = np.inf
                    continue
                
                self.scores[i][j] = self.__MLM__(self.n, self.counts_to_merge.copy(), self.weights.copy(), i, j, self.subtraction_metric, self.SM_version)
        smallest_distance_index = np.unravel_index(self.scores.argmin(), self.scores.shape)
        smallest_distance = self.scores[smallest_distance_index], *smallest_distance_index
        
        if not np.isfinite(smallest_distance[0]):
            raise ValueError("Distance function has produced nan/inf at some point with value" + str(smallest_distance[0]))
        
        return smallest_distance
    
    def run(self, target_bin_number=1):
        """runs the nonlocal binning

        Parameters
        ----------
        target_bin_number : int
            the number of bins you want

        Returns
        -------
        Tuple(numpy.ndarray, numpy.ndarray)
            a numpy histogram of the new counts and bins
        """
        
        if self.n_items <= target_bin_number:
            msg = "Bins requested >= current number of bins!"
            msg += "\n({:.0f} >= {:.0f})".format(target_bin_number, self.n_items)
            msg += "\nNo merging will occur"
            warnings.warn("\n"+h.print_msg_box(msg, title="WARNING"))
            
        pbar = tqdm.tqdm(total=self.n_items - target_bin_number)
        while self.n_items > target_bin_number:
            distance, i, j = self.__closest_pair__()
            self.__merge__(i,j)
            pbar.update(1)
            # print()
    
        return self.counts_to_merge, np.array(range(self.n_items+1))
    
    def visualize_changes(self, n=None, xlabel=None, fname=""):
        if len(self.tracker) == 1:
            errortext = "Need to have used the run command with track=True to visualize this!"
            raise RuntimeError('\n' + h.print_msg_box(errortext, title="ERROR"))
        
        plt.close('all')
        with plt.style.context(hep.style.CMS):
            plt.figure(figsize=(10,7), facecolor='white')
            centers = (self.post_stats_merge_bins[1:] + self.post_stats_merge_bins[:-1])/2
            
            if n == None:
                n = self.n_items
            
            color_wheel = iter(plt.cm.rainbow(np.linspace(0, 1, n)))
            for bin_row in self.tracker[n - 1]:
                # indices = self.__trace__(bin_index, len(self.tracker) - 1)
                if not isinstance(bin_row, tuple) and np.isnan(bin_row):
                    continue
                indices = bin_row
                c = next(color_wheel)
                for index in indices:
                    plt.scatter(centers[index], self.merged_counts[0][index], color=c, marker='.', s=25)
                    plt.scatter(centers[index], self.merged_counts[1][index], color=c, marker='x', s=25)
                    # plt.scatter(centers[index], self.merged_counts[2][index], color=c, marker='1', s=25)
            
            if xlabel == None:
                "Distribution Clustering"
            else:
                xlabel = "Nonlocal clustering for " + xlabel
            plt.title("Merging Non-locally to {:.0f} bins from {:.0f} bins".format(n, len(self.merged_counts[0])))
            plt.xlabel(xlabel)
            plt.tight_layout()
            if fname:
                plt.savefig(fname + '.png')
                # plt.show()
            plt.close()
            return self.tracker
