import numpy as np
import inspect

class Grim_Brunelle_merger(object):#Professor Nathan Brunelle!
    #https://engineering.virginia.edu/faculty/nathan-brunelle
    def __init__(self, bins, *counts, stats_check=True) -> None:
        """This is the money class

        Parameters
        ----------
        bins : numpy.ndarray
            These are your bin edges
        counts : numpy.ndarray
            These are a set of counts put in as args. Place an unlimited number of them

        Raises
        ------
        ValueError
            If your bin counts are not all the same length, raise an error
        ValueError
            If len(counts) != len(bins) - 1 then raise an error
        """
        
        it = iter(counts)
        the_len = len(next(it))
        if not all(len(l) == the_len for l in it):
            errortext = [str(len(count)) for count in counts]
            errortext = " ".join(errortext)
            raise ValueError('Not all counts have same length! Lengths are: ' + errortext)
        
        if len(counts[0]) != len(bins) - 1:
            raise ValueError("Invalid lengths! {:.0f} != {:.0f} + 1".format(len(counts[0]), len(bins)))

        self.n = len(counts)


        self.original_bins = bins.copy()
        self.original_counts = np.vstack(counts)
        self.original_counts = self.original_counts.T
        self.original_counts /= np.abs(self.original_counts).sum(axis=0)
        self.original_counts = self.original_counts.T
        
        stats_for_mean = np.concatenate(counts)
        
        if not stats_check:
            self.merged_counts, self.post_stats_merge_bins = self.original_counts.copy(), self.original_bins.copy()
        else:
            self.merged_counts, self.post_stats_merge_bins = merge_bins(0.05*np.mean( stats_for_mean ), 
                                            bins, 
                                            *self.original_counts.copy()
                                            )

        self.n_items = len(self.merged_counts[0])
        
        self.local_edges = self.post_stats_merge_bins.copy()
        
        self.counts_to_merge = self.merged_counts.copy()
        self.counts_to_merge = self.merged_counts.astype(float)
    
    def reset(self):
        """Resets the state
        """
        self.counts_to_merge = self.merged_counts.copy()
        self.local_edges = self.post_stats_merge_bins.copy()
        self.n_items = len(self.merged_counts[0])
    
    def __MLM__(self, b, bP, subtraction_metric=True):
        """The distance function MLM

        Parameters
        ----------
        b : int
            index 1 for the counts
        bP : int
            index 2 for the counts
        subtraction_metric : bool, optional
            whether you want to use the subtraction version or not, by default True

        Returns
        ------
        float
            The distance metric between the two indices
        """
        
        counts = self.counts_to_merge.copy()
        numerator = 0
        denomenator = 0
        
        for h in range(self.n):
            for hP in range(h+1, self.n):
                if subtraction_metric:
                    numerator += (counts[h][b]*counts[hP][bP])**2 + (counts[hP][b]*counts[h][bP])**2 - 2*np.prod([counts[h][b], counts[hP][bP], counts[h][bP], counts[hP][b]])
                else:
                    numerator += (counts[h][b]*counts[hP][bP])**2 + (counts[hP][b]*counts[h][bP])**2
                    denomenator += np.prod(
                        [counts[h][b], counts[hP][bP], counts[h][bP], counts[hP][b]]
                        )
        if subtraction_metric:
            return numerator#/(2*denomenator)
        else:
            return numerator/(2*denomenator)
    
    def __merge__(self, i, j, local=True):
        """Merges bins together by bin count index

        Parameters
        ----------
        i : int
            index i
        j : int
            index j
        local : bool, optional
            If true, merge bins locally - otherwise merge nonlocally, by default True

        Returns
        -------
        Tuple([numpy.ndarray, numpy.ndarray]) OR numpy.ndarray
            Returns a numpy histogram tuple if local merging, and a numpy list of counts if nonlocal

        Raises
        ------
        ValueError
            Raises a ValueError if the edges are merged upon
        """
        if local:
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
        else:
            merged_counts = np.zeros(shape=(self.n, self.n_items - 1), dtype=float)
            k = 0
            # print("merging columns:",i,j)
            for n in range(self.n_items):
                if n != i and n != j:
                    merged_counts[:,k] = self.counts_to_merge[:,n]
                    k += 1
            
            merged_counts[:,-1] = self.counts_to_merge[:,i] + self.counts_to_merge[:,j]
            
            # self.non_local_indices[i] = j
            # self.counts_to_merge[:,j] += self.counts_to_merge[:,i]
            # self.counts_to_merge[:,i] = 0
            self.n_items -= 1
            
            self.counts_to_merge = merged_counts
            
            return self.counts_to_merge
    
    def __closest_pair__(self, indices, brute_force=False, subtraction_metric=True):
        """A Recursive function to match the closest pair of points

        Parameters
        ----------
        indices : numpy.ndarray
            A list of possible indices to use in recursion
        brute_force : bool, optional
            If true, brute force the closest pair - use as a sanity check, by default False
        subtraction_metric : bool, optional
            whether you want to use the subtraction metric or not, by default True

        Returns
        -------
            the closest pair of points (the recursive version currently doesn't work properly just use the brute force)
        """
        # print("USING INDICES:", indices, "With BRUTE_FORCE=", brute_force)
        if brute_force:
            smallest_distance = [np.inf, None, None]
            for i in range(self.n_items):
                for j in range(i):
                    temp_dist = self.__MLM__(i, j, subtraction_metric=subtraction_metric)
                    # print("CURRENT DISTANCE:", temp_dist, i, j)
                    # print("distance between", i, j, ":", temp_dist)
                    if temp_dist < smallest_distance[0]:
                        smallest_distance = [temp_dist, i, j]
            
            if not np.isfinite(smallest_distance[0]):
                raise ValueError("Distance function has produced nan/inf at some point with value" + str(smallest_distance[0]))
            return smallest_distance
        else:
            if len(indices) == 1:
                return [np.inf, *indices]
            elif len(indices) == 2:
                return [self.__MLM__(*indices, subtraction_metric=subtraction_metric), *indices]
            else:
                # print("INDICES:", indices)
                m = len(indices)//2
                s1, s2 = indices[:m], indices[m:]
                # print("SPLIT INTO:", s1, s2)
                # left, right = self.counts_to_merge[s1], self.counts_to_merge[s2]
                
                d1, *s1 = self.__closest_pair__(s1.copy(), subtraction_metric=subtraction_metric)
                d2, *s2 = self.__closest_pair__(s2.copy(), subtraction_metric=subtraction_metric)
                d12 = self.__MLM__(m-1, m, subtraction_metric=subtraction_metric)
                # print("FOUND:", d1, s1, d2, s2, d12)
                mappable = {
                    d1  : s1,
                    d2  : s2,
                    d12 : [m-1,m],
                }
                
                min_dist = min(d1, d2, d12)
                min_indices = mappable[min_dist]
                return [min_dist, *min_indices]
    
    def run_local(self, target_bin_number, subtraction_metric=True):
        """runs the local bin merging

        Parameters
        ----------
        target_bin_number : int
            the number of bins you want
        subtraction_metric : bool, optional
            whether you want to use the subtraction metric or not, by default True

        Returns
        -------
        Tuple(numpy.ndarray, numpy.ndarray)
            a numpy histogram of the new counts and bins
        """
        
        while self.n_items > target_bin_number:
            combinations = {}
            scores = {}
            
            temp_counts = np.zeros(shape=(self.counts_to_merge.shape[0], self.counts_to_merge.shape[1] - 1), dtype=float)
            
            for i in range(1, self.n_items - 1): #don't merge edge bins/counts!
                score = self.__MLM__(i, i+1, subtraction_metric=subtraction_metric)
                combinations[i] = (i, i+1)
                scores[i] = score
            
            score = self.__MLM__(1, 0, subtraction_metric=subtraction_metric) #try merging counts 1 with count 0 (backwards)
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
    
    def run_local_faster(self, target_bin_number, subtraction_metric=True):
        """attempts to run faster - needs to be debugged

        Parameters
        ----------
        target_bin_number : int
            the number of bins you want
        subtraction_metric : bool, optional
            whether you want to use the subtraction metric or not, by default True

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
                    score = self.__MLM__(1, 0, subtraction_metric=subtraction_metric) #try merging counts 1 with count 0 (backwards)
                    combinations[0] = (1, 0)
                    scores[0] = score
                score = self.__MLM__(i, i+1, subtraction_metric=subtraction_metric)
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
    
    def run_nonlocal(self, target_bin_number, subtraction_metric=True):
        """runs the nonlocal binning

        Parameters
        ----------
        target_bin_number : int
            the number of bins you want
        subtraction_metric : bool, optional
            whether you want to use the subtraction metric or not, by default True

        Returns
        -------
        Tuple(numpy.ndarray, numpy.ndarray)
            a numpy histogram of the new counts and bins
        """
        
        # print("RUNNING NONLOCAL", self.n_items, target_bin_number)
        while self.n_items > target_bin_number:
            indices = np.array(list(range(self.n_items)))
            distance, i, j = self.__closest_pair__(indices.copy(), brute_force=True, subtraction_metric=subtraction_metric)
            # print("Final checks", i,j,self.n_items)
            self.__merge__(i,j, local=False)
    
        return self.counts_to_merge, np.array(range(self.n_items+1))