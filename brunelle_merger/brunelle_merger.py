import numpy as np
import inspect
def grim_metric(i, j, *counts):
    counts = np.array(counts)
    counts = counts.copy()
    it = iter(counts)
    the_len = len(next(it))
    
    if not all(len(l) == the_len for l in it):
        raise ValueError('not all lists have same length!')

    n = len(counts)
    n_items = len(counts[0])
    
    overall_counts = np.vstack(counts)
    
    h = np.ravel(overall_counts[:,i])
    hBar = np.ravel(overall_counts[:,j])
    
    terms = hBar/h
    numerator = (np.nancumprod(terms)**(1./n))[-1]
    denomenator = np.sum(terms)
    
    metric = (numerator/denomenator)#*np.exp(np.sum(h + hBar)/np.sum(overall_counts) - 1 )
    return metric

def __closest_pair__(distanceFunc, *counts):
    if len(counts[0]) == 1:
        return np.inf
    elif len(counts[0]) == 2:
        return distanceFunc(0,1,*counts)
    else:
        m = len(counts[0])//2
        s1, s2 = counts[:,:m], counts[:,m:]
        d1 = __closest_pair__(distanceFunc, s1)
        d2 = __closest_pair__(distanceFunc, s2)
        d12 = distanceFunc(m-1, m, *counts)
        
        return min(d1, d2, d12)


def merge_bins(target, bins, *counts, **kwargs):
    """Merges a set of bins that are given based off of the counts provided
    Eliminates any bin with a corresponding count that is less than the target
    Useful to do merge_bins(*np.histogram(data), ...)
    
    
    Parameters
    ----------
    counts : numpy.ndarray
        The counts of a histogram
    bins : numpy.ndarray
        The bins of a histogram
    target : int, optional
        The target value to achieve - any counts below this will be merged, by default 0
    ab_val : bool, optional
        If on, the target will consider the absolute value of the counts, not the actual value, by default True
    drop_first : bool, optional
        If on, the function will not automatically include the first bin edge, by default False

    Returns
    -------
    Tuple(numpy.ndarray, numpy.ndarray)
        A np.histogram object with the bins and counts merged

    Raises
    ------
    ValueError
        If the bins and counts are not sized properly the function will fail
    """
    
    drop_first = kwargs.get('drop_first',False)
    ab_val = kwargs.get('ab_val', True)
    
    new_counts = []
    [new_counts.append([]) for _ in counts]
        
    counts = np.vstack(counts)
    
    if any([len(bins) != len(count) + 1 for count in counts]):
        errortext = "Length of bins is {:.0f}, lengths of counts are ".format(len(bins))
        errortext += " ".join([str(len(count)) for count in counts])
        errortext += "\nlen(bins) should be len(counts) + 1!"
        raise ValueError("\n" + errortext)
    
    
    if not drop_first:
        new_bins = [bins[0]] #the first bin edge is included automatically if not explicitly stated otherwise
    else:
        new_bins = []
        
    if ab_val:
        counts = np.abs(counts)
    
    
    i = 0
    while i < len(counts[0]):
        summation = np.zeros(len(counts))
        start = i
        while np.any(summation <= target) and (i < len(counts[0])):
            summation += counts[:,i]
            i += 1
        
        if drop_first and len(new_bins) == 0:
                first_bin = max(i - 1, 0)
                new_bins += [bins[first_bin]]
                
        if not( np.any(summation <= target) and (i == len(counts[0])) ):
            for k in range(len(counts)):
                new_counts[k] += [np.sum(counts[k][start:i])]
            new_bins += [bins[i]]
        else:
            for k in range(len(counts)):
                new_counts[k][-1] += np.sum(counts[start:i])
            new_bins[-1] = bins[i]
    
    return np.vstack(new_counts), np.array(new_bins)


def merge_ahead(counts, bins, i):
    """Merges the bin at index i with its buddy at index i+1
    The buddy absorbs the selected bin
    i.e. [2,3,4] -> [2,4] if i=1

    Parameters
    ----------
    counts : numpy.ndarray
        the counts for your histogram
    bins : numpy.ndarray
        the bins for your histogram
    i : int
        the index

    Returns
    -------
    Tuple(numpy.ndarray, numpy.ndarray)
        the counts and bins for your merged histogram (numpy style histogram)

    Raises
    ------
    ValueError
        if you merge an invalid point
    """
    if i >= len(counts) - 1 or i < 1:
        raise ValueError("YOU IDIOT WHY WOULD YOU MERGE ON THE EDGES THAT LOSES THE RANGE")
    
    # rest_of_array = min(i+2, len(counts) - 1)
    
    counts = np.concatenate( (counts[:i], [counts[i] + counts[i+1]], counts[i+2:]) )
    #above does the following, <rest_of_list> + (merged_counts) + <rest_of_list>
    bins = np.concatenate( (bins[:i+1], bins[i+2:]) )
    #merge bins by removing a bin edge at index i
    return counts, bins


def merge_behind(counts, bins, i):
    """Merges the bin at index i with its buddy at index i-1
    The buddy absorbs the selected bin
    i.e. [2,3,4] -> [2,4] if i=1

    Parameters
    ----------
    counts : numpy.ndarray
        the counts for your histogram
    bins : numpy.ndarray
        the bins for your histogram
    i : int
        the index

    Returns
    -------
    Tuple(numpy.ndarray, numpy.ndarray)
        the counts and bins for your merged histogram (numpy style histogram)

    Raises
    ------
    ValueError
        if you merge an invalid point
    """
    if i > len(counts) - 1 or i < 1:
        raise ValueError("YOU IDIOT WHY WOULD YOU MERGE ON THE EDGES THAT LOSES THE RANGE")
    
    # rest_of_array = min(i+2, len(counts) - 1)
    
    counts = np.concatenate( (counts[:i-1], [counts[i-1] + counts[i]], counts[i+1:]) )
    #above does the following, <rest_of_list> + (merged_counts) + <rest_of_list>
    bins = np.concatenate( (bins[:i], bins[i+1:]) )
    #merge bins by removing a bin edge at index i
    return counts, bins

class Grim_Brunelle_merger(object):
    def __init__(self, bins, *counts) -> None:
        
        it = iter(counts)
        the_len = len(next(it))
        if not all(len(l) == the_len for l in it):
            errortext = [str(len(count)) for count in counts]
            errortext = " ".join(errortext)
            raise ValueError('Not all lists have same length! Lengths are: ' + errortext)
        
        if len(counts[0]) != len(bins) - 1:
            raise ValueError("Invalid lengths! {:.0f} != {:.0f} + 1".format(len(counts[0]), len(bins)))

        self.n = len(counts)


        self.original_bins = bins.copy()
        self.original_counts = np.vstack(counts)
        self.original_counts = self.original_counts.T
        self.original_counts /= np.abs(self.original_counts).sum(axis=0)
        self.original_counts = self.original_counts.T
        
        stats_for_mean = np.concatenate(counts)
        
        self.merged_counts, self.post_stats_merge_bins = merge_bins(0.05*np.mean( stats_for_mean ), 
                                        bins, 
                                        *counts
                                        )

        self.n_items = len(self.merged_counts[0])
        
        self.local_edges = self.post_stats_merge_bins.copy()
        
        self.counts_to_merge = self.merged_counts.copy()
        self.counts_to_merge = self.merged_counts.astype(float)
    
    def reset(self):
        self.counts_to_merge = self.merged_counts.copy()
        self.local_edges = self.post_stats_merge_bins.copy()
        self.n_items = len(self.merged_counts[0])
    
    def __MLM__(self, b, bP):
        counts = self.counts_to_merge.copy()
        numerator = 0
        denomenator = 0
        
        for h in range(self.n):
            for hP in range(h+1, self.n):
                numerator += (counts[h][b]*counts[hP][bP])**2 + (counts[hP][b]*counts[h][bP])**2
                denomenator += np.prod(
                    [counts[h][b], counts[hP][bP], counts[h][bP], counts[hP][b]]
                    )
        return numerator/(2*denomenator)
    
    def __merge__(self, i, j, local=True):
        if local:
            temp_counts = self.counts_to_merge.copy()
            temp_edges = self.local_edges.copy()
            
            if i == j + 1: #merges count i with the count behind of it
                if i > self.n_items - 1 or i < 0:
                    raise ValueError("YOU IDIOT WHY WOULD YOU MERGE ON THE EDGES THAT LOSES THE RANGE")
                temp_counts = np.concatenate( (self.counts_to_merge[:,:j], np.array([self.counts_to_merge[:,j] + self.counts_to_merge[:,i]]).T, self.counts_to_merge[:,i+1:]), axis=1 )
                temp_edges = np.concatenate( (self.local_edges[:i], self.local_edges[i+1:]) )
            elif i == j - 1: #merges count i with the count in front of it
                if i > self.n_items - 1 or i < 1:
                    raise ValueError("YOU IDIOT WHY WOULD YOU MERGE ON THE EDGES THAT LOSES THE RANGE")    
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
    
    def __closest_pair__(self, indices, brute_force=False):
        if brute_force:
            smallest_distance = [np.inf, None, None]
            for i in range(len(self.counts_to_merge)):
                for j in range(len(self.counts_to_merge)):
                    if i == j:
                        continue
                    temp_dist = self.__MLM__(i, j)
                    if temp_dist < smallest_distance[0]:
                        smallest_distance = [temp_dist, i, j]
            return smallest_distance
        
        elif len(indices) == 1:
            return np.inf
        elif len(indices) == 2:
            return [self.__MLM__(*indices), *indices]
        else:
            m = len(indices)//2
            s1, s2 = indices[:,:m], indices[:,m:]
            
            # left, right = self.counts_to_merge[s1], self.counts_to_merge[s2]
            
            d1, s1 = __closest_pair__(s1)
            d2, s2 = __closest_pair__(s2)
            
            d12 = self.__MLM__(m-1, m)
            
            mappable = {
                d1  : s1,
                d2  : s2,
                d12 : [m-1,m],
            }
            
            min_dist = min(d1, d2, d12)
            min_indices = mappable[min_dist]
            
            return [min_dist, *min_indices]
    
    def run_local(self, target_bin_number):
        
        things_to_recalculate = set(list(range(1, self.n_items - 1)))
        
        while self.n_items > target_bin_number:
            combinations = {}
            scores = {}
            
            temp_counts = np.zeros(shape=(self.counts_to_merge.shape[0], self.counts_to_merge.shape[1] - 1), dtype=float)
            
            for i in range(1, self.n_items - 1): #don't merge edge bins/counts!
                score = self.__MLM__(i, i+1)
                combinations[i] = (i, i+1)
                scores[i] = score
            
            score = self.__MLM__(1, 0) #try merging counts 1 with count 0 (backwards)
            combinations[0] = (1, 0)
            scores[0] = score
            i1, i2 = combinations[ min(scores, key=scores.get) ]
            print(scores)
            print(combinations, '\n')
            # for k in range(self.n):
            temp_counts, temp_bins = self.__merge__(i1, i2)
            
            self.counts_to_merge, self.local_edges = temp_counts, temp_bins
            
            self.n_items -= 1
            
        return self.counts_to_merge, self.local_edges
    
    def run_local_faster(self, target_bin_number):
        things_to_recalculate = set(list(range(1, self.n_items - 1)))
        combinations = {}
        scores = {}
        while self.n_items > target_bin_number:
            print(things_to_recalculate)
            print(scores)
            print(combinations)
            
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
            
            print(i1, i2, self.n_items, '\n')
            if i1 == self.n_items - 2:
                things_to_recalculate = set([i-1])
            else:
                things_to_recalculate = set((i1-1, i1))
            print(scores)
            print(combinations)
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
    
    def run_nonlocal(self, target_bin_number):
        while self.n_items > target_bin_number:
            indices = list(range(len(self.counts_to_merge)))
            distance, i, j = self.__closest_pair__(indices)
            # print(self.n_items, (i,j), distance)
            self.__merge__(i,j, local=False)
    
        return self.counts_to_merge, np.array(range(self.n_items))
    

class Brunelle_merger(object): #Professor Nathan Brunelle!
    #https://engineering.virginia.edu/faculty/nathan-brunelle
    def __init__(self, counts_1, counts_2, bins, distanceFunc) -> None:
        """Our Funky Merger (Name will be changed later)

        Parameters
        ----------
        counts_1 : numpy.ndarray
            histogram counts for hypothesis 1
        counts_2 : numpy.ndarray
            histogram counts for hypothesis 2
        bins : numpy.ndarray
            histogram bins for both hypotheses
        distanceFunc : function
            a function to determine distances between two sets of counts

        Raises
        ------
        TypeError
            distanceFunc must be a function that takes in two parameters (two counts)
        ValueError
            If your lengths are wrong
        """
        
        if len(counts_1) != len(counts_2):
            raise ValueError("Count lengths are not the same! {:.0f} != {:.0f}".format(len(counts_1), len(counts_2)))
        
        if len(counts_1) != len(bins) - 1:
            raise ValueError("Invalid lengths! {:.0f} != {:.0f} + 1".format(len(counts_1), len(bins)))
        
        if len(counts_2) != len(bins) - 1:
            raise ValueError("Invalid lengths! {:.0f} != {:.0f} + 1".format(len(counts_2), len(bins)))
        
        # self.counts_1 = counts_1
        # self.counts_2 = counts_2
        
        self.original_bins = bins
        
        
        (self.counts_1, self.counts_2), self.post_stats_merge = merge_bins(0.05*np.mean( np.concatenate((counts_1.copy(), counts_2.copy())) ), 
                                        bins.copy(), 
                                        counts_1.copy(),
                                        counts_2.copy()
                                        )
        
        self.counts_1 = np.array(self.counts_1)
        self.counts_2 = np.array(self.counts_2)
        self.post_stats_merge = np.array(self.post_stats_merge)
        
        # self.original_bins = merge_bins(self.counts_2.copy(), self.original_bins.copy(), 
        #                                 0.05*np.mean(self.counts_2.copy()))
        
        if not callable(distanceFunc):
            raise TypeError("Function provided must be of type function, not type {}!".format(type(distanceFunc)))
        
        if len(inspect.signature(distanceFunc).parameters) != 2:
            raise TypeError("Function provided must take in 2 parameters, not {:.0f}!".format(
                            len(inspect.signature(distanceFunc).parameters))
                            )
        
        self.distanceFunc = distanceFunc
        
        
    def greedy_merge(self, target_bin_number):
        """merges greedily. You can never dig too deep!

        Parameters
        ----------
        target_bin_number : int
            The target number of bins you want

        Returns
        -------
        Tuple(numpy.ndarray, numpy.ndarray, numpy.ndarray)
            a tuple of the new counts (1 and 2) and the new bins
        """
        new_counts_1 = self.counts_1.copy()
        new_counts_2 = self.counts_2.copy()
        new_bins = self.post_stats_merge.copy()
        
        while len(new_counts_1) > target_bin_number:
            
            # print(len(new_bins))
            
            current_distance = self.distanceFunc(new_counts_1, new_counts_2)
            
            combinations = {}
            scores = {}
            
            for i in range(1, len(new_counts_1) - 1): #don't merge edge bins/counts!
                temp_counts_1, temp_bins = merge_ahead(new_counts_1, new_bins, i)
                temp_counts_2, _ = merge_ahead(new_counts_2, new_bins, i)
                
                score = current_distance - self.distanceFunc(temp_counts_1, temp_counts_2)
                
                combinations[i] = (temp_counts_1, temp_counts_2, temp_bins)
                scores[i] = score
                
            new_counts_1, new_counts_2, new_bins = combinations[ min(scores, key=scores.get) ]
            # print(combinations.keys())
            # new_counts_1, new_counts_2, new_bins = combinations[
            #     np.min(list(combinations.keys()))
            #     ]
        
        return new_counts_1, new_counts_2, new_bins
    
    def greedy_grim_merge(self, target_bin_number):
        """merges greedily. You can never dig too deep!

        Parameters
        ----------
        target_bin_number : int
            The target number of bins you want

        Returns
        -------
        Tuple(numpy.ndarray, numpy.ndarray, numpy.ndarray)
            a tuple of the new counts (1 and 2) and the new bins
        """
        new_counts_1 = self.counts_1.copy()
        new_counts_2 = self.counts_2.copy()
        new_bins = self.post_stats_merge.copy()
        
        def grim_metric(i, ahead, *counts):
            it = iter(counts)
            the_len = len(next(it))
            
            if not all(len(l) == the_len for l in it):
                raise ValueError('not all lists have same length!')

            n = len(counts)
            n_items = len(counts[0])
            
            overall_counts = np.vstack(counts)
            
            h = overall_counts[:,i]
            if ahead:
                if i + 1 >= n_items:
                    raise IndexError("IDIOT")
                hBar = np.ravel(overall_counts[:,i+1])
            else:
                if i - 1 < 0:
                    raise IndexError("IDIOT")
                hBar = np.ravel(overall_counts[:,i-1])
            
            terms = hBar/h
            numerator = (np.nancumprod(terms)**(1./n))[-1]
            denomenator = np.sum(terms)
            
            metric = (numerator/denomenator)#*np.exp(np.sum(h + hBar)/np.sum(overall_counts) - 1 )
            return metric
            
            
            # counts = np.concatenate( (counts[:i], [counts[i] + counts[i+1]], counts[i+2:]) )
            # #above does the following, <rest_of_list> + (merged_counts) + <rest_of_list>
            # bins = np.concatenate( (bins[:i], bins[i+1:]) )
            # #merge bins by removing a bin edge at index i
            # return counts, bins
            
            
            
        while len(new_counts_1) > target_bin_number:
            combinations = {}
            scores = {}
            
            for i in range(1, len(new_counts_1) - 1): #don't merge edge bins/counts!
                temp_counts_1_ahead, temp_bins = merge_ahead(new_counts_1, new_bins, i)
                temp_counts_2_ahead, _ = merge_ahead(new_counts_2, new_bins, i)
                score_ahead = grim_metric(i, True, new_counts_1, new_counts_2)
                combinations[i] = (temp_counts_1_ahead, temp_counts_2_ahead, temp_bins)
                scores[i] = score_ahead
            
            for i in range(len(new_counts_1) - 2, 0, -1):
                temp_counts_1_behind, temp_bins = merge_behind(new_counts_1, new_bins, i)
                temp_counts_2_behind, _ = merge_behind(new_counts_2, new_bins, i)
                score_behind = grim_metric(i, False, new_counts_1, new_counts_2)
                combinations[-i] = (temp_counts_1_behind, temp_counts_2_behind, temp_bins)
                scores[-i] = score_behind
            
            # print(scores)
            new_counts_1, new_counts_2, new_bins = combinations[ max(scores, key=scores.get) ]
            # print(combinations.keys())
            # new_counts_1, new_counts_2, new_bins = combinations[
            #     np.min(list(combinations.keys()))
            #     ]
        
        return new_counts_1, new_counts_2, new_bins

    def nonlocal_1d_greedy_merge(self, target_bin_number):
        """merges greedily. You can never dig too deep!

        Parameters
        ----------
        target_bin_number : int
            The target number of bins you want

        Returns
        -------
        Tuple(numpy.ndarray, numpy.ndarray, numpy.ndarray)
            a tuple of the new counts (1 and 2) and the new bins
        """
        new_counts_1 = self.counts_1.copy()
        new_counts_2 = self.counts_2.copy()
        new_bins = self.post_stats_merge.copy()
        
        def grim_metric(i, j, *counts):
            it = iter(counts)
            the_len = len(next(it))
            
            if not all(len(l) == the_len for l in it):
                raise ValueError('not all lists have same length!')

            n = len(counts)
            n_items = len(counts[0])
            if i >= n_items or j >= n_items:
                raise IndexError("IDIOT")
            
            overall_counts = np.vstack(counts)
            
            h = overall_counts[:,i]
            hBar = np.ravel(overall_counts[:,j])
            
            terms = hBar/h
            numerator = (np.nancumprod(terms)**(1./n))[-1]
            denomenator = np.sum(terms)
            
            metric = (numerator/denomenator)#*np.exp(np.sum(h + hBar)/np.sum(overall_counts) - 1 )
            return metric


