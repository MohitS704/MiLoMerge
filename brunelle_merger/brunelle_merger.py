import numpy as np
import inspect
# def split_hist(counts, bins):
#     median = len(bins)//2
    
#     split_up = (counts[:median], bins[:median+1]), (counts[median:], bins[median:])
#     return split_up

# def combine_hist(left_counts, left_bins, right_counts, right_bins):
#     if left_bins[-1] != right_bins[0]:
#         print(left_bins, right_bins)
#         raise ValueError("Bins must be continuous!")
    
#     return np.concatenate((left_counts, right_counts)), np.concatenate((left_bins, right_bins[1:]))

# def merge_bin_on_edge(left_counts, left_bins, right_counts, right_bins):
#     if left_bins[-1] != right_bins[0]:
#         print(left_bins, right_bins)
#         raise ValueError("Bins must be continuous!")
        
#     merged = (np.concatenate( (left_counts[:-1], [left_counts[-1] + right_counts[0]], right_counts[1:]) ), 
#             np.concatenate( (left_bins[:-1], right_bins[1:]) ) )
    
#     return merged

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
    if i > len(counts) - 1 or i <= 1:
        raise ValueError("YOU IDIOT WHY WOULD YOU MERGE ON THE EDGES THAT LOSES THE RANGE")
    
    # rest_of_array = min(i+2, len(counts) - 1)
    
    counts = np.concatenate( (counts[:i-1], [counts[i-1] + counts[i]], counts[i+1:]) )
    #above does the following, <rest_of_list> + (merged_counts) + <rest_of_list>
    bins = np.concatenate( (bins[:i], bins[i+1:]) )
    #merge bins by removing a bin edge at index i
    return counts, bins

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
        
        self.counts_1 = counts_1
        self.counts_2 = counts_2
        
        self.original_bins = bins
        
        if not callable(distanceFunc):
            raise TypeError("Function provided must be of type function, not type {}!".format(type(distanceFunc)))
        
        if len(inspect.signature(distanceFunc).parameters) != 2:
            raise TypeError("Function provided must take in 2 parameters, not {:.0f}!".format(
                            len(inspect.signature(distanceFunc).parameters))
                            )
        
        self.distanceFunc = distanceFunc
        
        self.scores = {}

    # def compare(self, counts_left_1, counts_left_2, bins_left, counts_right_1, counts_right_2, bins_right): OLD DO NOT USE
    #     merged_1, merged_bins = merge_bin_on_edge(counts_left_1, bins_left, counts_right_1 , bins_right)
    #     merged_2, _ = merge_bin_on_edge(counts_left_2, bins_left, counts_right_2, bins_right)
        
    #     unmerged_1, unmerged_bins = combine_hist(counts_left_1, bins_left, counts_right_1, bins_right)
    #     unmerged_2, _ = combine_hist(counts_left_2, bins_left, counts_right_2, bins_right)
        
    #     T = self.distanceFunc(unmerged_1, unmerged_2) - self.distanceFunc(merged_1, merged_2)
    #     # print("USING:", counts_left_1, counts_left_2, bins_left, "&", counts_right_1, counts_right_2, bins_right)
        
    #     if T:
    #         # print("MERGED:", merged_1, merged_2, merged_bins)
    #         return merged_1, merged_2, merged_bins
    #     else:
    #         # print("REMAINED:", unmerged_1, unmerged_2, unmerged_bins)
    #         return unmerged_1, unmerged_2, unmerged_bins


    # def brunelle_merge(self, counts_1, counts_2, bins): OLD DO NOT USE
    #     if len(bins) < 2:
    #         raise ValueError("Cannot have fewer than 2 bins!")
    #     if len(bins) == 2:
    #         # print("BASECASE", counts_1, counts_2, bins)
    #         return counts_1, counts_2, bins
        
    #     # print("ITERATION:", counts_1, counts_2, bins)
        
    #     (counts_left_1, bins_left), (counts_right_1, bins_right) = split_hist(counts_1, bins)
    #     (counts_left_2, _), (counts_right_2, _) = split_hist(counts_2, bins)
        
    #     counts_left_1, counts_left_2, bins_left = self.brunelle_merge(counts_left_1, counts_left_2, bins_left)
    #     counts_right_1, counts_right_2, bins_right = self.brunelle_merge(counts_right_1, counts_right_2, bins_right)
        
    #     # print("ENTERING COMPARE:", counts_left_1, counts_left_2, bins_left, "&", counts_right_1, counts_right_2, bins_right)
        
    #     return self.compare(counts_left_1, counts_left_2, bins_left, counts_right_1, counts_right_2, bins_right)
    
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
        new_bins = self.original_bins.copy()
        
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
        new_bins = self.original_bins.copy()
        
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
                if i - 1 <= 0:
                    raise IndexError("IDIOT")
                hBar = np.ravel(overall_counts[:,i-1])
            
            terms = hBar/h
            terms[~np.isfinite(terms)] = 0
            numerator = (np.nancumprod(terms)**(1./n))[-1]
            denomenator = np.sum(terms)
            
            return numerator/denomenator
            
            
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
            
            for i in range(len(new_counts_1) - 2, 1, -1):
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



