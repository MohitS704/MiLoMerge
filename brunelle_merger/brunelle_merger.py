import numpy as np
import scipy.stats as sp

def split_hist(counts, bins):
    median = len(bins)//2
    
    split_up = (counts[:median], bins[:median+1]), (counts[median:], bins[median:])
    return split_up

def combine_hist(left_counts, left_bins, right_counts, right_bins):
    if left_bins[-1] != right_bins[0]:
        print(left_bins, right_bins)
        raise ValueError("Bins must be continuous!")
    
    return np.concatenate((left_counts, right_counts)), np.concatenate((left_bins, right_bins[1:]))

def merge_bin_on_edge(left_counts, left_bins, right_counts, right_bins):
    if left_bins[-1] != right_bins[0]:
        print(left_bins, right_bins)
        raise ValueError("Bins must be continuous!")
        
    merged = (np.concatenate( (left_counts[:-1], [left_counts[-1] + right_counts[0]], right_counts[1:]) ), 
            np.concatenate( (left_bins[:-1], right_bins[1:]) ) )
    
    return merged

def merge_ahead(counts, bins, i):
    counts[i]

class Brunelle_merger(object):
    def __init__(self, counts_1, counts_2, bins, distanceFunc) -> None:
        self.counts_1 = counts_1
        self.counts_2 = counts_2
        
        self.original_bins = bins
        
        if not isinstance(distanceFunc, function):
            raise TypeError("Function provided must be of type function, not type {}!".format(type(distanceFunc)))
        
        self.distanceFunc = distanceFunc
        
        self.scores = {}

    def compare(self, counts_left_1, counts_left_2, bins_left, counts_right_1, counts_right_2, bins_right):
        merged_1, merged_bins = merge_bin_on_edge(counts_left_1, bins_left, counts_right_1 , bins_right)
        merged_2, _ = merge_bin_on_edge(counts_left_2, bins_left, counts_right_2, bins_right)
        
        unmerged_1, unmerged_bins = combine_hist(counts_left_1, bins_left, counts_right_1, bins_right)
        unmerged_2, _ = combine_hist(counts_left_2, bins_left, counts_right_2, bins_right)
        
        T = self.distanceFunc(unmerged_1, unmerged_2) - self.distanceFunc(merged_1, merged_2)
        # print("USING:", counts_left_1, counts_left_2, bins_left, "&", counts_right_1, counts_right_2, bins_right)
        
        if T:
            # print("MERGED:", merged_1, merged_2, merged_bins)
            return merged_1, merged_2, merged_bins
        else:
            # print("REMAINED:", unmerged_1, unmerged_2, unmerged_bins)
            return unmerged_1, unmerged_2, unmerged_bins


    def brunelle_merge(self, counts_1, counts_2, bins):
        if len(bins) < 2:
            raise ValueError("Cannot have fewer than 2 bins!")
        if len(bins) == 2:
            # print("BASECASE", counts_1, counts_2, bins)
            return counts_1, counts_2, bins
        
        # print("ITERATION:", counts_1, counts_2, bins)
        
        (counts_left_1, bins_left), (counts_right_1, bins_right) = split_hist(counts_1, bins)
        (counts_left_2, _), (counts_right_2, _) = split_hist(counts_2, bins)
        
        counts_left_1, counts_left_2, bins_left = self.brunelle_merge(counts_left_1, counts_left_2, bins_left)
        counts_right_1, counts_right_2, bins_right = self.brunelle_merge(counts_right_1, counts_right_2, bins_right)
        
        # print("ENTERING COMPARE:", counts_left_1, counts_left_2, bins_left, "&", counts_right_1, counts_right_2, bins_right)
        
        return self.compare(counts_left_1, counts_left_2, bins_left, counts_right_1, counts_right_2, bins_right)
    
    def greedy_merge(self, target_bin_number):
        new_counts_1 = self.counts_1
        new_counts_2 = self.counts_2
        new_bins = self.original_bins
        
        
        
        while len(new_bins) > target_bin_number:
            
    



