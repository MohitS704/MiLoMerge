import warnings
import os
import sys
from abc import ABC, abstractmethod
import numpy as np
import tqdm
import numba as nb
import h5py

DIR = __file__[:__file__.rfind("/")]
sys.path.append(os.path.abspath(DIR + "/../metrics/"))
from ROC_curves import ROC_score

class Merger(ABC):
    """Abstract class that serves as a baseplate for both the local and nonlocal merger"""
    def __init__(
            self,
            bin_edges,
            *counts,
            weights=None,
            comp_to_first=False,
            map_at=None,
            brute_force_at=10
        ) -> None:
        """The initializer for the baseplate class

        Parameters
        ----------
        bin_edges : numpy.ndarray
            These are the edges of your histogram that correspond to physical quantities
        counts : numpy.ndarray
            A series of arrays that correspond to the number of events between your bin edges
        weights : numpy.ndarray, optional
            An array of the weights associated for each of the counts. 
            If none are provided, the weights will be 1, by default None
        comp_to_first : bool, optional
            Whether you would like to compare all samples to the first one provided, 
            as opposed to all of them to each other, by default False
        map_at : list, optional
            A list of bin numbers at which you would like the mapping 
            from the original sample to be recorded, by default None
        brute_force_at : int, optional
            A value at or below which the merger will utilize the "brute-force" 
            approach of merging by calculating a total ROC score, by default 10

        Raises
        ------
        ValueError
            If all the counts provided do not have the same length, raise an error
        ValueError
            If the length of the bin counts are not equal to 1 - len(bin_edges) raise an error
        ValueError
            If the number of samples and the number of weights are not the same, raise an error
        TypeError
            If map_at is not an iterable of some kind, raise an error
        """

        it = iter(counts)
        the_len = len(next(it))
        if not all(len(l) == the_len for l in it):
            errortext = [str(len(count)) for count in counts]
            errortext = " ".join(errortext)
            errortext = 'Not all counts have same length! Lengths are: ' + errortext
            raise ValueError('\n'+errortext)

        if len(counts[0]) != len(bin_edges) - 1:
            errortext = f"len(counts) = {len(counts[0])} != len(bin_edges) - 1 = {len(bin_edges)-1}"
            raise ValueError('\n'+errortext)

        if weights is not None and len(weights) != len(counts):
            errortext = "The # of weight values and the # of samples should be the same!"
            errortext += f'\nThere are {len(counts)} samples and {len(weights)} weight values'
            raise ValueError('\n'+errortext)

        if weights is None:
            weights = np.ones(len(counts), dtype=np.float64)
        else:
            weights = np.array(weights, dtype=np.float64)

        self._merger_type = None

        self.weights = np.outer(weights, weights)
        self.n_hypotheses = len(counts)
        self.n_items = self.original_n_items = len(counts[0])
        #self.n is the number of hypotheses, self.n_items is the current number of bin_edges

        self.comp_to_first = comp_to_first

        self.counts = np.vstack(counts).astype(np.float64)
        self.counts /= np.abs(self.counts).sum(axis=1)[:, None]
        self.counts[~np.isfinite(self.counts)] = 0

        self.bin_edges = np.array(bin_edges, dtype=np.float64)

        if map_at is None:
            map_at = []
        else:
            try:
                iter(map_at)
            except Exception as e:
                raise TypeError("Parameter map_at must be an iterable!") from e

        self.map_at = [i for i in map_at if i < self.n_items]
        self.brute_force_at = brute_force_at
        self.utilize_brute_force = self.n_items <= brute_force_at

    @staticmethod
    @nb.njit(fastmath=True, cache=True, nogil=True, parallel=True)
    def __roc_score_comp_to_first(counts, weights, b, b_prime):
        """This method calls ROC_curves.ROC_score to compare all samples to the first "sm" sample 
        provided and issue a score between the two bin indices.
        
        Internal method!

        Parameters
        ----------
        counts : numpy.ndarray
            An ndarray of the counts with shape (#samples, #bins)
        weights : numpy.ndarray
            An ndarray of the weights for each of the samples
        b : int
            the index of the first bin that is being calculated
        b_prime : int
            the index of the second bin that is being calculated

        Returns
        -------
        float
            The inverse of the ROC score that is calculated
        """
        roc_summation = 0
        merged_into, to_be_merged =  sorted((b, b_prime))
        h = 0

        to_be_merged_val = counts[0, to_be_merged]
        h_counts = np.delete(counts[0], to_be_merged)
        h_counts[merged_into] += to_be_merged_val
        for h_prime, h_prime_counts in enumerate(counts[1:]):
            h_prime += 1

            to_be_merged_val = h_prime_counts[to_be_merged]
            h_prime_counts = np.delete(h_prime_counts, to_be_merged)
            h_prime_counts[merged_into] += to_be_merged_val
            roc_summation += ROC_score(h_counts, h_prime_counts)*weights[h, h_prime]

        return 1/roc_summation


    @staticmethod
    @nb.njit(fastmath=True, cache=True, nogil=True)
    def __mlm_driver_comp_to_first(n, counts, weights, b, b_prime):
        """This method uses the MLM metric to compare all samples to the first "sm" sample 
        provided and issue a score between the two bin indices.

        Parameters
        ----------
        n : int
            The number of samples being dealt with
        counts : numpy.ndarray
            An ndarray of the counts with shape (#samples, #bins)
        weights : numpy.ndarray
            An ndarray of the weights for each of the samples
        b : int
           #if c == k then you don't need to do anything! The bins pre and post merge will be the same the index of the first bin that is being calculated
        b_prime : int
            the index of the second bin that is being calculated

        Returns
        -------
        float
            The score, as prescribed by the MLM metric
        """

        metric_val = 0
        for h_prime in np.arange(1, n, dtype=np.int64):
            t1 = counts[0, b]*weights[0, h_prime]
            t2 = counts[h_prime, b_prime]*weights[0, h_prime]
            t3 = counts[0, b_prime]*weights[0, h_prime]
            t4 = counts[h_prime, b]*weights[0, h_prime]

            metric_val += (t1*t2)**2 + (t3*t4)**2 - 2*t1*t2*t3*t4

        return metric_val


    @staticmethod
    @nb.njit(fastmath=True, cache=True, nogil=True, parallel=True)
    def __roc_score_comp_to_all(counts, weights, b, b_prime):
        """This method calls ROC_curves.ROC_score to compare all samples to each other 
        and issue a score between the two bin indices.
        
        Internal method!

        Parameters
        ----------
        counts : numpy.ndarray
            An ndarray of the counts with shape (#samples, #bins)
        weights : numpy.ndarray
            An ndarray of the weights for each of the samples
        b : int
            the index of the first bin that is being calculated
        b_prime : int
            the index of the second bin that is being calculated

        Returns
        -------
        float
            The inverse of the ROC score that is calculated
        """
        roc_summation = 0
        merged_into, to_be_merged =  sorted((b, b_prime))
        for h, h_counts in enumerate(counts):
            to_be_merged_val = h_counts[to_be_merged]
            h_counts = np.delete(h_counts, to_be_merged)
            h_counts[merged_into] += to_be_merged_val
            for h_prime, h_prime_counts in enumerate(counts[h+1:]):
                h_prime += h+1

                to_be_merged_val = h_prime_counts[to_be_merged]
                h_prime_counts = np.delete(h_prime_counts, to_be_merged)
                h_prime_counts[merged_into] += to_be_merged_val
                roc_summation += ROC_score(h_counts, h_prime_counts)*weights[h, h_prime]

        return 1/roc_summation

    @staticmethod
    @nb.njit(fastmath=True, cache=True, nogil=True)
    def __mlm_driver_comp_to_all(n, counts, weights, b, b_prime):
        """This method uses the MLM metric to compare all samples to each other 
        and issue a score between the two bin indices.

        Parameters
        ----------
        n : int
            The number of samples being dealt with
        counts : numpy.ndarray
            An ndarray of the counts with shape (#samples, #bins)
        weights : numpy.ndarray
            An ndarray of the weights for each of the samples
        b : int
            the index of the first bin that is being calculated
        b_prime : int
            the index of the second bin that is being calculated

        Returns
        -------
        float
            The score, as prescribed by the MLM metric
        """

        metric_val = 0
        for h in np.arange(n):
            for h_prime in np.arange(h+1, n):
                t1 = counts[h, b]*weights[h, h_prime]
                t2 = counts[h_prime, b_prime]*weights[h, h_prime]
                t3 = counts[h, b_prime]*weights[h, h_prime]
                t4 = counts[h_prime, b]*weights[h, h_prime]

                metric_val += (t1*t2)**2 + (t3*t4)**2 - 2*t1*t2*t3*t4

        return metric_val


    def _mlm(self, b, b_prime):
        """Computes the score between selected bins. 
        If the number of bins is below the brute_force_at value, 
        then the "brute-force" algorithm will be used, otherwise
        the MLM metric will be used to eavaluate

        Parameters
        ----------
        b : int
            the first bin index you would like to calculate
        b_prime : int
            the second bin index you would like to calculate

        Returns
        -------
        float
            A score as determined by the appropriate method being called
        """
        if self.utilize_brute_force:
            if self.comp_to_first:
                return self.__roc_score_comp_to_first(
                    self.counts.copy(), self.weights,
                    b, b_prime
                )
            else:
                return self.__roc_score_comp_to_all(
                    self.counts.copy(), self.weights,
                    b, b_prime
                )
        else:
            if self.comp_to_first:
                return self.__mlm_driver_comp_to_first(
                    self.n_hypotheses, self.counts,
                    self.weights, b, b_prime
                )

            return self.__mlm_driver_comp_to_all(
                self.n_hypotheses, self.counts,
                self.weights, b, b_prime
            )

    def __repr__(self) -> str:
        """Representation of the merger object

        Returns
        -------
        str
            A brief summary of the merger object
        """
        return f"Merger of type {self._merger_type} merging {self.original_n_items}"

    @abstractmethod
    def _merge(self, i, j):
        """Merges bins i,j together

        Parameters
        ----------
        i : int
            The first bin index you would like to merge
        j : int
            The second bin you would like to merge

        Returns
        -------
        NotImplemented
            Abstract class does not have a proper implementation!
        """
        return NotImplemented


    @abstractmethod
    def run(self, target_bin_number=2):
        """Runs the merger

        Parameters
        ----------
        target_bin_number : int, optional
            The number of bins you would like to merge down to, by default 2

        Returns
        -------
        NotImplemented
            Abstract class does not have a proper implementation!
        """
        return NotImplemented

class MergerLocal(Merger):
    """
    A merger that merges bins locally. 
    This will not change the physical ordering of the bin edges.
    """
    def __init__(
            self,
            bin_edges,
            *counts,
            weights=None,
            comp_to_first=False,
            map_at = None,
            brute_force_at=0,
            file_prefix=""
        ) -> None:
        """The initializer for the local merging class

        Parameters
        ----------
        bin_edges : numpy.ndarray
            These are the edges of your histogram that correspond to physical quantities
        counts : numpy.ndarray
            A series of arrays that correspond to the number of events between your bin edges
        weights : numpy.ndarray, optional
            An array of the weights associated for each of the counts. 
            If none are provided, the weights will be 1, by default None
        comp_to_first : bool, optional
            Whether you would like to compare all samples to the first one 
            provided, as opposed to all of them to each other, by default False
        map_at : list, optional
            A list of bin numbers at which you would like the mapping from 
            the original sample to be recorded, by default None
        brute_force_at : int, optional
            A value at or below which the merger will utilize the "brute-force" approach 
            of merging by calculating a total ROC score, by default 10
        file_prefix : str, optional
            This is the prefix that comes before the file bin map before "_tracker.hdf5"

        Raises
        ------
        ValueError
            The dimension of the bin edges can only be 1-dimensional
        """
        super().__init__(
            bin_edges, *counts, weights=weights, comp_to_first=comp_to_first,
            map_at=map_at, brute_force_at=brute_force_at
            )

        if self.bin_edges.ndim > 1:
            raise ValueError("LOCAL MERGING CAN ONLY HANDLE 1-DIMENSIONAL ARRAYS")

        self._merger_type = "Local"

        if any(self.map_at):
            self.tracker =  h5py.File(
                f".{file_prefix}_tracker.hdf5", 'w', 
                libver='latest', driver=None, 
                )

            for mapped_bincount in self.map_at:
                self.tracker.create_dataset(
                    str(mapped_bincount), (mapped_bincount + 1), np.float64,
                    compression='gzip', compression_opts=9,
                    fletcher32=True, fillvalue=0,
                    maxshape=len(bin_edges), shuffle=True
                )
        else:
            self.tracker = None


    @staticmethod
    @nb.njit( 
        cache=True, fastmath=True, nogil=True
        )
    def __merge_driver(counts, bin_edges, first_part, second_part):
        """This method is the numba-ified function that is called by _merge. 
        It handles merging two histogram bins together and editing the bin edges inplace

        Parameters
        ----------
        counts : numpy.ndarray
            A ndarray of the counts with shape (#samples, #bins)
        bin_edges : numpy.ndarray
            A 1-d array of bin edges that correspond to some physical meaning
        first_part : int
            The index that corresponds with the smaller of the two bin eges being merged
        second_part : int
            The index that corresponds with the larger of the two bin eges being merged

        Returns
        -------
        Tuple[numpy.ndarray, numpy.ndarray]
            A tuple of the new counts with two bins merged
            and a 1-d array of the new bin edges with one of the bin edges removed 
        """
        new_counts = counts[:, :-1].copy()

        new_counts[:, first_part] = (counts[:, first_part] + counts[:, second_part]).T
        new_counts[:, first_part + 1:] = counts[:, second_part + 1:]

        new_bin_edges = bin_edges[:-1].copy()
        new_bin_edges[first_part + 1:] = bin_edges[second_part + 1:]

        return (new_counts, new_bin_edges)


    def _merge(self, i, j):
        """Merges bins i and j such that
        the first and last bin edge are always preserved

        Parameters
        ----------
        i : int
            The first index to merge
        j : int
            The second index to merge

        Returns
        -------
        Tuple[numpy.ndarray, numpy.ndarray]
            A tuple of the new counts with two bins merged
            and a 1-d array of the new bin edges with one of the bin edges removed 

        Raises
        ------
        ValueError
            Raise an error if there are unhandled edge cases where i or j are invalid
        ValueError
            Raise an error if bins are sent to the function nonlocally
        """
        if i == j + 1:
            if i > self.n_items - 1 or i < 0:
                raise ValueError("UNHANDLED EDGE CASE WHILE MERGING")
            return self.__merge_driver(self.counts, self.bin_edges, j, i)

        elif i == j - 1:
            if i > self.n_items - 1 or i < 1:
                raise ValueError("UNHANDLED EDGE CASE WHILE MERGING")
            return self.__merge_driver(self.counts, self.bin_edges, i, j)

        elif i == j:
            return (self.counts, self.bin_edges)

        else:
            raise ValueError("This is local binning! Can only merge ahead or behind!")


    def run(self, target_bin_number=2):
        """Runs the merger

        Parameters
        ----------
        target_bin_number : int, optional
            The number of bins you would like to merge down to, by default 2

        Returns
        -------
        numpy.ndarray
            Returns the 1-d bin edges that would 
            correspond to the best binning for the number of bins you want
        """
        if self.n_items <= target_bin_number:
            warnings.warn("Merging is pointless! Number of bins already >= target")

        pbar = tqdm.tqdm(
            total=self.n_items - target_bin_number,
            desc="Binning locally:", leave=True, position=0
            )
        while self.n_items > target_bin_number:
            best_combo = None
            best_score = np.inf

            for i in range(1, self.n_items - 1):
                score = self._mlm(i, i+1)
                if score < best_score:
                    best_combo = (i, i+1)
                    best_score = score

            score = self._mlm(1, 0)
            if score < best_score:
                best_combo = (1, 0)

            self.counts, self.bin_edges = self._merge(*best_combo)

            self.n_items -= 1

            if self.n_items in self.map_at:
                self.tracker[str(self.n_items)][:] = self.bin_edges
            if not self.utilize_brute_force and self.n_items == self.brute_force_at:
                self.utilize_brute_force = True
            pbar.update(1)

        if self.tracker is not None:
            self.tracker.close()
        return self.bin_edges


class MergerNonlocal(Merger):
    """A merger that merges bins non-locally. Bin edges are irrelevant here."""
    def __init__(
            self,
            bin_edges,
            *counts,
            weights=None,
            comp_to_first=False,
            map_at = None,
            brute_force_at=10,
            file_prefix=""
        ) -> None:
        """The initializer for the non-local merging class

        Parameters
        ----------
        bin_edges : numpy.ndarray
            These are the edges of your histogram that correspond to physical quantities
        counts : numpy.ndarray
            A series of arrays that correspond to the number of events between your bin edges
        weights : numpy.ndarray, optional
            An array of the weights associated for each of the counts. 
            If none are provided, the weights will be 1, by default None
        comp_to_first : bool, optional
            Whether you would like to compare all samples to the first one 
            provided, as opposed to all of them to each other, by default False
        map_at : list, optional
            A list of bin numbers at which you would like the mapping from 
            the original sample to be recorded, by default None
        brute_force_at : int, optional
            A value at or below which the merger will utilize the "brute-force" approach 
            of merging by calculating a total ROC score, by default 10
        file_prefix : str, optional
            This is the prefix that comes before the file bin map before "_tracker.hdf5"

        Raises
        ------
        ValueError
            The dimension of the bin edges can only be 1-dimensional
        """

        unrolled_counts = list(map(np.ndarray.ravel, map(np.array, counts)))
        unrolled_bins = np.arange(len(unrolled_counts[0]) + 1)

        super().__init__(
            unrolled_bins, *unrolled_counts,
            weights=weights, comp_to_first=comp_to_first,
            map_at=map_at, brute_force_at=brute_force_at
            )

        self._merger_type = "Non-local"

        self.physical_bins = np.array(bin_edges)

        if len(self.physical_bins.shape) > 1:
            self.n_observables = bin_edges.shape[0]
        else:
            self.n_observables = 1

        self.scores = np.zeros((self.n_items, self.n_items), dtype=np.float64)
        self.things_to_recalculate = tuple(range(self.n_items))

        if any(self.map_at):
            self.__cur_iteration_tracker = dict(
                zip(
                    self.things_to_recalculate,
                    [(i,) for i in self.things_to_recalculate]
                    )
            )
            self.tracker =  h5py.File(
                f".{file_prefix}_tracker.hdf5", 'w',
                libver='latest', driver=None,
                )

            for mapped_bincount in self.map_at:
                self.tracker.create_dataset(
                    str(mapped_bincount), (self.original_n_items), np.uint32,
                    compression='gzip', compression_opts=9,
                    fletcher32=True, fillvalue=0,
                    maxshape=(self.original_n_items), shuffle=True
                )

                np.save(f".{file_prefix}_physical_bins.npy", self.physical_bins,
                        fix_imports=False, allow_pickle=False)
        else:
            self.__cur_iteration_tracker = {}
            self.tracker = None


    def _merge(self, i, j):
        """Merge bins i and j together
        The merged bins are placed at the end of the new array
        

        Parameters
        ----------
        i : int
            The first index to merge
        j : int
            The second index to merge
        """
        k = 0
        sum_term = np.zeros(self.n_hypotheses)
        if self.tracker is not None:
            #add tuples that contain original indices within the key of the new indices
            old_tracker_entries = self.__cur_iteration_tracker[i] + self.__cur_iteration_tracker[j]

        for c in np.arange(self.n_items):
            if c not in (i, j):
                #if c == k then you don't need to do anything! 
                #The bins pre and post merge will be the same
                if c != k: 
                    self.counts[:, k] = self.counts[:, c]
                    self.scores[:, k], self.scores[k] = self.scores[:, c], self.scores[c]

                    if self.tracker is not None:
                        self.__cur_iteration_tracker[k] = self.__cur_iteration_tracker[c]

                k += 1
            else:
                # add the merged terms to an accumulator
                sum_term += self.counts[:, c]

        #last entry of the new array is the sum of the merged terms
        self.counts[:, k] = sum_term.T
        self.counts = self.counts[:, :-1]

        self.scores = self.scores[:k+1, :k+1]

        self.scores[k] = np.inf
        self.scores[:, k] = np.inf

        self.things_to_recalculate = (k, )

        if self.tracker is not None:
            self.__cur_iteration_tracker[k] = old_tracker_entries
            del self.__cur_iteration_tracker[k+1]


    def __closest_pair(self):
        """Recalculates the scores that it needs to
        according to self.things_to_recalculate
        and picks the smallest score from
        the score matrix

        Returns
        -------
        tuple[int, int]
            A 2d index for self.scores
        """
        for i in np.arange(self.n_items, dtype=np.int64):
            for j in self.things_to_recalculate:
                if i == j:
                    self.scores[i][j] = np.inf
                    continue
                self.scores[i][j] = self._mlm(i, j)

        smallest_distance_index = np.unravel_index(self.scores.argmin(), self.scores.shape)

        return smallest_distance_index


    def __convert_tracker(self):
        """Converts the internal tracker into a bin map
        That maps original bin indices to new ones

        Returns
        -------
        numpy.ndarray
            A 1-d array where the each element holds the new index of that index
            (i.e. the $i^{th}$ element of the array contains a value j,
            that value j is the new placement for any element that would
            land in i for the original binning)
        """
        new_map = np.empty(self.original_n_items)
        for new_place, original_placement in self.__cur_iteration_tracker.items():
            for original_place in original_placement:
                new_map[original_place] = new_place

        return new_map


    def run(self, target_bin_number=2):
        """Runs the merger

        Parameters
        ----------
        target_bin_number : int, optional
            The number of bins you would like to merge down to, by default 2

        Returns
        -------
        numpy.ndarray
            Returns an array containing all the new counts for the nonlocal array
            since bin edges are now meaningless. The array has shape
            (# samples, target_bin_number)
        """
        if self.n_items <= target_bin_number:
            warnings.warn("Merging is pointless! Number of bins already >= target")

        pbar = tqdm.tqdm(
            total=self.n_items - target_bin_number,
            desc="Binning non-locally:", leave=True, position=0
            )
        while self.n_items > target_bin_number:
            min_1, min_2 = self.__closest_pair()
            self._merge(min_1, min_2)

            self.n_items -= 1
            if self.n_items in self.map_at:
                self.tracker[str(self.n_items)][:] = self.__convert_tracker()

            if not self.utilize_brute_force and self.n_items == self.brute_force_at:
                self.utilize_brute_force = True

            if self.utilize_brute_force:
                self.things_to_recalculate = tuple(range(self.n_items))

            pbar.update(1)

        if self.tracker is not None:
            self.tracker.close()
        return self.counts
