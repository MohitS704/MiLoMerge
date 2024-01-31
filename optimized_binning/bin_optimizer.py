import warnings
from abc import ABC, abstractmethod
import numpy as np
import tqdm
import numba as nb
import h5py
from ROC_curves import ROC_score

class Merger(ABC):
    def __init__(
            self,
            bin_edges,
            *counts,
            weights=None,
            comp_to_first=False,
            map_at=None,
            brute_force_at=10
        ) -> None:

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
            errortext = "The # of weight values and the # of hypotheses should be the same!"
            errortext += f'\nThere are {len(counts)} hypotheses and {len(weights)} weight values'
            raise ValueError('\n'+errortext)

        if weights is None:
            weights = np.ones(len(counts), dtype=float)
        else:
            weights = np.array(weights)

        self._merger_type = None

        self.weights = np.outer(weights, weights)
        self.n_hypotheses = len(counts)
        self.n_items = self.original_n_items = len(counts[0])
        #self.n is the number of hypotheses, self.n_items is the current number of bin_edges

        self.comp_to_first = comp_to_first

        self.counts = np.vstack(counts)
        self.counts /= self.counts.sum(axis=1)[:, None]
        self.counts[~np.isfinite(self.counts)] = 0

        self.bin_edges = np.array(bin_edges)

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

    # @staticmethod
    # @nb.njit(nb.float64(nb.float64[:], nb.float64[:]), fastmath=True, cache=True)
    # def ROC_score(hypo_1, hypo_2):
    #     raw_ratio = hypo_1.copy()/hypo_2
    #     ratio_indices = np.argsort(raw_ratio)
    #     # ratio_indices = np.isfinite(raw_ratio[ratio_indices])[::-1]

    #     length = len(ratio_indices) + 1

    #     TPR = np.zeros(length)
    #     FPR = np.zeros(length)

    #     for n in nb.prange(length):
    #         above_cutoff = ratio_indices[n:]
    #         below_cutoff = ratio_indices[:n]

    #         TPR[n] = hypo_1[above_cutoff].sum()/(
    #             hypo_1[above_cutoff].sum() + hypo_1[below_cutoff].sum()
    #             ) #gets the indices listed

    #         FPR[n] = hypo_2[below_cutoff].sum()/(
    #             hypo_2[above_cutoff].sum() + hypo_2[below_cutoff].sum()
    #             )

    #     return np.trapz(TPR, FPR) - 0.5
    
    @staticmethod
    @nb.njit(fastmath=True, cache=True, nogil=True)
    def __ROC_score_comp_to_first(counts, weights, b, b_prime):
        ROC_summation = 0
        h = 0
        h_temp_added_counts = counts[h].copy()
        h_temp_added_counts[b] += counts[h][b_prime]
        h_temp_added_counts = h_temp_added_counts.delete(b_prime)

        for h_prime, h_prime_counts in enumerate(counts[1:]):
            h_prime += 1
            
            h_prime_counts[b] += h_prime_counts[b_prime]
            h_prime_counts = h_prime_counts.delete(b_prime)
            
            ROC_summation += ROC_score(h_temp_added_counts, h_prime_counts)*weights[h, h_prime]

        return 1/ROC_summation

    @staticmethod
    @nb.njit(fastmath=True, cache=True, nogil=True)
    def __mlm_driver_comp_to_first(n, counts, weights, b, b_prime):

        metric_val = 0
        for h_prime in np.arange(1, n, dtype=np.int64):
            t1 = counts[0, b]*weights[0, h_prime]
            t2 = counts[h_prime, b_prime]*weights[0, h_prime]
            t3 = counts[0, b_prime]*weights[0, h_prime]
            t4 = counts[h_prime, b]*weights[0, h_prime]

            metric_val += (t1*t2)**2 + (t3*t4)**2 - 2*t1*t2*t3*t4

        return metric_val


    @staticmethod
    @nb.njit(fastmath=True, cache=True, nogil=True)
    def __ROC_score_comp_to_all(counts, weights, b, b_prime):
        roc_summation = 0
        for h, h_counts in enumerate(counts):
            h_counts[b] += h_counts[h][b_prime]
            h_counts = h_counts.delete(b)
            for h_prime, h_prime_counts in enumerate(counts[h+1:]):
                h_prime += h+1
                
                merged_quantity_hprime = h_prime_counts[b_prime]
                h_prime_counts[b] += merged_quantity_hprime
                h_prime_counts = h_prime_counts.delete(b_prime)
                
                roc_summation += ROC_score(h_counts, h_prime_counts)*weights[h, h_prime]

        return 1/roc_summation

    @staticmethod
    @nb.njit(fastmath=True, cache=True, nogil=True)
    def __mlm_driver_comp_to_all(n, counts, weights, b, b_prime):

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
        if self.utilize_brute_force:
            if self.comp_to_first:
                return self.__ROC_score_comp_to_first(
                    self.counts, self.weights,
                    b, b_prime
                )
            else:
                return self.__ROC_score_comp_to_all(
                    self.counts, self.weights,
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
        return f"Merger of type {self._merger_type} merging {self.original_n_items}"

    @abstractmethod
    def _merge(self, i, j):
        return NotImplemented


    @abstractmethod
    def run(self, target_bin_number=1):
        return NotImplemented

class MergerLocal(Merger):
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
        super().__init__(bin_edges, *counts, weights=weights, comp_to_first=comp_to_first, map_at=map_at, brute_force_at=brute_force_at)
        if len(self.bin_edges.shape) > 1:
            raise ValueError("LOCAL MERGING CAN ONLY HANDLE 1-DIMENSIONAL ARRAYS")

        self._merger_type = "Local"

        if any(map_at):
            self.tracker =  h5py.File(f".{file_prefix}_tracker.hdf5", 'w', libver='latest', driver=None, )
            for mapped_bincount in map_at:
                self.tracker.create_dataset(
                    str(mapped_bincount), (mapped_bincount + 1), np.float64,
                    compression='gzip', compression_opts=9,
                    fletcher32=True, fillvalue=0,
                    maxshape=len(bin_edges), shuffle=True
                )
        else:
            self.tracker = None



    @staticmethod
    @nb.njit(nb.types.Tuple((nb.float64[:, :], nb.float64[:]))(nb.float64[:, :], nb.float64[:], nb.int32, nb.int32), cache=True, fastmath=True, nogil=True)
    def __merge_driver(counts, bin_edges, first_part, second_part):
        new_counts = counts[:, :-1].copy()

        new_counts[:, first_part] = (counts[:, first_part] + counts[:, second_part]).T
        new_counts[:, first_part + 1:] = counts[:, second_part + 1:]

        new_bin_edges = bin_edges[:-1].copy()
        new_bin_edges[first_part + 1:] = bin_edges[second_part + 1:]

        return (new_counts, new_bin_edges)


    def _merge(self, i, j):
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


    def run(self, target_bin_number=1):
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
            pbar.update(1)

        if self.tracker is not None:
            self.tracker.close()
        return self.bin_edges


class MergerNonlocal(Merger):
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

        if any(map_at):
            self.__cur_iteration_tracker = dict(
                zip(
                    self.things_to_recalculate,
                    [(i,) for i in self.things_to_recalculate]
                    )
            )
            self.tracker =  h5py.File(f".{file_prefix}_tracker.hdf5", 'w', libver='latest', driver=None, )
            for mapped_bincount in map_at:
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
        k = 0
        sum_term = np.zeros(self.n_hypotheses)
        if self.tracker is not None:
            old_tracker_entries = self.__cur_iteration_tracker[i] + self.__cur_iteration_tracker[j]

        for c in np.arange(self.n_items):
            if c not in (i, j):
                if c != k:
                    self.counts[:, k] = self.counts[:, c]
                    self.scores[:, k], self.scores[k] = self.scores[:, c], self.scores[c]

                    if self.tracker is not None:
                        self.__cur_iteration_tracker[k] = self.__cur_iteration_tracker[c]

                k += 1
            else:
                sum_term += self.counts[:, c]

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
        for i in np.arange(self.n_items, dtype=np.int32):
            for j in self.things_to_recalculate:
                if i == j:
                    self.scores[i][j] = np.inf
                    continue
                self.scores[i][j] = self._mlm(i, j)

        smallest_distance_index = np.unravel_index(self.scores.argmin(), self.scores.shape)

        return smallest_distance_index


    def __convert_tracker(self):
        new_map = np.empty(self.original_n_items)
        for new_place, original_placement in self.__cur_iteration_tracker.items():
            for original_place in original_placement:
                new_map[original_place] = new_place

        return new_map


    def run(self, target_bin_number=1):
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

            pbar.update(1)

        if self.tracker is not None:
            self.tracker.close()
        return self.counts, self.bin_edges[:self.n_items + 1]
