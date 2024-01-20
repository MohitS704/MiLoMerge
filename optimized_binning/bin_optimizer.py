import warnings
import numpy as np
import tqdm
import numba as nb
import dask.array as da
import time

class Merger():
    def __init__(
            self,
            bin_edges,
            *counts,
            weights=None,
            comp_to_first=False
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

        self.bin_edges = np.array(bin_edges)

    @staticmethod
    @nb.njit(fastmath=True, cache=True)
    def __mlm_driver_comp_to_first(n, counts, weights, b, b_prime):

        metric_val = 0
        for h_prime in nb.prange(h+1, n):
            t1, t2 = counts[h][b]*weights[h][h_prime], counts[h_prime][b_prime]*weights[h][h_prime]
            t3, t4 = counts[h][b_prime]*weights[h][h_prime], counts[h_prime][b]*weights[h][h_prime] 

            metric_val += (t1*t2)**2 + (t3*t4)**2 - 2*t1*t2*t3*t4

        return metric_val

    @staticmethod
    @nb.njit(fastmath=True, cache=True)
    def __mlm_driver_comp_to_all(n, counts, weights, b, b_prime):

        metric_val = 0
        for h in np.arange(n):
            for h_prime in nb.prange(h+1, n):
                t1, t2 = counts[h][b]*weights[h][h_prime], counts[h_prime][b_prime]*weights[h][h_prime]
                t3, t4 = counts[h][b_prime]*weights[h][h_prime], counts[h_prime][b]*weights[h][h_prime] 

                metric_val += (t1*t2)**2 + (t3*t4)**2 - 2*t1*t2*t3*t4

        return metric_val

    def _mlm(self, b, b_prime):
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

class MergerLocal(Merger):
    def __init__(
            self,
            bin_edges,
            *counts,
            weights=None,
            comp_to_first=False
        ) -> None:
        super().__init__(bin_edges, *counts, weights=weights, comp_to_first=comp_to_first)
        if len(self.bin_edges.shape) > 1:
            raise ValueError("LOCAL MERGING CAN ONLY HANDLE 1-DIMENSIONAL ARRAYS")

        self._merger_type = "Local"

    def _merge(self, i, j):
        if i == j + 1:
            if i > self.n_items - 1 or i < 0:
                raise ValueError("UNHANDLED EDGE CASE WHILE MERGING")
            temp_counts = np.concatenate(
                (
                    self.counts[:, :j],
                    np.array([self.counts[:, j] + self.counts[:, i]]).T,
                    self.counts[:, i+1:]
                ),
                axis=1
            )
            temp_edges = np.concatenate((self.bin_edges[:i], self.bin_edges[i+1:]))

        elif i == j - 1:
            if i > self.n_items - 1 or i < 1:
                raise ValueError("UNHANDLED EDGE CASE WHILE MERGING")
            temp_counts = np.concatenate(
                (
                    self.counts[:, :i],
                    np.array([self.counts[:, i] + self.counts[:, j]]).T,
                    self.counts[:, j+1:]
                ),
                axis=1
            )
            temp_edges = np.concatenate((self.bin_edges[:i+1], self.bin_edges[i+2:]))

        elif i == j:
            pass

        else:
            raise ValueError("This is local binning! Can only merge ahead or behind!")

        return (temp_counts, temp_edges)

    def run(self, target_bin_number=1):
        if self.n_items <= target_bin_number:
            warnings.warn("Merging is pointless! Number of bins already >= target")

        pbar = tqdm.tqdm(
            total=self.n_items - target_bin_number,
            desc="Binning locally:", leave=True, position=0
            )
        while self.n_items > target_bin_number:
            combinations = {}
            scores = {}

            for i in range(1, self.n_items - 1):
                score = self._mlm(i, i+1)

                combinations[i] = (i, i+1)
                scores[i] = score

            score = self._mlm(1, 0)
            #try merging counts 1 with count 0 (backwards)
            combinations[0] = (1, 0)
            scores[0] = score

            min_1, min_2 = combinations[
                min(scores, key=scores.get)
                ]

            self.counts, self.bin_edges = self._merge(min_1, min_2)

            self.n_items -= 1
            pbar.update(1)

        return self.bin_edges



class MergerNonlocal(Merger):
    def __init__(
            self,
            bin_edges,
            *counts,
            weights=None,
            comp_to_first=False,
            tracker_size=50
        ) -> None:

        unrolled_counts = list(map(np.ndarray.ravel, map(np.array, counts)))
        unrolled_bins = np.arange(len(unrolled_counts[0]) + 1)

        super().__init__(
            unrolled_bins, *unrolled_counts,
            weights=weights, comp_to_first=comp_to_first
            )

        self._merger_type = "Non-local"

        self.physical_bins = da.array(bin_edges)

        if len(self.physical_bins.shape) > 1:
            self.n_observables = bin_edges.shape[0]
        else:
            self.n_observables = 1

        self.__cur_iteration_tracker = {}
        self.scores = np.zeros((self.n_items, self.n_items), dtype=np.float64)
        self.things_to_recalculate = tuple(range(self.n_items))


    @staticmethod
    @nb.njit
    def _merge(n_orig, n, counts, score_matrix, i, j):
        merged_counts = np.zeros(shape=(n_orig, n - 1), dtype=np.float64)

        # k = 0
        # for c in nb.prange(n):
        #     if c not in (i, j):
        #         merged_counts[:, k] = counts[:, c]
        #         score_matrix[:, k], score_matrix[k] = score_matrix[:, n], score_matrix[n]

        #         k += 1
        merged_counts[:, :-1] = np.delete(counts, [i,j])
        merged_counts[:, -1] = counts[:, i] + counts[:, j]

        score_matrix = np.delete(np.delete(score_matrix, [i, j], axis=0), [i, j], axis=1)
        # score_matrix[k:] = score_matrix[:, k:] = np.inf

        return counts, score_matrix, k

    @staticmethod
    @nb.njit
    def __closest_pair_driver(n, counts, weights, things_to_recalculate, score_matrix, comp_to_first):
        for i in nb.prange(n):
            for j in nb.prange(things_to_recalculate):
                if i == j:
                    score_matrix[i][j] = np.inf
                    continue
                score_matrix[i][j] = Merger._mlm(n, counts, weights, i, j, comp_to_first)
        return score_matrix

    def _closest_pair(self):
        self.scores = self.__closest_pair_driver(self.n_items, self.counts, self.weights, self.things_to_recalculate, self.scores, self.comp_to_first)
        smallest_distance_index = np.unravel_index(self.scores.argmin(), self.scores.shape)
        smallest_distance = self.scores[smallest_distance_index]

        if not np.isfinite(smallest_distance):
            raise ValueError("Distance function has produced nan/inf at some point")

        return smallest_distance, smallest_distance_index
