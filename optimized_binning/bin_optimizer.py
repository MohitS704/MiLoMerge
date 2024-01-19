import warnings
import numpy as np
import tqdm
import numba as nb
import dask.array as da

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

        self.weights = np.outer(weights, weights)
        self.n_hypotheses = len(counts)
        self.n_items = len(counts[0])
        #self.n is the number of hypotheses, self.n_items is the current number of bin_edges

        self.comp_to_first = comp_to_first

        self.counts = da.vstack(counts)
        self.counts /= self.counts.sum(axis=1)[:, None]

        self.bin_edges = da.from_array(bin_edges)

    @staticmethod
    @nb.njit(parallel=True)
    def _mlm(n, counts, weights, b, b_prime, comp_to_first):
        metric_val = 0
        if comp_to_first:
            initial_range = (0,)
        else:
            comp_to_first = nb.prange(n)

        for h in initial_range:
            for h_prime in nb.prange(h+1, n):

                mat = np.array([
                    [counts[h][b], counts[h][b_prime]],
                    [counts[h_prime][b], counts[h_prime][b_prime]]
                ], dtype=np.float64)

                mat *= weights[h][h_prime]

                metric_val += (mat[0][0]*mat[1][1])**2 + (mat[0][1]*mat[1][0])**2 - 2*np.prod(mat)
        return metric_val

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

    @staticmethod
    @nb.njit
    def _merge(n, counts, bin_edges, i, j):
        if i == j + 1:
            if i > n - 1 or i < 0:
                raise ValueError("UNHANDLED EDGE CASE WHILE MERGING")
            temp_counts = np.concatenate(
                (
                    counts[:, :j],
                    np.array([counts[:, j] + counts[:, i]]).T,
                    counts[:, i+1:]
                ),
                axis=1
            )
            temp_edges = np.concatenate((bin_edges[:i], bin_edges[i+1:]))

        elif i == j - 1:
            if i > n - 1 or i < 1:
                raise ValueError("UNHANDLED EDGE CASE WHILE MERGING")
            temp_counts = np.concatenate(
                (
                    counts[:, :i],
                    np.array([counts[:, i] + counts[:, j]]).T,
                    counts[:, j+1:]
                ),
                axis=1
            )
            temp_edges = np.concatenate((bin_edges[:i+1], bin_edges[i+2:]))

        elif i == j:
            pass

        else:
            raise ValueError("This is local binning! Can only merge ahead or behind!")

        return (temp_counts, temp_edges)

    def run(self, target_bin_number=1):
        if self.n_hypotheses <= target_bin_number:
            warnings.warn("Merging is pointless! Number of bins already >= target")

        pbar = tqdm.tqdm(
            total=self.n_items - target_bin_number,
            desc="Binning locally:", leave=True, position=0
            )
        while self.n_items > target_bin_number:
            combinations = {}
            scores = {}

            def sweep_back_and_forth(i):
                score = self._mlm(self.n_hypotheses, self.counts,
                                  self.weights, i, i+1, self.comp_to_first)

                combinations[i] = (i, i+1)
                scores[i] = score

            _ = [sweep_back_and_forth(i) for i in range(1, self.n_items - 1)]
            score = self._mlm(
                self.n_hypotheses, self.counts, self.weights,
                1, 0, self.comp_to_first
                )
            #try merging counts 1 with count 0 (backwards)
            combinations[0] = (1, 0)
            scores[0] = score

            min_1, min_2 = combinations[
                min(scores, key=scores.get)
                ]

            self.counts, self.bin_edges = self._merge(self.n_items, self.counts,
                                                      self.bin_edges, min_1, min_2)

            pbar.update(1)




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

        self.physical_bins = da.array(bin_edges)

        if len(self.physical_bins.shape) > 1:
            self.n_observables = bin_edges.shape[0]
        else:
            self.n_observables = 1

        self.__cur_iteration_tracker = {}
        self.scores = da.zeros((self.n_items, self.n_items), dtype=np.float64)


    @staticmethod
    @nb.njit
    def _merge(n_orig, n, counts, score_matrix, i, j):
        merged_counts = da.zeros(shape=(n_orig, n - 1), dtype=np.float64)

        k = 0
        for c in nb.prange(n):
            if c not in (i, j):
                merged_counts[:, k] = counts[:, c]
                score_matrix[:, k], score_matrix[k] = score_matrix[:, n], score_matrix[n]

                k += 1
        merged_counts[:, k] = counts[:, i] + counts[:, k]
        score_matrix[k:] = score_matrix[:, k:] = np.inf

        return counts, score_matrix, k

    @staticmethod
    @nb.njit
    def _closest_pair(n, counts, weights, things_to_recalculate, score_matrix, comp_to_first):
        for i in nb.prange(n):
            for j in nb.prange(things_to_recalculate):
                if i == j:
                    score_matrix[i][j] = np.inf
                    continue
                score_matrix[i][j] = Merger._mlm(n, counts, weights, i, j, comp_to_first)

        smallest_distance_index = np.unravel_index(score_matrix.argmin(), score_matrix.shape)
        smallest_distance = score_matrix[smallest_distance_index]

        if not np.isfinite(smallest_distance):
            raise ValueError("Distance function has produced nan/inf at some point")

        return smallest_distance, smallest_distance_index
