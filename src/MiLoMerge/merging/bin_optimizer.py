import warnings
import os

from abc import ABC, abstractmethod
import numpy as np
import tqdm
import numba as nb
import h5py
from collections.abc import Iterable
import typing


@nb.njit(
    "float64(int64, Array(float64, 2, 'C'), Array(float64, 2, 'C'), int64, int64, b1)",
    cache=True,
    parallel=False,
)
def mlm_driver(
    n: int,
    counts: Iterable[Iterable[float]],
    weights: Iterable[Iterable[float]],
    b: int,
    b_prime: int,
    comp_to_first: bool,
) -> float:
    """This method uses the MLM metric to compare all samples to each other
    and issue a score between the two bin indices.

    Parameters
    ----------
    n : int
        The number of samples being dealt with
    counts : numpy.ndarray
        An ndarray of the counts with shape (#samples, #bins)
    weights : numpy.ndarray
        An ndarray of per-sample weight with size (#samples)
    b : int
        The index of the first bin that is being calculated
    b_prime : int
        The index of the second bin that is being calculated
    comp_to_first : bool
        Whether comparisons are only done between every sample and sample 0

    Returns
    -------
    float
        The score, as prescribed by the MLM metric
    """

    metric_val = 0
    h_range = range(1) if comp_to_first else range(n)
    for h in h_range:
        t1 = counts[h, b]
        t3 = counts[h, b_prime]
        for h_prime in range(h + 1, n):
            t2 = counts[h_prime, b_prime]
            t4 = counts[h_prime, b]

            metric_val += (t1 * t2 - t3 * t4) ** 2 * weights[h, h_prime] ** 4

    return metric_val

@nb.njit(
    "float64(int64, Array(float64, 2, 'C'), Array(float64, 2, 'C'), int64, int64)",
    cache=True,
    parallel=False,
)
def mlm_driver_revised(
    n: int,
    counts: Iterable[Iterable[float]],
    weights: Iterable[Iterable[float]],
    b: int,
    b_prime: int,
) -> float:
    t1 = t2 = t3 = t4 = t5 = t6 = 0

    for h in range(n):
        cHB_sq = counts[h,b] * counts[h,b]
        cHBp_sq = counts[h,b_prime] * counts[h,b_prime]
        w_hh = weights[h,h] * weights[h,h]
        
        t1 += cHB_sq * w_hh
        t2 += cHBp_sq * w_hh
        t3 += (counts[h,b] + counts[h,b_prime])**2 * w_hh
        
        for h_prime in range(h,n):
            w_hhp = weights[h,h_prime]

            cHpB_plus_cHpBp_sq = (counts[h_prime,b] + counts[h_prime,b_prime])**2
            cHpBp_sq = counts[h_prime,b_prime] * counts[h_prime,b_prime]

            t4 += cHB_sq*cHpBp_sq * w_hhp
            t5 += cHB_sq*cHpB_plus_cHpBp_sq * w_hhp
            t6 += cHBp_sq*cHpB_plus_cHpBp_sq * w_hhp
    
    return t1 + t2 + t3 + 2*(np.sqrt(t4) - np.sqrt(t5) - np.sqrt(t6))



@nb.njit(
    "(int64, Array(int64, 1, 'C'), Array(float64, 2, 'A'), Array(float64, 2, 'F'), int64, Array(float64, 2, 'C'), b1)",
    parallel=False,
    cache=True,
)
def _closest_pair_driver(
    n_items: int,
    things_to_recalculate: Iterable[int],
    scores: Iterable[Iterable[float]],
    counts: Iterable[Iterable[float]],
    n_hypotheses: int,
    weights: Iterable[Iterable[float]],
    comp_to_first: bool,
) -> int:
    h_range = range(1) if comp_to_first else range(n_hypotheses)

    for i in range(n_items):
        for j_idx in range(len(things_to_recalculate)):
            j = things_to_recalculate[j_idx]
            if i == j:
                scores[i][j] = np.inf
                continue
            # metric_val = 0
            # for h in h_range:
            #     t1 = counts[h, i]
            #     t3 = counts[h, j]
            #     for h_prime in range(h + 1, n_hypotheses):
            #         t2 = counts[h_prime, j]
            #         t4 = counts[h_prime, i]

            #         term = t1 * t2 - t3 * t4
            #         w = weights[h, h_prime]
            #         metric_val += (term * term) * (w * w * w * w)

            if comp_to_first:
                scores[i][j] = mlm_driver(
                    n_hypotheses, counts, weights, i, j, False
                )
            else:
                scores[i][j] = mlm_driver_revised(
                    n_hypotheses, counts, weights, i, j
                )

    return np.argmin(scores)


class Merger(ABC):
    """Abstract class that serves as a baseplate for both the local and nonlocal merger"""

    def __init__(
        self,
        bin_edges: Iterable[float],
        *counts: Iterable[float],
        weights: typing.Union[int, Iterable[float], Iterable[Iterable[float]]] = None,
        comp_to_first: bool = False,
        map_at: Iterable[int] = None,
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
            errortext = "Not all counts have same length! Lengths are: " + errortext
            raise ValueError("\n" + errortext)

        if len(counts[0]) != len(bin_edges) - 1:
            errortext = f"len(counts) = {len(counts[0])} != len(bin_edges) - 1 = {len(bin_edges)-1}"
            raise ValueError("\n" + errortext)

        if weights is not None and len(weights) != len(counts):
            errortext = (
                "The # of weight values and the # of samples should be the same!"
            )
            errortext += (
                f"\nThere are {len(counts)} samples and {len(weights)} weight values"
            )
            raise ValueError("\n" + errortext)

        if weights is None:
            weights = np.ones(len(counts), dtype=np.float64)
        else:
            weights = np.array(weights, dtype=np.float64)

        self._merger_type = None

        if len(weights.shape) > 1:
            if (weights.shape[0] != weights.shape[1]) or (len(weights.shape) > 2):
                raise ValueError("Weights must be either a 1d array of # samples or a 2d square matrix of the same dimension!")
            self.weights = np.copy(weights)
        else:
            self.weights = np.outer(weights, weights)

        self.n_hypotheses = len(counts)
        self.n_items = self.original_n_items = len(counts[0])
        # self.n is the number of hypotheses, self.n_items is the current number of bin_edges

        self.comp_to_first = comp_to_first

        self.counts = np.vstack(counts).astype(np.float64)
        # self.counts[~np.isfinite(self.counts)] = 0

        self.bin_edges = np.array(bin_edges, dtype=np.float64)

        if map_at is None:
            map_at = []
        else:
            try:
                iter(map_at)
            except Exception as e:
                raise TypeError("Parameter map_at must be an iterable!") from e

        # sets are nice and hashed
        self.map_at = set([i for i in map_at if i < self.n_items])

    def __repr__(self) -> str:
        """Representation of the merger object

        Returns
        -------
        str
            A brief summary of the merger object
        """
        return f"Merger of type {self._merger_type} merging {self.original_n_items}"

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
        bin_edges: Iterable[float],
        *counts: Iterable[float],
        weights: Iterable[float] = None,
        comp_to_first: bool = False,
        map_at: Iterable[int] = None,
        file_path: str = "./",
        file_name: str = "",
    ) -> None:
        """The initializer for the local merging class

        Parameters
        ----------
        bin_edges : numpy.ndarray
            These are the edges of your histogram that correspond to physical quantities
        counts : numpy.ndarray
            A series of arrays that correspond to the number of events between your bin edges.
        weights : numpy.ndarray, optional
            An array of the weights associated for each of the counts.
            These weights can be either a 2d array comparing weights pair-wise,
            or a 1-d array for a flat weight per hypothesis.
            If none are provided, the weights will be 1, by default None
        comp_to_first : bool, optional
            Whether you would like to compare all samples to the first one
            provided, as opposed to all of them to each other, by default False
        map_at : list, optional
            A list of bin numbers at which you would like the mapping from
            the original sample to be recorded, by default None
        file_path : str, optional
            The directory to place "_tracker.hdf5" should mapping be desired,
            by default "./"
        file_name : str, optional
            The file prefix before "_tracker.hdf5" to identify this mapping,
            by default ""

        Raises
        ------
        ValueError
            The dimension of the bin edges can only be 1-dimensional
        NotADirectoryError
            If file_path is not a valid directory, raise an error
        """
        super().__init__(
            bin_edges,
            *counts,
            weights=weights,
            comp_to_first=comp_to_first,
            map_at=map_at,
        )

        if self.bin_edges.ndim > 1:
            raise ValueError("LOCAL MERGING CAN ONLY HANDLE 1-DIMENSIONAL ARRAYS")

        self._merger_type = "Local"

        if any(self.map_at):
            if not os.path.exists(file_path):
                raise NotADirectoryError(f"{file_path} is not a valid directory!")
            if file_path[-1] != "/":
                file_path += "/"

            self.tracker = h5py.File(
                f"{file_path}{file_name}_tracker.hdf5",
                "w",
                libver="latest",
                driver=None,
            )

            for mapped_bincount in self.map_at:
                self.tracker.create_dataset(
                    str(mapped_bincount),
                    (mapped_bincount + 1),
                    np.float64,
                    compression="gzip",
                    compression_opts=9,
                    fletcher32=True,
                    fillvalue=0,
                    maxshape=len(bin_edges),
                    shuffle=True,
                )
        else:
            self.tracker = None

    @staticmethod
    @nb.njit(
        "(Array(float64, 2, 'C'), Array(float64, 1, 'C'), int64, int64)",
        cache=True,
        fastmath=True,
    )
    def __merge_driver(
        counts: Iterable[Iterable[float]],
        bin_edges: Iterable[float],
        first_part: int,
        second_part: int,
    ):
        """This method is the numba-ified function that is called by __merge.
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
        tuple[numpy.ndarray, numpy.ndarray]
            A tuple of the new counts with two bins merged
            and a 1-d array of the new bin edges with one of the bin edges removed
        """
        new_counts = counts[:, :-1].copy()

        new_counts[:, first_part] = (counts[:, first_part] + counts[:, second_part]).T
        new_counts[:, first_part + 1 :] = counts[:, second_part + 1 :]

        new_bin_edges = bin_edges[:-1].copy()
        new_bin_edges[first_part + 1 :] = bin_edges[second_part + 1 :]

        return (new_counts, new_bin_edges)

    def __merge(self, i: int, j: int):
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
            raise ValueError("TRYING TO MERGE THE SAME INDEX!")
            # return (self.counts, self.bin_edges)

        else:
            raise ValueError("This is local binning! Can only merge ahead or behind!")

    def run(self, target_bin_number: int, return_counts: bool = False):
        """Runs the merger

        Parameters
        ----------
        target_bin_number : int
            The number of bins you would like to merge down to

        return_counts : bool, optional
            Whether the returned value will have the bin counts
            returned alongside the bin edges, by default False

        Returns
        -------
        numpy.ndarray or tuple(numpy.ndarray, numpy.ndarray)
            Returns the 1-d bin edges that would
            correspond to the best binning for the number of bins you want.
            If return_counts is set to true, also return the internal final bin counts,
            which have shape (#samples, target_bin_number).
        """
        if self.n_items <= target_bin_number:
            warnings.warn("Merging is pointless! Number of bins already >= target")

        pbar = tqdm.tqdm(
            total=self.n_items - target_bin_number,
            desc="Binning locally:",
            leave=True,
            position=0,
        )

        things_to_recalculate = range(1, self.n_items - 1)

        scores = np.full((self.n_items - 1, 3), np.inf, dtype=np.float64)
        # stores the 2 indices being merged and the score as a matrix

        while self.n_items > target_bin_number:
            for i in things_to_recalculate:
                scores[i] = (
                    i,
                    i + 1,
                    mlm_driver(
                        self.n_hypotheses,
                        self.counts,
                        self.weights,
                        i,
                        i + 1,
                        self.comp_to_first,
                    ),
                )
                if i == 1:
                    scores[0] = (
                        0,
                        1,
                        mlm_driver(
                            self.n_hypotheses,
                            self.counts,
                            self.weights,
                            1,
                            0,
                            self.comp_to_first,
                        ),
                    )

            # The last column is the scores, and the first 2 the indices
            i1, i2 = scores[np.argmin(scores[:, 2])][:-1].astype(int)

            # remove the merged index and move all indices down 1
            scores[:, :-1][i1:] -= 1
            scores = np.delete(scores, i1, axis=0)

            if i1 == 0:
                things_to_recalculate = (0, 1)
                i1, i2 = i2, i1
            elif i1 == self.n_items - 2:
                if self.n_items <= 4:
                    things_to_recalculate = (i1 - 1, i1 - 2)
                else:
                    things_to_recalculate = (i1 - 2, i1 - 3)
            else:
                things_to_recalculate = (i1 - 1, i1)

            self.counts, self.bin_edges = self.__merge(i1, i2)

            self.n_items -= 1

            if self.n_items in self.map_at:
                self.tracker[str(self.n_items)][:] = self.bin_edges
            pbar.update(1)

        if return_counts:
            return self.bin_edges, self.counts
        return self.bin_edges


class MergerNonlocal(Merger):
    """A merger that merges bins non-locally. Bin edges are irrelevant here."""

    def __init__(
        self,
        bin_edges: Iterable[float],
        *counts: Iterable[float],
        weights: Iterable[float] = None,
        comp_to_first: bool = False,
        map_at: Iterable[int] = None,
        file_path: str = "./",
        file_name: str = "",
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
        file_path : str, optional
            The directory to place "_tracker.hdf5" should mapping be desired,
            by default "./"
        file_name : str, optional
            The file prefix before "_tracker.hdf5" to identify this mapping,
            by default ""

        Raises
        ------
        ValueError
            The dimension of the bin edges can only be 1-dimensional
        """

        unrolled_counts = list(map(np.ndarray.ravel, map(np.array, counts)))
        sizes = np.array([len(i) for i in unrolled_counts])
        if np.any(sizes != sizes[0]):
            raise ValueError(
                f"All arrays must be the same length! Your unrolled arrays are of length: {sizes}"
            )

        for nd_arr in counts:
            nd_arr = np.array(nd_arr)
            if len(nd_arr.shape) > 1:
                for i, size in enumerate(nd_arr.shape):
                    if i == len(bin_edges):
                        raise IndexError(f"No bin edges provided for dimension {i}!")
                    if len(bin_edges[i]) != size + 1:
                        raise ValueError(
                            f"Bin edge for dimension {i} is of invalid size {len(bin_edges[i])} != {size + 1}!"
                        )
            else:
                if len(bin_edges) != len(counts[0]) + 1:
                    raise ValueError(
                        f"Bin edges are of invalid size {len(bin_edges)} != {len(counts[0]) + 1}!"
                    )

        unrolled_bins = np.arange(len(unrolled_counts[0]) + 1)

        super().__init__(
            unrolled_bins,
            *unrolled_counts,
            weights=weights,
            comp_to_first=comp_to_first,
            map_at=map_at,
        )

        self._merger_type = "Non-local"
        self.counts = np.asfortranarray(self.counts)

        self.physical_bins = np.array(bin_edges, dtype=object)
        self.n_observables = len(bin_edges)

        self.scores = np.zeros((self.n_items, self.n_items), dtype=np.float64)
        self.things_to_recalculate = np.arange(self.n_items, dtype=int)

        self.__cur_iteration_tracker = nb.typed.Dict()
        for i in self.things_to_recalculate:
            self.__cur_iteration_tracker[i] = np.array([i], dtype=np.int64)

        if any(self.map_at):
            if not os.path.isdir(file_path):
                raise NotADirectoryError(f"{file_path} is not a valid directory!")
            if file_path[-1] != "/":
                file_path += "/"

            tracker_name = f"{file_path}{file_name}_tracker.hdf5"
            bins_name = f"{file_path}{file_name}_physical_bins.npy"

            if os.path.exists(tracker_name):
                os.system(f"rm {tracker_name} {bins_name}")

            self.tracker = h5py.File(
                tracker_name,
                "w",
                libver="latest",
                driver=None,
            )

            for mapped_bincount in self.map_at:
                self.tracker.create_dataset(
                    str(mapped_bincount),
                    (self.original_n_items),
                    np.uint32,
                    compression="gzip",
                    compression_opts=9,
                    fletcher32=True,
                    fillvalue=0,
                    maxshape=(self.original_n_items),
                    shuffle=True,
                )

                np.save(
                    bins_name, self.physical_bins, fix_imports=False, allow_pickle=True
                )
        else:
            self.tracker = None

    @staticmethod
    @nb.njit(
        "(int64, int64, Array(float64, 2, 'F'), Array(float64, 2, 'A'), DictType(int64, Array(int64, 1, 'C')), int64, int64)",
        cache=True,
        fastmath=True,
    )
    def __merge_driver_tracker(
        n_items, n_hypotheses, counts, scores, cur_tracker, i, j
    ):
        k = 0
        sum_term = np.zeros(n_hypotheses)
        for c in range(n_items):
            if c != i and c != j:
                # if c == k then you don't need to do anything!
                # The bins pre and post merge will be the same
                if c != k:
                    counts[:, k] = counts[:, c]
                    scores[:, k], scores[k] = scores[:, c], scores[c]

                    cur_tracker[k] = cur_tracker[c]

                k += 1
            else:
                # add the merged terms to an accumulator
                sum_term += counts[:, c]

        counts[:, k] = sum_term.T
        counts = counts[:, :-1:1]
        scores = scores[: k + 1, : k + 1]
        scores[k] = np.inf
        scores[:, k] = np.inf

        return k, counts, scores

    @staticmethod
    @nb.njit("(DictType(int64, Array(int64, 1, 'C')), int64)", cache=True)
    def __convert_tracker(cur_tracker, original_n_items):
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
        new_map = np.zeros(original_n_items, dtype=np.int64)
        for new_place, original_placement in cur_tracker.items():
            for original_place in original_placement:
                new_map[original_place] = new_place

        return new_map

    def run(self, target_bin_number: int):
        """Runs the merger

        Parameters
        ----------
        target_bin_number : int, optional
            The number of bins you would like to merge down to

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
            desc="Binning non-locally:",
            leave=True,
            position=0,
        )
        while self.n_items > target_bin_number:

            ###### FIND BINS TO MERGE ######
            min_arg = _closest_pair_driver(
                self.n_items,
                self.things_to_recalculate,
                self.scores,
                self.counts,
                self.n_hypotheses,
                self.weights,
                self.comp_to_first,
            )
            min_1, min_2 = np.unravel_index(min_arg, self.scores.shape)

            ##### MERGE STEP ######
            if self.tracker is not None:
                # add tuples that contain original indices within the key of the new indices
                old_tracker_entries = np.concatenate(
                    (
                        self.__cur_iteration_tracker[min_1],
                        self.__cur_iteration_tracker[min_2],
                    )
                )

            k, self.counts, self.scores = self.__merge_driver_tracker(
                self.n_items,
                self.n_hypotheses,
                self.counts,
                self.scores,
                self.__cur_iteration_tracker,
                min_1,
                min_2,
            )
            self.things_to_recalculate = np.array(
                [
                    k,
                ],
                dtype=int,
            )

            if self.tracker is not None:
                self.__cur_iteration_tracker[k] = old_tracker_entries

            del self.__cur_iteration_tracker[
                k + 1
            ]  # in the end k is the number of entries left over

            ##### UPDATE STATE ######
            self.n_items -= 1
            if self.n_items in self.map_at:
                self.tracker[str(self.n_items)][:] = self.__convert_tracker(
                    self.__cur_iteration_tracker, self.original_n_items
                )

            pbar.update(1)

        if self.tracker is not None:
            self.tracker.close()
        return self.counts
