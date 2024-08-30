import warnings
import numpy as np
import awkward as ak
import tqdm
import numba as nb
import h5py
from ROC_curves import ROC_score

class Split_Node(object):
    def __init__(self, parent=None) -> None:
        if parent is None:
            self.cuts = set()
        else:
            self.cuts = parent.cuts

    def add_cut(self, var_name, cut_value):
        self.cuts.add((var_name, cut_value))

    def act_on_cuts(self, arr):
        cut_str = []
        for cut in self.cuts:
            var_name, cut_val = cut
            cut_str.append(f"(arr.{var_name} > {cut_val})")
        cut_str = " & ".join(cut_str)

class Splitter(object):
    def __init__(
        self,
        *data,
        weights=None
        ) -> None:

        if len(weights) != len(data):
            raise ValueError("data and weights should be the same length!")

        if not np.all(np.array([len(data[n]) for n in range(len(data))]) == len(data[0])):
            raise ValueError("All dictionaries must have the same length!")

        self.observables = tuple(data[0].keys())
        self.n_observables = len(self.observables)
        self.maxima_and_minima = np.zeros((self.n_observables, 2))

        if weights is None:
            weights = [np.fill(1) for i in self.n_observables]
        for n, weight in enumerate(weights):
            data[n]["w"] = weight

        self.data = {observable:None for observable in self.observables}
        for observable in self.observables:
            self.data[observable] = ak.Array(
                np.column_stack(
                    [obs_dict[observable] for obs_dict in data]
                )
            )
        self.data = ak.from_regular(ak.Array(self.data))
        del data, weights, weight, observable

        # for n, inp in enumerate(data):
        #     if not isinstance(inp, dict):
        #         raise TypeError("All inputs must be dictionaries!")
            
        #     for j, (key, value) in enumerate(inp.items()):
        #         inp[key] = ak.Array(value)
        #         if first:
        #             self.observables.append(key)
        #         self.maxima_and_minima[j][0] = min(self.maxima_and_minima[j][0], ak.min(value))
        #         self.maxima_and_minima[j][1] = max(self.maxima_and_minima[j][1], ak.max(value))
        #     first = False
        #     inp["w"] = ak.Array(weights[n])
        #     data[n] = ak.zip(inp)

        # self.observables = np.array(self.observables)
        # self.data = ak.zip({f"h{n}":data[n] for n in range(len(data))})

    def split(
        self,
        granularity=10,
        stat_limit=0.025
    ):
        possible_edges = [
            np.linspace(
                self.maxima_and_minima[i][0], 
                self.maxima_and_minima[i][0],
                granularity + 1
            )[-1:1] for i in range(self.n_observables)
        ]
        
        for i, axis_candidates in enumerate(possible_edges):
            observable = self.observables[i]
            for j, possible_bin_edge in enumerate(axis_candidates):
                
    

