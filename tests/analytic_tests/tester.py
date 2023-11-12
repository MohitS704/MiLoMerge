import numpy as np
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib.patches
import scipy.stats as sp
import mplhep as hep
import itertools
import pickle
import uproot
import time
import warnings
import sys
import os
warnings.filterwarnings("ignore")
sys.path.append('../../brunelle_merger/')
import brunelle_merger as bm

if __name__ == "__main__": #TEST DATA!!
    ################################ ANALYTIC ############################
    
    files = (
        "../../test_data/out_g1.txt",
        "../../test_data/out_g4.txt",
        "../../test_data/out_g1g4.txt"
    )
    
    with open(files[0] + ".pkl", 'rb') as p:
        shape, edges, centers = pickle.load(p)
    
    
    data_g1 = np.memmap(files[0] + ".mm", mode='r', shape=shape, dtype=np.float64)
    data_g4 = np.memmap(files[1] + ".mm", mode='r', shape=shape, dtype=np.float64)
    data_g1g4 = np.memmap(files[2] + ".mm", mode='r', shape=shape, dtype=np.float64)
    
    # print(data_g1, data_g4, data_g1g4, sep='\n')
    
    dim_bins = bm.Grim_Brunelle_nonlocal(edges, data_g1, data_g1g4 - data_g1 - data_g4, stats_check=False, SM_version=True)
    nonlocal_counts = dim_bins.run()
    dim_bins.dump_edges("Analytic_pairwise_INT")
    # dim_bins.visualize_changes(10, fname="Analytic_pairwise_INT")
    
    dim_bins = bm.Grim_Brunelle_nonlocal(edges, data_g1, data_g4, data_g1g4 - data_g1 - data_g4, stats_check=False, SM_version=True)
    nonlocal_counts = dim_bins.run()
    dim_bins.dump_edges("Analytic")
    # dim_bins.visualize_changes(10, fname="Analytic")
    
    os.system('mv *.png ../../plots/')
