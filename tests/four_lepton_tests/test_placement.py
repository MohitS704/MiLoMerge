import helphi as phi

import numpy as np
import pickle as pkl
import uproot
import matplotlib.pyplot as plt
import mplhep as hep
hep.style.use(hep.style.CMS)
import sys
sys.path.append('../../brunelle_merger/')
import SUPER_ROC_Curves as ROC
import warnings
import tqdm
warnings.filterwarnings("ignore")

def test_phi():
    
    data = uproot.open('../../test_data/data.root')
    branches = ['Z1Mass', 'Z2Mass', 'helphi', 'helcosthetaZ1', 'helcosthetaZ2']
    sm_data = data['sm'].arrays(branches, library='np')
    ps_data = data['ps'].arrays(branches, library='np')
    
    N = 10
    
    g1_counts, bins = np.histogram(sm_data['helphi'], 100, range=[-np.pi, np.pi])
    g4_counts, _ = np.histogram(ps_data['helphi'], bins)
    
    merged_g1_counts = np.zeros(N, dtype=np.float64)
    merged_g4_counts = np.zeros(N, dtype=np.float64)
    
    for z1 in tqdm.tqdm(sm_data['helphi'], total=len(sm_data['helphi'])):
        unrolled_index, mapped_index = phi.place_entry(N, [z1])
        unrolled_index = tuple(unrolled_index)
        merged_g1_counts[mapped_index] += 1
    
    for z4 in tqdm.tqdm(ps_data['helphi'], total=len(ps_data['helphi'])):
        unrolled_index, mapped_index = phi.place_entry(N, [z4])
        unrolled_index = tuple(unrolled_index)
        merged_g4_counts[mapped_index] += 1
        
    print("SUM:", len(sm_data['helphi']), merged_g1_counts.sum())
    print("SUM:", len(ps_data['helphi']), merged_g4_counts.sum())
    
    g14 = ROC.length_scale_ROC(g1_counts, np.sum(g1_counts), g4_counts, np.sum(g4_counts))
    g14_m = ROC.length_scale_ROC(merged_g1_counts, np.sum(g1_counts), merged_g4_counts, np.sum(g4_counts))
    
    plt.plot(*g14[:2],label="0PM vs 0M: l={:.3f} A={:.3f}".format(*g14[-2:]))
    plt.plot(*g14_m[:2],label="0PM vs 0M Merged: l={:.3f} A={:.3f}".format(*g14_m[-2:]))
    plt.legend()
    plt.show()
    
    
if __name__ == '__main__':
    test_phi()
