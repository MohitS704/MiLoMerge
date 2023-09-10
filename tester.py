import brunelle_merger.brunelle_merger as bm
import numpy as np
import sys
import matplotlib.pyplot as plt
sys.path.append("/eos/home-m/msrivast/CMSSW_12_2_0_/src/HexUtils/")
import useful_helpers as help
import uproot

g1_discr = uproot.open("/eos/home-m/msrivast/CMSSW_12_2_0_/src/HexUtils/g1discrs.root")
g4_discr = uproot.open("/eos/home-m/msrivast/CMSSW_12_2_0_/src/HexUtils/g4discrs.root")


g1_data = {
    'd0-':[]
}

g4_data = {
    'd0-':[]
}

for key in g1_discr.keys():
    g1_data['d0-'] += list(g1_discr[key]['d0-'].array(library='np'))
    g4_data['d0-'] += list(g4_discr[key]['d0-'].array(library='np'))

initial_bins = np.linspace(0,1,200)

initial_data_1 = g1_data['d0-']

initial_counts_1, _ = np.histogram(initial_data_1, initial_bins, density=True)

initial_data_2 = g4_data['d0-']

initial_counts_2, _ = np.histogram(initial_data_1, initial_bins, density=True)

plt.hist(initial_data_1, initial_bins, histtype='step', density=True)
plt.hist(initial_data_2, initial_bins, histtype='step', density=True)
plt.savefig('testing_hist.png')


def earth_mover(counts_1, counts_2):
    return sp.wasserstein_distance(counts_1, counts_2)

def heshy_metric(counts_1, counts_2):
        
    sorted_ratios = sorted(
        list(enumerate(counts_1/counts_2)), key=lambda x: x[1], reverse=True
    )
    sorted_ratios = np.array(sorted_ratios)
    
    sorted_counts = sorted_ratios[:,1]
    
    cumsum = np.cumsum(sorted_counts)
    
    return np.trapz(cumsum, range(1, len(cumsum) + 1))
    
def new_metric(counts_1, counts_2):
    return counts_1/counts_2


_, _, area_orig = help.ROC_curve(initial_data_1, initial_data_2, initial_bins)

x_w = bm.brunelle_merge(initial_counts_1, initial_counts_2, initial_bins, earth_mover)

print("DONE!")

_, _, area = help.ROC_curve(initial_data_1, initial_data_2, x_w[-1])

x_h = bm.brunelle_merge(initial_counts_1, initial_counts_2, initial_bins, heshy_metric)

print("DONE!")

_, _, area_2 = help.ROC_curve(initial_data_1, initial_data_2, x_h[-1])

# print(x_w)
# print(x_h)

print(area_orig, area, area_2)