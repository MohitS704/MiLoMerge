import brunelle_merger.brunelle_merger as bm
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sp
import mplhep as hep

data_1 = np.random.normal(3, 2, 1000)
data_2 = np.random.normal(5, 5, 1000)

counts, bins = np.histogram(data_1, 50, range=[0,10], density=True)
counts /= counts.sum()

counts_2, _ = np.histogram(data_2, bins, density=True)
counts_2 /= counts_2.sum()

# print(counts, counts_2, bins)

# merging = 1

# print("Merging bin {:.0f} with bin {:.0f}".format(merging, merging+1))
# print(bm.merge_ahead(counts, bins, merging))

def earth_mover(counts_1, counts_2):
    """A stupid wrapper for the wasserstein/earth mover's distance

    Parameters
    ----------
    counts_1 : numpy.ndarray
        counts for hypo 1
    counts_2 : numpy.ndarray
        counts for hypo 2

    Returns
    -------
    float
        a distance!
    """
    return sp.wasserstein_distance(counts_1, counts_2)

def heshy_metric(counts_1, counts_2):
    """A stupid wrapper for heshy's distance (calculates a ROC curve)

    Parameters
    ----------
    counts_1 : numpy.ndarray
        counts for hypo 1
    counts_2 : numpy.ndarray
        counts for hypo 2

    Returns
    -------
    float
        a distance!
    """
    sorted_ratios = sorted(
        list(enumerate(counts_1/counts_2)), key=lambda x: x[1], reverse=True
    )
    
    
    sorted_ratios = np.array(sorted_ratios)[:,0].astype(int) #gets the bin indices only for the ordered ratio pairs
    sorted_ratios[~np.isfinite(sorted_ratios)] = 0
    
    length = len(sorted_ratios) + 1
    
    PAC = np.zeros(length) #"positive" above cutoff
    PBC = np.zeros(length) #"positive" below cutoff
    NAC = np.zeros(length) #"negative" above cutoff
    NBC = np.zeros(length) #"negative" below cutoff
    
    
    for n in range(length):
        above_cutoff = sorted_ratios[n:]
        below_cutoff = sorted_ratios[:n]
        
        PAC[n] = counts_1[above_cutoff].sum() #gets the indices listed
        PBC[n] = counts_1[below_cutoff].sum()
        
        NAC[n] = counts_2[above_cutoff].sum()
        NBC[n] = counts_2[below_cutoff].sum()
    
    TPR = 1 - PAC/(PAC + PBC) #vectorized calculation
    FPR = 1 - NAC/(NAC + NBC)
    
    TPR[~np.isfinite(TPR)] = 0
    FPR[~np.isfinite(FPR)] = 0
    
    return np.trapz(TPR, FPR)
    
def frown_metric(counts_1, counts_2):
    """A stupid wrapper for the ratio/frowning metric

    Parameters
    ----------
    counts_1 : numpy.ndarray
        counts for hypo 1
    counts_2 : numpy.ndarray
        counts for hypo 2

    Returns
    -------
    float
        a distance!
    """
    return np.sum(counts_1/counts_2)

def clown_metric(counts_1, counts_2):
    """A stupid wrapper for the enhanced ratio/clown metric

    Parameters
    ----------
    counts_1 : numpy.ndarray
        counts for hypo 1
    counts_2 : numpy.ndarray
        counts for hypo 2

    Returns
    -------
    float
        a distance!
    """
    return np.sum( np.abs(counts_1 - counts_1)/(counts_1 + counts_2) )

def ROC_curve(sample1, sample2, bins=100, lower=0, upper=1):
    """This function produces a ROC curve from an attribute like phi, cos(theta1), D_{0-}, etc.

    Parameters
    ----------
    sample1 : numpy.ndarray
        The first data sample for your attribute. This is your "True" data
    sample2 : numpy.ndarray
        The second data sample for your attribute. This if your "False" data
    bins : int or numpy.ndarray, optional
        The number of bins for the ROC calculation. Can also be given a list of bins., by default 100
    lower : float
        The lower end of your sample range
    upper : float
        The upper end of your sample range
    

    Returns
    -------
    tuple(numpy.ndarray, numpy.ndarray, float)
        returns the true rate, the false rate, and the area under the curve (assuming true rate is the x value)
    """
    
    sample1 = np.array(sample1)
    sample2 = np.array(sample2)
    
    if isinstance(bins, int):
        _, bins = np.histogram([], bins=bins, range=[lower, upper])
    
    hypo2_counts, bins = np.histogram(sample2, bins=bins, density=True)
    hypo2_counts /= hypo2_counts.sum()
    
    hypo1_counts, _ = np.histogram(sample1, bins=bins, density=True)
    hypo1_counts /= hypo1_counts.sum()
    
    # print(list(g1_phi_counts))
    # print()
    # print(list(g4_phi_counts))
    
    ratios = sorted(
        list(enumerate(hypo1_counts/hypo2_counts)), key=lambda x: x[1], reverse=True
    )
    # print(ratios)
    
    ratios = np.array(ratios)[:,0].astype(int) #gets the bin indices only for the ordered ratio pairs
    ratios[~np.isfinite(ratios)] = 0
    # print(ratios)
    # print()
    length = len(ratios) + 1
    
    PAC = np.zeros(length) #"positive" above cutoff
    PBC = np.zeros(length) #"positive" below cutoff
    NAC = np.zeros(length) #"negative" above cutoff
    NBC = np.zeros(length) #"negative" below cutoff
    
    
    for n in range(length):
        above_cutoff = ratios[n:]
        below_cutoff = ratios[:n]
        
        PAC[n] = hypo1_counts[above_cutoff].sum() #gets the indices listed
        PBC[n] = hypo1_counts[below_cutoff].sum()
        
        NAC[n] = hypo2_counts[above_cutoff].sum()
        NBC[n] = hypo2_counts[below_cutoff].sum()
        
        # for bin_index in above_cutoff: #The above lines are the same as this commented code but vectorized
        #     PAC += g1_phi_counts[bin_index]
        #     NAC += g4_phi_counts[bin_index]
        
        # for bin_index in below_cutoff:
        #     PBC += g1_phi_counts[bin_index]
        #     NBC += g4_phi_counts[bin_index]
        # TPR.append(1 - PAC/(PAC + PBC))
        # FPR.append(1 - NAC/(NAC + NBC))
        
        
    TPR = 1 - PAC/(PAC + PBC) #vectorized calculation
    FPR = 1 - NAC/(NAC + NBC)
    
    TPR[~np.isfinite(TPR)] = 0
    FPR[~np.isfinite(FPR)] = 0
    
    return TPR, FPR, np.trapz(TPR, FPR)


print( "OG Score:", ROC_curve(data_1.copy(), data_2.copy(), bins.copy() )[-1] )

edm = bm.Brunelle_merger(counts, counts_2, bins, earth_mover)
edm_terms = edm.greedy_merge(5)

print(*edm_terms)
print("EMD Score:", ROC_curve(data_1, data_2, edm_terms[-1], 0, 10)[-1] )


heshy = bm.Brunelle_merger(counts, counts_2, bins, heshy_metric)
heshy_terms = heshy.greedy_merge(5)

print(*heshy_terms)
print("HESHY Score:", ROC_curve(data_1, data_2, heshy_terms[-1], 0, 10)[-1] )

frown = bm.Brunelle_merger(counts, counts_2, bins, frown_metric)
frown_terms = frown.greedy_merge(5)

print(*frown_terms)
print("FROWN SCORE:", ROC_curve(data_1, data_2, frown_terms[-1], 0, 10)[-1])

clown = bm.Brunelle_merger(counts, counts_2, bins, clown_metric)
clown_terms = clown.greedy_merge(5)

print(*clown_terms)
print("CLOWN SCORE:", ROC_curve(data_1, data_2, clown_terms[-1], 0, 10)[-1])



mosaic = [[1,2,3],
          [1,4,5]]

ref_dict = {
    1 : bins,
    2 : edm_terms[-1],
    3 : heshy_terms[-1],
    4 : frown_terms[-1],
    5 : clown_terms[-1]
}

ref_names = {
    1 : "OG",
    2 : "EMD",
    3 : "HESHY",
    4 : "FROWN",
    5 : "CLOWN"
}


fig, ax = plt.subplot_mosaic(mosaic, figsize=(10,7))

for i in ref_dict.keys():
    counts_1, _ = np.histogram(data_1, ref_dict[i], density=True)
    counts_1 /= counts_1.sum()
    counts_2, _ = np.histogram(data_2, ref_dict[i], density=True)
    counts_2 /= counts_2.sum()
    
    hep.histplot(counts_1, ref_dict[i], label="1", lw=3, ax=ax[i])
    hep.histplot(counts_2, ref_dict[i], label="2", lw=3, ax=ax[i])
    
    ax[i].set_title(ref_names[i] + " SCORE={:.2f}".format(ROC_curve(data_1.copy(), data_2.copy(), ref_dict[i].copy() )[-1]))
    ax[i].legend()


fig.suptitle("Reducing LHS to 5 bins")
fig.tight_layout()
fig.savefig('testing_hist.png')