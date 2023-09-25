import brunelle_merger.brunelle_merger as bm
import numpy as np
import matplotlib.pyplot as plt
# import scipy.stats as sp
import mplhep as hep
import pickle

# import multidimensionaldiscriminant.optimizeroc as optimizeroc #clone an instance of multidimensional roc in your area

# def earth_mover(counts_1, counts_2):
#     """A stupid wrapper for the wasserstein/earth mover's distance

#     Parameters
#     ----------
#     counts_1 : numpy.ndarray
#         counts for hypo 1
#     counts_2 : numpy.ndarray
#         counts for hypo 2

#     Returns
#     -------
#     float
#         a distance!
#     """
#     return sp.wasserstein_distance(counts_1, counts_2)

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
        
        
    TPR = PAC/(PAC + PBC) #vectorized calculation
    FPR = NAC/(NAC + NBC)
    
    TPR[~np.isfinite(TPR)] = 0
    FPR[~np.isfinite(FPR)] = 0
    
    return TPR, FPR, np.abs(np.trapz(FPR, TPR))



if __name__ == "__main__":
    data_1 = np.random.normal(3, 2, 10000)
    data_2 = np.random.normal(5, 2, 10000)

    counts, bins = np.histogram(data_1, 50, range=[0,10], density=True)
    counts /= counts.sum()

    counts_2, _ = np.histogram(data_2, bins, density=True)
    counts_2 /= counts_2.sum()

    # print(counts, counts_2, bins)

    merging = len(counts) - 1

    N_BINS_WANTED = 10
    clown = bm.Brunelle_merger(counts, counts_2, bins, clown_metric)
    grim_terms = clown.greedy_grim_merge(N_BINS_WANTED)

    other_test = bm.Grim_Brunelle_merger(bins, counts, counts_2)
    new_grim_terms = other_test.run_local(N_BINS_WANTED)
    
    other_other_test = bm.Grim_Brunelle_merger(bins, counts, counts_2)
    
    
    x = other_other_test.closest_pair_1d( list(range(len(other_other_test.merged_counts))), brute_force=False)
    xx = other_other_test.closest_pair_1d(list(range(len(other_other_test.merged_counts))), brute_force=True)
    
    print(x)
    print(xx)
    
    new_new_grim_terms = other_other_test.run_local(N_BINS_WANTED, True)
    
    # print(grim_terms[2])
    # 
    # print(new_grim_terms[1])
    # 
    # print(new_new_grim_terms[1])
    
    # print(bins)
    # print("{:<40}".format("OG Score:"), ROC_curve(data_1.copy(), data_2.copy(), bins.copy() )[-1] )
    
    # print(grim_terms[-1])
    # print("{:<40}".format("GRIM SCORE:"), ROC_curve(data_1, data_2, grim_terms[-1], 0, 10)[-1])
    
    # print(new_grim_terms[-1])
    # print("{:<40}".format("GRIM SCORE new object old style:"), ROC_curve(data_1, data_2, new_grim_terms[-1], 0, 10)[-1])
    
    # print(new_new_grim_terms[-1])
    # print("{:<40}".format("GRIM SCORE new object new style:"), ROC_curve(data_1, data_2, new_new_grim_terms[-1], 0, 10)[-1])
    
    
    # hists = {
    #     1 : counts,
    #     2 : counts_2
    # }

    # rocareacoeffdict = {(1,2):1}
    # optimizer = optimizeroc.OptimizeRoc(hists, rocareacoeffdict, mergemultiplebins=True, smallchangetolerance=1e-5)
    # # result = optimizer.run()
    # saved_result = optimizer.run(resultfilename="output.pkl", rawfilename="output_raw.pkl")
    # with open("output_raw.pkl", "rb") as f:
    #     saved_result = pickle.load(f)


    # print( "OG Score:", ROC_curve(data_1.copy(), data_2.copy(), bins.copy() )[-1] )

    # edm = bm.Brunelle_merger(counts, counts_2, bins, earth_mover)
    # edm_terms = edm.greedy_merge(N_BINS_WANTED)

    # print(*edm_terms)
    # print("EMD Score:", ROC_curve(data_1, data_2, edm_terms[-1], 0, 10)[-1] )


    # heshy = bm.Brunelle_merger(counts, counts_2, bins, heshy_metric)
    # heshy_terms = heshy.greedy_merge(N_BINS_WANTED)

    # print(*heshy_terms)
    # print("HESHY Score:", ROC_curve(data_1, data_2, heshy_terms[-1], 0, 10)[-1] )

    # frown = bm.Brunelle_merger(counts, counts_2, bins, frown_metric)
    # frown_terms = frown.greedy_merge(N_BINS_WANTED)

    # print(*frown_terms)
    # print("FROWN SCORE:", ROC_curve(data_1, data_2, frown_terms[-1], 0, 10)[-1])

    # clown = bm.Brunelle_merger(counts, counts_2, bins, clown_metric)
    # clown_terms = clown.greedy_merge(N_BINS_WANTED)

    # print(*clown_terms)
    # print("CLOWN SCORE:", ROC_curve(data_1, data_2, clown_terms[-1], 0, 10)[-1])

    # new_bins = []

    # bins_made = sorted(saved_result[-N_BINS_WANTED], key=min)
    # for bin in bins_made:
    # #         print(min(bin), max(bin))
    #     index = min(bin)[0]
    #     new_bins.append(bins[index])

    # index = max(bins_made[-1])[0]
    # new_bins.append(bins[index])

    # if new_bins[-1] != bins[-1]:
    #     new_bins[-1] = bins[-1]

    # print("ALGO SCORE:", ROC_curve(data_1, data_2, new_bins)[-1])

    # grim_terms = clown.greedy_grim_merge(N_BINS_WANTED)
    # print(*grim_terms)
    # print("GRIM SCORE:", ROC_curve(data_1, data_2, grim_terms[-1], 0, 10)[-1])


    # mosaic = [[1,2,3,6],
    #         [1,4,5,7]]

    # ref_dict = {
    #     1 : bins,
    #     2 : edm_terms[-1],
    #     3 : heshy_terms[-1],
    #     4 : frown_terms[-1],
    #     5 : clown_terms[-1],
    #     6 : grim_terms[-1],
    #     7 : new_bins
    # }

    # ref_names = {
    #     1 : "OG",
    #     2 : "EMD",
    #     3 : "RAW ROC",
    #     4 : "FROWN",
    #     5 : "CLOWN",
    #     6 : "GRIM",
    #     7 : "OG ALGO"
    # }


    # fig, ax = plt.subplot_mosaic(mosaic, figsize=(10,7))

    # fig2, ax2 = plt.subplots(1,1, figsize=(10,7))

    # for i in ref_dict.keys():
    #     counts_1, _ = np.histogram(data_1, ref_dict[i], density=True)
    #     counts_1 /= counts_1.sum()
    #     counts_2, _ = np.histogram(data_2, ref_dict[i], density=True)
    #     counts_2 /= counts_2.sum()
        
    #     print("BIN_LENGTH FOR:", ref_names[i], ":", len(ref_dict[i]))
        
    #     hep.histplot(counts_1, ref_dict[i], label="1", lw=3, ax=ax[i])
    #     hep.histplot(counts_2, ref_dict[i], label="2", lw=3, ax=ax[i])
        
    #     TPR, FPR, score = ROC_curve(data_1.copy(), data_2.copy(), ref_dict[i].copy() )
        
    #     ax[i].set_title(ref_names[i] + " SCORE={:.2f}".format(score))
    #     ax[i].legend()
        
    #     ax2.plot(TPR, FPR, label=ref_names[i] + " SCORE={:.2f}".format(score))


    # fig.suptitle("Reducing LHS to {:.0f} bins".format(N_BINS_WANTED))
    # fig.tight_layout()
    # fig.savefig('testing_hist.png')

    # ax2.legend()
    # fig2.suptitle("Reducing OG to 5 bins")
    # fig2.tight_layout()
    # fig2.savefig('testing_hist_ROC.png')