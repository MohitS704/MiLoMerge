import brunelle_merger.brunelle_merger as bm
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sp
import mplhep as hep
import pickle
import uproot
import time

import multidimensionaldiscriminant.optimizeroc as optimizeroc #clone an instance of multidimensional roc in your area

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
        
        
    TPR = PAC/(PAC + PBC) #vectorized calculation
    FPR = NAC/(NAC + NBC)
    
    TPR[~np.isfinite(TPR)] = 0
    FPR[~np.isfinite(FPR)] = 0
    
    return TPR, FPR, np.abs(np.trapz(FPR, TPR))



if __name__ == "__main__":
    
    data = uproot.open('test_data/data.root')
    
    sm_data = data['sm'].arrays(['Z1Mass', 'Z2Mass', 'helphi', 'helcosthetaZ1', 'helcosthetaZ2'], library='np')
    ps_data = data['ps'].arrays(['Z1Mass', 'Z2Mass', 'helphi', 'helcosthetaZ1', 'helcosthetaZ2'], library='np')


    N_BINS_WANTED = 10
    counts_sm, edges = np.histogramdd([sm_data[i] for i in ['Z1Mass', 'Z2Mass', 'helphi', 'helcosthetaZ1', 'helcosthetaZ2']], 20,
                                range=[ None, None, [-3.14, 3.14], [-1, 1], [-1, 1] ])
    counts_sm /= counts_sm.sum()
    counts_ps, _ = np.histogramdd([ps_data[i] for i in ['Z1Mass', 'Z2Mass', 'helphi', 'helcosthetaZ1', 'helcosthetaZ2']], edges,
                                range=[ None, None, [-3.14, 3.14], [-1, 1], [-1, 1] ])
    counts_ps /= counts_ps.sum()

    OG_edges = edges.copy()
    
    grim_bins = [None]*5
    grim_bins_fast = [None]*5
    EMD_bins = [None]*5
    heshy_bins = [None]*5
    post_merge_bins = [None]*5

    bins_wanted = 5
    for i in range(5):
        indiv_axis = tuple([j for j in range(5) if j != i])
        x = np.sum(counts_sm, axis=indiv_axis)
        xp = np.sum(counts_ps, axis=indiv_axis)
        
        dim_bins = bm.Grim_Brunelle_merger(edges[i], x.copy(), xp.copy())
        start1 = time.time()
        *_, temp_bins = dim_bins.run_local(bins_wanted)
        end1 = time.time()
        grim_bins[i] = temp_bins.copy()
        print("Recalculating everything:", end1 - start1)
        
        dim_bins.reset()
        start2 = time.time()
        *_, temp_bins = dim_bins.run_local_faster(bins_wanted)
        end2 = time.time()
        grim_bins_fast[i] = temp_bins.copy()
        print("Sets:", end2 - start2)
        
        dim_bins = bm.Brunelle_merger(x.copy(), xp.copy(), edges[i], earth_mover)
        *_, temp_bins = dim_bins.greedy_merge(bins_wanted)
        EMD_bins[i] = temp_bins.copy()
        
        post_merge_bins[i] = dim_bins.post_stats_merge
        hists = {
        "0+": x,
        "0-": xp
        }

        rocareacoeffdict = {("0+", "0-"): 1}
        result = []
        # optimizer = optimizeroc.OptimizeRoc(hists, rocareacoeffdict, mergemultiplebins=True, smallchangetolerance=1e-5)
        # optimizer.run(resultfilename="output.pkl", rawfilename="output_raw.pkl")
        
        with open("output_raw.pkl", "rb") as f:
            saved_result = pickle.load(f)
        
        new_bins = []

        bins_made = sorted(saved_result[-bins_wanted], key=min)
        for bin in bins_made:
        #         print(min(bin), max(bin))
            index = min(bin)[0]
            new_bins.append(edges[i][index])

        index = max(bins_made[-1])[0]
        
        new_bins.append(edges[i][index])

        if new_bins[-1] != edges[i][-1]:
            new_bins[-1] = edges[i][-1]
        
        heshy_bins[i] = new_bins
        
    for i, key in enumerate(sm_data.keys()):
        mosaic = [[1,2,4,6],
                [1,3,5,6]]
        fig, ax = plt.subplot_mosaic(mosaic, figsize=(10,7), facecolor='white')
        print("GRIM BINS:", grim_bins[i])
        print("HESHY BINS:", heshy_bins[i])
            
        *_, score = ROC_curve(sm_data[key], ps_data[key], grim_bins[i])
        ax[2].hist([sm_data[key], ps_data[key]], grim_bins[i], label=[r'$0^+$', r'$0^-$'], histtype='step', lw=3)
        ax[2].set_title("{:.3f}".format(score) + ": GRIM METRIC")
    #     plt.plot(TPR, FPR, label="{:.3f}".format(score) + ": GRIM METRIC", lw=3)
        
        *_, score = ROC_curve(sm_data[key], ps_data[key], heshy_bins[i])
        ax[3].hist([sm_data[key], ps_data[key]], heshy_bins[i], label=[r'$0^+$', r'$0^-$'], histtype='step', lw=3)
        ax[3].set_title("{:.3f}".format(score) + ": HESHY ALGO")
        
        *_, score = ROC_curve(sm_data[key], ps_data[key], EMD_bins[i])
        ax[4].hist([sm_data[key], ps_data[key]], EMD_bins[i], label=[r'$0^+$', r'$0^-$'], histtype='step', lw=3)
        ax[4].set_title("{:.3f}".format(score) + ": EMD METRIC")
        
        *_, score = ROC_curve(sm_data[key], ps_data[key], grim_bins_fast[i])
        ax[5].hist([sm_data[key], ps_data[key]], grim_bins_fast[i], label=[r'$0^+$', r'$0^-$'], histtype='step', lw=3)
        ax[5].set_title("{:.3f}".format(score) + ": FAST GRIM METRIC")
        
        *_, score = ROC_curve(sm_data[key], ps_data[key], OG_edges[i])
        ax[1].hist([sm_data[key], ps_data[key]], OG_edges[i], label=[r'$0^+$', r'$0^-$'], histtype='step', lw=3)
        ax[1].set_title("{:.3f}".format(score) + ": UNMERGED")
        
        *_, score = ROC_curve(sm_data[key], ps_data[key], post_merge_bins[i])
        ax[6].hist([sm_data[key], ps_data[key]], post_merge_bins[i], label=[r'$0^+$', r'$0^-$'], histtype='step', lw=3)
        ax[6].set_title("{:.3f}".format(score) + ": STATS MERGE ONLY")
        
        ax[1].legend(frameon=False, loc='best')
        fig.suptitle(key + " with {:.0f} histogram entries".format(bins_wanted))
        fig.tight_layout()
        fig.savefig(key + '.png')
        plt.close(fig)