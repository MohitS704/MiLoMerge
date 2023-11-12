import numpy as np
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib.patches
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
import SUPER_ROC_Curves as ROC
import histogram_helpers as h

def run_test(stats_check, bins_wanted, subtraction_metric):
    OG_edges = edges.copy()
    OG_counts = [None]*5
    
    grim_bins = [None]*5
    grim_counts = [None]*5
    
    grim_bins_fast = [None]*5
    grim_counts_fast = [None]*5
    
    
    nonlocal_bins = [None]*5
    nonlocal_counts = [None]*5
    
    heshy_bins = [None]*5
    heshy_counts = [None]*5
    
    post_merge_bins = [None]*5
    post_merge_counts = [None]*5

    for i in range(5):
        # indiv_axis = tuple([j for j in range(5) if j != i])
        # x = np.sum(counts_sm, axis=indiv_axis)
        # xp = np.sum(counts_ps, axis=indiv_axis)
        x = counts_sm[i]
        xp = counts_ps[i]
        OG_counts[i] = [x.copy(), xp.copy()]
        
        dim_bins = bm.Grim_Brunelle_merger(edges[i], x.copy(), xp.copy(), stats_check=stats_check, subtraction_metric=subtraction_metric)
        start1 = time.time()
        temp_counts, temp_bins = dim_bins.run(bins_wanted)
        end1 = time.time()
        grim_bins[i] = temp_bins.copy()
        grim_counts[i] = temp_counts.copy()
        print("Recalculating everything:", end1 - start1)
        
        # dim_bins.reset()
        # start2 = time.time()
        # temp_counts, temp_bins = dim_bins.run_local_faster(bins_wanted)
        # end2 = time.time()
        # grim_bins_fast[i] = temp_bins.copy()
        # grim_counts_fast[i] = temp_counts.copy()
        # print("Sets:", end2 - start2)
        
        
        dim_bins = bm.Grim_Brunelle_nonlocal(edges[i], x.copy(), xp.copy(), stats_check=stats_check, subtraction_metric=subtraction_metric)
        start3 = time.time()
        nonlocal_counts[i], temp_bins = dim_bins.run(bins_wanted, track=False)
        tracked_points = dim_bins.tracker
        end3 = time.time()
        nonlocal_bins[i] = temp_bins.copy()
        print("Nonlocal:", end3 - start3)
        post_merge_bins[i] = dim_bins.post_stats_merge_bins.copy()
        post_merge_counts[i] = dim_bins.merged_counts.copy()
        
        hists = {
        "0+": dim_bins.merged_counts.copy()[0],
        "0-": dim_bins.merged_counts.copy()[1]
        }

        rocareacoeffdict = {("0+", "0-"): 1}
        optimizer = optimizeroc.OptimizeRoc(hists, rocareacoeffdict, mergemultiplebins=True, smallchangetolerance=1e-5)
        start4 = time.time()
        optimizer.run(resultfilename="output.pkl", rawfilename="output_raw.pkl")
        end4 = time.time()
        print("Heshy:", end4 - start4)
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
        temp_heshy = [[],[]]
        temp_heshy[0], _ = np.histogram(sm_data[branches[i]], new_bins)
        temp_heshy[1], _ = np.histogram(ps_data[branches[i]], new_bins)
        heshy_counts[i] = np.array(temp_heshy).copy()
        print()
        
    for i, key in enumerate(sm_data.keys()):
        mosaic = [[1,2,4,6,7,7],
                [1,3,5,6,7,7]]
        fig, ax = plt.subplot_mosaic(mosaic, figsize=(14,7), facecolor='white')
        # print("GRIM BINS:", grim_bins[i])
        # print("HESHY BINS:", heshy_bins[i])
        
        # print("grim counts:", grim_counts[i], grim_counts[i].shape)
        # print("grim bins:", grim_bins[i])
                
        TPR, FPR, score = ROC.ROC_curve(grim_counts[i][0], grim_counts[i][1])
        ax[7].plot(TPR, FPR, label="GRIM METRIC")
        ax[2].hist([sm_data[key], ps_data[key]], grim_bins[i], label=[r'$0^+$', r'$0^-$'], histtype='step', lw=3)
        ax[2].set_title("{:.3f}".format(score) + ": GRIM METRIC")
    #     plt.plot(TPR, FPR, label="{:.3f}".format(score) + ": GRIM METRIC", lw=3)
        
        TPR, FPR, score = ROC.ROC_curve(heshy_counts[i][0], heshy_counts[i][1])
        ax[7].plot(TPR, FPR, label="HESHY ALGO")
        hep.histplot([heshy_counts[i][0], heshy_counts[i][1]], heshy_bins[i], label=[r'$0^+$', r'$0^-$'], histtype='step', lw=3, ax=ax[3])
        # ax[3].hist([sm_data[key], ps_data[key]], heshy_bins[i], label=[r'$0^+$', r'$0^-$'], histtype='step', lw=3)
        ax[3].set_title("{:.3f}".format(score) + ": HESHY ALGO")
        
        TPR, FPR, score = ROC.ROC_curve(*nonlocal_counts[i])
        ax[7].plot(TPR, FPR, label="NONLOCAL ALGO")
        hep.histplot([nonlocal_counts[i][0], nonlocal_counts[i][1]], nonlocal_bins[i], label=[r'$0^+$', r'$0^-$'], histtype='step', lw=3, ax=ax[4])
        ax[4].set_title("{:.3f}".format(score) + ": NONLOCAL ALGO")
        
        # TPR, FPR, score = ROC_curve(grim_counts_fast[i][0], grim_counts_fast[i][1], grim_bins_fast[i], supplied_counts=True)
        # ax[7].plot(TPR, FPR, label="FAST GRIM")
        # ax[5].hist([sm_data[key], ps_data[key]], grim_bins_fast[i], label=[r'$0^+$', r'$0^-$'], histtype='step', lw=3)
        # ax[5].set_title("{:.3f}".format(score) + ": FAST GRIM METRIC")
        
        TPR, FPR, score = ROC.ROC_curve(OG_counts[i][0], OG_counts[i][1])
        ax[7].plot(TPR, FPR, label="UNMERGED")
        hep.histplot([OG_counts[i][0], OG_counts[i][1]], OG_edges[i], label=[r'$0^+$', r'$0^-$'], histtype='step', lw=3, ax=ax[1])
        # ax[1].hist([sm_data[key], ps_data[key]], OG_edges[i], label=[r'$0^+$', r'$0^-$'], histtype='step', lw=3)
        ax[1].set_title("{:.3f}".format(score) + ": UNMERGED")
        
        TPR, FPR, score = ROC.ROC_curve(post_merge_counts[i][0], post_merge_counts[i][1])
        ax[7].plot(TPR, FPR, label="STATS MERGE")
        hep.histplot([post_merge_counts[i][0], post_merge_counts[i][1]], post_merge_bins[i], label=[r'$0^+$', r'$0^-$'], histtype='step', lw=3, ax=ax[6])
        # ax[6].hist([sm_data[key], ps_data[key]], post_merge_bins[i], label=[r'$0^+$', r'$0^-$'], histtype='step', lw=3)
        ax[6].set_title("{:.3f}".format(score) + ": STATS MERGE ONLY")
        
        ax[1].legend(frameon=False, loc='best')
        ax[7].legend(frameon=False, loc='lower right')
        ax[7].set_xlim(0,1)
        ax[7].set_ylim(0,1)
        ax[7].plot([0, 1], [0, 1], transform=ax[5].transAxes, linestyle='dashed', color='black')
        fig.suptitle(key + " with {:.0f} histogram entries".format(bins_wanted))
        fig.tight_layout()
        
        fname = key + '_' + str(bins_wanted)
        if stats_check:
            fname += '_stats'
        else:
            fname += '_nostats'
        
        if subtraction_metric:
            fname += '_sub'
        else:
            fname += '_div'
        
        if stats_check:
            fig.savefig(fname + '.png')
        else:
            fig.savefig(fname+'.png')
        plt.close(fig)
        

if __name__ == "__main__": #TEST DATA!!
    data = uproot.open('../../test_data/data.root')
    
    branches = ['Z1Mass', 'Z2Mass', 'helphi', 'helcosthetaZ1', 'helcosthetaZ2']
    sm_data = data['sm'].arrays(branches, library='np')
    ps_data = data['ps'].arrays(branches, library='np')
    
    counts_sm = [None]*5
    counts_ps = [None]*5
    edges = [None]*5    
    
    ranges = [
        None,
        None,
        [-np.pi, np.pi],
        [-1,1],
        [-1,1]
    ]
    
    mosaic = [
        [0,0,'.','.','.'],
        [1,1,3,3,2],
        [1,1,3,3,2]
    ]
    
    for n, i in enumerate(branches):
        counts_sm[n], edges[n] = np.histogram(sm_data[i], 100, range=ranges[n], density=True)
        counts_ps[n], _ = np.histogram(ps_data[i], edges[n], density=True)
    
    ## 2D combinations
    
    # for comb in itertools.combinations(branches, 2):
    #     q1, q2 = comb
    #     Z1Z2_counts_sm, *Z1Z2_edges = np.histogram2d(sm_data[q1], sm_data[q2], 20)
    #     Z1Z2_counts_ps, *_ = np.histogram2d(ps_data[q1], ps_data[q2], Z1Z2_edges)
    #     sm_unroll = h.ND_histogram(Z1Z2_counts_sm.copy(), Z1Z2_edges.copy())
    #     sm_unroll.unroll()
        
    #     ps_unroll = h.ND_histogram(Z1Z2_counts_ps.copy(), Z1Z2_edges.copy())
    #     ps_unroll.unroll()
    #     dim_bins = bm.Grim_Brunelle_nonlocal(sm_unroll.unrolled_bins, sm_unroll.unrolled_counts, ps_unroll.unrolled_counts, stats_check=False, subtraction_metric=True)
    #     dim_bins.run(10, track=True)
    #     mapping = dim_bins.visualize_changes()
    #     # print(mapping)
    #     # print(sm_unroll.unrolled_to_rolled_converter)
        
    #     fig, ax = plt.subplot_mosaic(mosaic, constrained_layout=True, gridspec_kw={'wspace':0, 'hspace':0})
    #     hep.hist2dplot(Z1Z2_counts_sm, Z1Z2_edges[0], Z1Z2_edges[1], lw=2, ax=ax[1], cbar=False, cmap='Greys', norm=colors.SymLogNorm(linthresh=0.01, linscale=0.01, vmin=Z1Z2_counts_sm.min(), vmax=Z1Z2_counts_sm.max()))
    #     hep.hist2dplot(Z1Z2_counts_ps, Z1Z2_edges[0], Z1Z2_edges[1], lw=2, ax=ax[3], cbar=False, cmap='Greys', norm=colors.SymLogNorm(linthresh=0.01, linscale=0.01, vmin=Z1Z2_counts_ps.min(), vmax=Z1Z2_counts_ps.max()))
    #     hep.histplot([np.sum(Z1Z2_counts_sm, axis=1), np.sum(Z1Z2_counts_ps, axis=1)], Z1Z2_edges[0], lw=2, ax=ax[0])
    #     hep.histplot([np.sum(Z1Z2_counts_sm, axis=0), np.sum(Z1Z2_counts_ps, axis=0)], Z1Z2_edges[1], lw=2, ax=ax[2], orientation='horizontal')
    #     ax[1].set_xlabel(r'$M_{Z_1}$', fontsize=40)
    #     ax[1].set_ylabel(r'$M_{Z_2}$', fontsize=40)
    #     ax[0].set_xticks([])
    #     ax[2].set_yticks([])
    #     ax[3].set_yticks([])
    #     fig.tight_layout()
    #     fig.savefig(q1+"_"+q2+"_distr")
        
    #     plt.close('all')
    #     fig, ax = plt.subplot_mosaic(mosaic, constrained_layout=True, gridspec_kw={'wspace':0, 'hspace':0})
        
    #     Z1_centers, Z2_centers = sm_unroll.centers
    #     color_wheel = iter(plt.cm.gist_ncar(np.linspace(0, 1, 10)))
    #     for indices in mapping:
    #         c = next(color_wheel)
    #         for index in mapping[indices]:
    #             x, y = sm_unroll.unrolled_to_rolled_converter[index]
    #             # print(*(Z1_centers[x] - sm_unroll.diff[0], Z2_centers[y] - sm_unroll.diff[1]))
                
    #             rect = matplotlib.patches.Rectangle(
    #                 (Z1_centers[x] - sm_unroll.diff[0], Z2_centers[y] - sm_unroll.diff[1]),
    #                 sm_unroll.diff[0], sm_unroll.diff[1], color=c, zorder=np.inf)
                
    #             ax[0].scatter(Z1_centers[x], dim_bins.merged_counts[0][index], color=c, marker='o', s=50)
    #             ax[0].scatter(Z1_centers[x], dim_bins.merged_counts[1][index], color=c, marker='X', s=50)
                
    #             ax[2].scatter(dim_bins.merged_counts[0][index], Z2_centers[y], color=c, marker='o', s=50)
    #             ax[2].scatter(dim_bins.merged_counts[1][index], Z2_centers[y], color=c, marker='X', s=50)
    #             ax[2].invert_yaxis()
                
    #             ax[1].add_patch(rect)
    #             # plt.xlim()
                
    #             ax[1].scatter(Z1_centers[x], Z2_centers[y], color=c, marker='o', s=0)
    #             ax[1].scatter(Z1_centers[x], Z2_centers[y], color=c, marker='X', s=0)
        
    #     ax[1].set_xlabel(r'$M_{Z_1}$', fontsize=40)
    #     ax[1].set_ylabel(r'$M_{Z_2}$', fontsize=40)
    #     ax[0].set_xticks([])
    #     ax[2].set_yticks([])
    #     fig.tight_layout()
    #     fig.savefig(q1+"_"+q2+"_distr_merging")
        
    #     plt.close('all')
    
    # # A bunch of 1d combinations
    
    # for stat_check in [True, False]:
    #     for subtraction_metric in [True]: #division metric WILL nan out
    #         for n_bins in [5,7,10]:
    #             run_test(stat_check, n_bins, subtraction_metric)
    #             print()
    
    nonlocal_bins = [None]*5
    nonlocal_counts = [None]*5
    bins_wanted=5
    
    for i in range(len(branches)):
        # if i == 1: break
        x = counts_sm[i]
        xp = counts_ps[i]
        
        dim_bins = bm.Grim_Brunelle_nonlocal(edges[i], x.copy(), xp.copy(), stats_check=False, subtraction_metric=True, SM_version=True)
        start3 = time.time()
        nonlocal_counts[i], temp_bins = dim_bins.run()
        end3 = time.time()
        # tracked_points = dim_bins.tracker
        nonlocal_bins[i] = temp_bins.copy()
        print("Nonlocal:", end3 - start3)
        
        dim_bins.visualize_changes(2, xlabel=branches[i], fname="clustering_new_"+branches[i])
        dim_bins.dump_edges(branches[i])
        
    
    os.system('mv *.png ../../plots')
