import Analytic as a
import helphi as phi
import Analytic_pairwise_INT as aT

import numpy as np
import pickle as pkl
import uproot
import matplotlib.pyplot as plt
import mplhep as hep
hep.style.use(hep.style.CMS)
import brunelle_merger.SUPER_ROC_Curves as ROC
import warnings
import tqdm
warnings.filterwarnings("ignore")


def test_analytic():
    files = (
            "test_data/out_g1.txt",
            "test_data/out_g4.txt",
            "test_data/out_g1g4.txt"
        )
    with open(files[0] + ".pkl", 'rb') as p:
            shape, edges, centers = pkl.load(p)

    
    N = 10
    
    
    data_g1 = np.memmap(files[0] + ".mm", mode='r', shape=shape, dtype=np.float64)
    data_g4 = np.memmap(files[1] + ".mm", mode='r', shape=shape, dtype=np.float64)
    data_g1g4 = np.memmap(files[2] + ".mm", mode='r', shape=shape, dtype=np.float64)
    
    data_int = (data_g1g4 - data_g1 - data_g4).copy()
    
    print("INTERFERENCE", data_int.sum(), data_int.sum()/np.sqrt(data_g1.sum() * data_g4.sum()), data_g1.sum(), data_g4.sum())
    
    g1_counts =  data_g1.copy().ravel()
    g4_counts =  data_g4.copy().ravel()
    int_counts = data_int.copy().ravel()
    
    g14 = ROC.length_scale_ROC(g1_counts, g4_counts)
    
    g14_I = ROC.length_scale_ROC(g1_counts, int_counts)
    
    merged_g1_counts = np.zeros(N, dtype=np.float64)
    merged_g4_counts = np.zeros(N, dtype=np.float64)
    merged_int_counts = np.zeros(N, dtype=np.float64)
    
    
    indices_already_done = set()
    for z1 in centers[0]:
        for z2 in centers[1]:
            for t1 in centers[2]:
                for t2 in centers[3]:
                    for phi in centers[4]:
                        unrolled_index, mapped_index = a.place_entry(N, [z1, z2, t1, t2, phi])
                        # print([z1, z2, t1, t2, phi], ":", unrolled_index, "->", mapped_index)
                        unrolled_index = tuple(unrolled_index)
                        if unrolled_index in indices_already_done:
                            print("REPEATED INDEX:", unrolled_index)
                            # exit()
                        indices_already_done.add(unrolled_index)
                        merged_g1_counts[mapped_index] += data_g1[unrolled_index]
                        merged_g4_counts[mapped_index] += data_g4[unrolled_index]
                        merged_int_counts[mapped_index] += data_int[unrolled_index]
    
    print("SUM INT", data_int.sum(), merged_int_counts.sum(), int_counts.sum())
    print("SUM POS", data_g1.sum(), merged_g1_counts.sum(), g1_counts.sum())
    
    
    g14_m = ROC.length_scale_ROC(merged_g1_counts, merged_g4_counts)
    g14_I_m = ROC.length_scale_ROC(merged_g1_counts, merged_int_counts)
    
    plt.plot(*g14[:2],label="0PM vs 0M: {:.3f}".format(g14[-1]))
    g14_test = ROC.ROC_curve(g1_counts, g4_counts)
    # plt.plot(*g14_test[:2],label="0PM vs 0M test: {:.3f}".format(g14_test[-1]))
    plt.plot(*g14_I[:2], label="0PM vs INT: {:.3f}".format(g14_I[-1]))
    
    plt.plot(*g14_m[:2],label="0PM vs 0M Merged: {:.3f}".format(g14_m[-1]))
    plt.plot(*g14_I_m[:2], label="0PM vs INT Merged: {:.3f}".format(g14_I_m[-1]))
    plt.gca().axhline(0, color='k', ls='dashed', zorder=-1)
    plt.legend()
    plt.show()
    
    hep.histplot(merged_g1_counts ,label = "g1")
    hep.histplot(merged_g4_counts ,label = "g4")
    hep.histplot(merged_int_counts,label = "INT")
    plt.legend()
    plt.title('Final Bins')
    plt.show()
    
    # for i in range(5):
    #     plt.figure()
    #     indiv_axis = tuple([j for j in range(5) if j != i])
        
    #     g1_temp = np.sum(data_g1, axis=indiv_axis)
    #     g1_scalefac = np.abs(g1_temp).sum()
    #     g1_err = np.sqrt(g1_temp)/g1_scalefac
    #     g1_temp /= g1_scalefac
        
    #     g4_temp = np.sum(data_g4, axis=indiv_axis)
    #     g4_scalefac = np.abs(g4_temp).sum()
    #     g4_err = np.sqrt(g4_temp)/g4_scalefac
    #     g4_temp /= g4_scalefac
        
    #     int_temp = np.sum(data_int, axis=indiv_axis)
    #     int_scalefac = np.sqrt( g1_scalefac*g4_scalefac )
    #     int_err = np.sqrt(np.abs(int_temp))/int_scalefac
    #     int_temp /= int_scalefac
        
    #     # counts = np.zeros(N, dtype=np.float64)
        
    #     print(g1_temp)
    #     print(g4_temp)
    #     print(int_temp)
    #     print()
        
    #     hep.histplot(g1_temp, edges[i] , histtype='errorbar', yerr=g1_err , xerr=True)
    #     hep.histplot(g4_temp, edges[i] , histtype='errorbar', yerr=g4_err , xerr=True)
    #     hep.histplot(int_temp, edges[i], histtype='errorbar', yerr=int_err, xerr=True)
    #     plt.show()
    #     plt.close('all')


def test_phi():
    
    data = uproot.open('test_data/data.root')
    branches = ['Z1Mass', 'Z2Mass', 'helphi', 'helcosthetaZ1', 'helcosthetaZ2']
    sm_data = data['sm'].arrays(branches, library='np')
    ps_data = data['ps'].arrays(branches, library='np')
    
    N = 10
    
    g1_counts, bins = np.histogram(sm_data['helphi'], 100, range=[-np.pi, np.pi])
    g4_counts, _ = np.histogram(ps_data['helphi'], bins)
    
    merged_g1_counts = np.zeros(N, dtype=np.float64)
    merged_g4_counts = np.zeros(N, dtype=np.float64)
    
    for z1, z4 in tqdm.tqdm(zip(sm_data['helphi'], ps_data['helphi']), total=len(sm_data['helphi'])):
        unrolled_index, mapped_index = phi.place_entry(N, [z1])
        unrolled_index = tuple(unrolled_index)
        merged_g1_counts[mapped_index] += 1
        
        unrolled_index, mapped_index = phi.place_entry(N, [z4])
        unrolled_index = tuple(unrolled_index)
        merged_g4_counts[mapped_index] += 1
        
    print("SUM:", len(sm_data['helphi']), merged_g1_counts.sum())
    
    g14 = ROC.length_scale_ROC(g1_counts, g4_counts)
    g14_m = ROC.length_scale_ROC(merged_g1_counts, merged_g4_counts)
    
    plt.plot(*g14[:2],label="0PM vs 0M: {:.3f}".format(g14[-1]))
    plt.plot(*g14_m[:2],label="0PM vs 0M Merged: {:.3f}".format(g14_m[-1]))
    plt.legend()
    plt.show()
    

def test_analytic_INT_only():
    files = (
            "test_data/out_g1.txt",
            "test_data/out_g4.txt",
            "test_data/out_g1g4.txt"
        )
    with open(files[0] + ".pkl", 'rb') as p:
            shape, edges, centers = pkl.load(p)

    
    N = 10
    
    
    data_g1 = np.memmap(files[0] + ".mm", mode='r', shape=shape, dtype=np.float64)
    data_g4 = np.memmap(files[1] + ".mm", mode='r', shape=shape, dtype=np.float64)
    data_g1g4 = np.memmap(files[2] + ".mm", mode='r', shape=shape, dtype=np.float64)
    
    data_int = (data_g1g4 - data_g1 - data_g4).copy()
    
    print("INTERFERENCE", data_int.sum(), data_int.sum()/np.sqrt(data_g1.sum() * data_g4.sum()), data_g1.sum(), data_g4.sum())
    
    g1_counts =  data_g1.copy().ravel()
    int_counts = data_int.copy().ravel()
        
    g14_I = ROC.length_scale_ROC(g1_counts, int_counts)
    
    merged_g1_counts = np.zeros(N, dtype=np.float64)
    merged_int_counts = np.zeros(N, dtype=np.float64)
    
    
    indices_already_done = set()
    total_sum = 0
    for z1 in centers[0]:
        for z2 in centers[1]:
            for t1 in centers[2]:
                for t2 in centers[3]:
                    for phi in centers[4]:
                        unrolled_index, mapped_index = aT.place_entry(N, [z1, z2, t1, t2, phi])
                        # print([z1, z2, t1, t2, phi], ":", unrolled_index, "->", mapped_index)
                        unrolled_index = tuple(unrolled_index)
                        if unrolled_index in indices_already_done:
                            print("REPEATED INDEX:", unrolled_index)
                            exit()
                        indices_already_done.add(unrolled_index)
                        merged_g1_counts[mapped_index] += data_g1[unrolled_index]
                        merged_int_counts[mapped_index] += data_int[unrolled_index]
                        total_sum += data_int[unrolled_index]
    
    print("TOTAL SUM INT:", total_sum)
    print("SUM INT", data_int.sum(), merged_int_counts.sum(), int_counts.sum())
    print("SUM POS", data_g1.sum(), merged_g1_counts.sum(), g1_counts.sum())
    
    
    g14_I_m = ROC.length_scale_ROC(merged_g1_counts, merged_int_counts)
    
    plt.plot(*g14_I[:2], label="0PM vs INT: l={:.3f}, A={:.3f}".format(*g14_I[-2:]))
    
    plt.plot(*g14_I_m[:2], label="0PM vs INT Merged: l={:.3f} A={:.3f}".format(*g14_I_m[-2:]))
    plt.legend()
    plt.show()
    
    hep.histplot(merged_g1_counts ,label = "g1")
    hep.histplot(merged_int_counts,label = "INT")
    plt.gca().axhline(0, color='k', ls='dashed', zorder=-1)
    plt.legend()
    plt.title('Final Bins')
    plt.show()

if __name__ == '__main__':
    # test_phi()
    # test_analytic()
    test_analytic_INT_only()
    