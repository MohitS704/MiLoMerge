import Analytic as a
import Analytic_pairwise_INT as aT

import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
import mplhep as hep
hep.style.use(hep.style.CMS)
import sys
sys.path.append("../../brunelle_merger/")
import SUPER_ROC_Curves as ROC
import warnings
warnings.filterwarnings("ignore")


def test_analytic():
    files = (
            "../../test_data/out_g1.txt",
            "../../test_data/out_g4.txt",
            "../../test_data/out_g1g4.txt"
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
    
    g14 = ROC.length_scale_ROC(g1_counts, np.abs(g1_counts).sum(), g4_counts, np.abs(g4_counts).sum())
    
    g14_I = ROC.length_scale_ROC(g1_counts, np.abs(g1_counts).sum(), int_counts, np.abs(int_counts).sum())
    
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
    
    print("SUM INT positive negative before", np.maximum(0, data_int).sum(), np.minimum(0, data_int).sum())
    print("SUM INT positive negative after", np.maximum(0, merged_int_counts).sum(), np.minimum(0, merged_int_counts).sum())
    
    print(merged_int_counts)
    print("SUM POS", data_g1.sum(), merged_g1_counts.sum(), g1_counts.sum())
    
    
    g14_m = ROC.length_scale_ROC(merged_g1_counts, np.abs(g1_counts).sum(), merged_g4_counts, np.abs(g4_counts).sum())
    g14_I_m = ROC.length_scale_ROC(merged_g1_counts, np.abs(g1_counts).sum(), merged_int_counts, np.abs(int_counts).sum())
    
    plt.plot(*g14[:2],label="0PM vs 0M: l={:.3f}, A={:.3f}".format(*g14[-2:]))
    g14_test = ROC.ROC_curve(g1_counts, g4_counts)
    # plt.plot(*g14_test[:2],label="0PM vs 0M test: {:.3f}".format(g14_test[-1]))
    plt.plot(*g14_I[:2], label="0PM vs INT: l={:.3f}, A={:.3f}".format(*g14_I[-2:]))
    
    plt.plot(*g14_m[:2],label="0PM vs 0M Merged: l={:.3f}, A={:.3f}".format(*g14_m[-2:]))
    plt.plot(*g14_I_m[:2], label="0PM vs INT Merged: l={:.3f}, A={:.3f}".format(*g14_I_m[-2:]))
    plt.gca().axhline(0, color='k', ls='dashed', zorder=-1)
    plt.legend()
    plt.show()
    
    hep.histplot(merged_g1_counts ,label = "g1")
    hep.histplot(merged_g4_counts ,label = "g4")
    hep.histplot(merged_int_counts,label = "INT")
    plt.legend()
    plt.title('Final Bins')
    plt.show()

def test_analytic_INT_only():
    files = (
            "../../test_data/out_g1.txt",
            "../../test_data/out_g4.txt",
            "../../test_data/out_g1g4.txt"
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
    
    g14_I = ROC.length_scale_ROC(g1_counts, np.abs(g1_counts).sum(), int_counts, np.abs(int_counts).sum())
    
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
    
    print("SUM INT positive negative before", np.maximum(0, data_int).sum(), np.minimum(0, data_int).sum())
    print("SUM INT positive negative after", np.maximum(0, merged_int_counts).sum(), np.minimum(0, merged_int_counts).sum())
    
    
    g14_I_m = ROC.length_scale_ROC(merged_g1_counts, np.abs(g1_counts).sum(), merged_int_counts, np.abs(data_int).sum())
    
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
    test_analytic()
    test_analytic_INT_only()
    
