import numpy as np
import os
import sys
sys.path.append(os.path.abspath("../"))
import merging.bin_optimizer as bo
import merging.place_from_map as pm
import metrics.ROC_curves
import tqdm
import cProfile


testing1 = np.hstack(
    (
        np.random.normal(3, 0.5, (200000,1)), 
        np.random.choice([-1, 1], size=(200000,1))*np.sinh(np.random.rand(200000,1)*2*np.pi)
    ),
)
testing1 = testing1[
    (testing1[:,0] > 0) & (testing1[:,0] < 10) & (testing1[:,1] > -1) & (testing1[:,1] < 1)
]


testing2 = np.hstack(
    (
        np.abs(np.random.normal(0, 3, (1000000,1))),
        np.cos(np.random.rand(1000000,1)*2*np.pi)
    )
)
testing2 = testing2[
    (testing2[:,0] > 0) & (testing2[:,0] < 10) & (testing2[:,1] > -1) & (testing2[:,1] < 1)
]

bins = np.linspace(0, 10, 31)
bins2 = np.linspace(-1,1, 31)

signal, *_ = np.histogram2d(testing1[:,0], testing1[:,1], (bins, bins2))
signal_sum = np.sum(signal)
signal = signal/signal_sum
bkg, *_ = np.histogram2d(testing2[:,0], testing2[:,1], (bins, bins2))
bkg_sum = np.sum(bkg)
bkg = bkg/bkg_sum



merger = bo.MergerNonlocal(
    (
        bins,
        bins2
    ),
    signal.copy(),
    bkg.copy(),
    # map_at=(
    #     1800,
    #     1700,
    #     1600,
    #     1500,
    #     1300,
    #     1000,
    #     900,
    #     800,
    #     700,
    #     500,
    #     350,
    #     238,
    #     80,
    #     90,
    #     70,
    #     60, 
    #     50,
    #     10,
    # )
)
# pr = cProfile.Profile()
# pr.enable()
new_bins = merger.run(3)
# pr.disable()
# pr.print_stats()
print(new_bins)
# signalNew, _ = np.histogram(testing1, new_bins)
# bkgNew, _ = np.histogram(testing2, new_bins)
# hep.histplot([signalNew, bkgNew], new_bins);
# ROC_curves.ROC_score(signalNew.astype(float), bkgNew.astype(float))

