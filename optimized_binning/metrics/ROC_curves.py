import numpy as np
import numba as nb

@nb.njit()
def ROC_curve(sample1, sample2):
    ratios = np.argsort(sample1/sample2)
    PAC = np.zeros(len(sample1) + 1, dtype=np.float64)
    NAC = np.zeros(len(sample1) + 1, dtype=np.float64)
    
    PAC[1:] = np.cumsum(sample1[ratios])
    NAC[1:] = np.cumsum(sample2[ratios])

    TPR = PAC/(sample1.sum()) #vectorized calculation
    FPR = NAC/(sample2.sum())

    return TPR, FPR, np.abs(np.trapz(FPR, TPR))

@nb.njit()
def length_scale_ROC(sample1, sample2):
    if np.any(sample2 < 0):
        if np.any(sample1 < 0):
            raise ValueError("Need 1 positive-definite sample!")
        negative_counts = sample2
        positive_counts = sample1
    else:
        negative_counts = sample1
        positive_counts = sample2

    ratios = np.argsort(negative_counts/positive_counts)
    TPR = np.zeros(len(negative_counts) + 1, dtype=np.float64)
    FPR = np.zeros(len(negative_counts) + 1, dtype=np.float64)

    TPR[1:] = np.cumsum(negative_counts[ratios])
    FPR[1:] = np.cumsum(positive_counts[ratios])

    length = np.sqrt(np.diff(TPR)**2 + np.diff(FPR)**2).sum() #vectorized distance formula

    return TPR, FPR, length

