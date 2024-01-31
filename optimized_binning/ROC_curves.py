import numpy as np
import numba as nb

@nb.njit(nb.float64(nb.float64[:], nb.float64[:]), fastmath=True, cache=True)
def ROC_score(hypo_1, hypo_2):
    raw_ratio = hypo_1.copy()/hypo_2
    ratio_indices = np.argsort(raw_ratio)
    # ratio_indices = np.isfinite(raw_ratio[ratio_indices])[::-1]

    length = len(ratio_indices) + 1

    TPR = np.zeros(length)
    FPR = np.zeros(length)

    for n in nb.prange(length):
        above_cutoff = ratio_indices[n:]
        below_cutoff = ratio_indices[:n]

        TPR[n] = hypo_1[above_cutoff].sum()/(
            hypo_1[above_cutoff].sum() + hypo_1[below_cutoff].sum()
            ) #gets the indices listed

        FPR[n] = hypo_2[below_cutoff].sum()/(
            hypo_2[above_cutoff].sum() + hypo_2[below_cutoff].sum()
            )

    return np.trapz(TPR, FPR) - 0.5