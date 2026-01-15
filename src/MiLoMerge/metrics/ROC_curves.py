import numpy as np
import numba as nb
from collections.abc import Iterable


@nb.njit("(Array(float64, 1, 'A'), Array(float64, 1, 'A'))", fastmath=True, cache=True)
def ROC_curve(sample1: Iterable[float], sample2: Iterable[float]):
    """A function to calculate the classical ROC curve given 2 distributions

    Parameters
    ----------
    sample1 : Iterable[float]
        The "signal" sample. Must be a 1-d array.
    sample2 : Iterable[float]
        The "background" sample. Must be the same size as sample1

    Returns
    -------
    tuple[Iterable[float], Iterable[float], float]
        Returns 2 arrays with the same size as sample1 indicating the True Positive Rate (TPR)
        and False Positive Rate (FPR) per-bin, as well as the Area Under the Curve (AUC)
    """
    ratios = np.argsort(sample1 / sample2)
    PAC = np.zeros(len(sample1) + 1, dtype=np.float64)
    NAC = np.zeros(len(sample1) + 1, dtype=np.float64)

    PAC[1:] = np.cumsum(sample1[ratios])
    NAC[1:] = np.cumsum(sample2[ratios])

    TPR = PAC / (sample1.sum())  # vectorized calculation
    FPR = NAC / (sample2.sum())

    return TPR, FPR, np.trapz(FPR, TPR)


@nb.njit("(Array(float64, 1, 'A'), Array(float64, 1, 'A'))", fastmath=True, cache=False)
def LOC_curve(sample1: Iterable[float], sample2: Iterable[float]):
    """A function to calculate the LOC curve described in (ARXIV LINK)
    given 2 distributions.

    Parameters
    ----------
    sample1 : Iterable[float]
        The "signal" sample. Must be a 1-d array.
    sample2 : Iterable[float]
        The "background" sample. Must be the same size as sample1

    Returns
    -------
    tuple[Iterable[float], Iterable[float], float]
        Returns 2 arrays with the same size as sample1 indicating the True Positive Rate (TPR)
        and False Positive Rate (FPR) per-bin, as well as the Length of the Curve (LoC).

    Raises
    ------
    ValueError
        If both samples are not wholly positive, raise an error. At least one
        sample must be completely positive.
    """
    if np.any(sample2 < 0):
        if np.any(sample1 < 0):
            raise ValueError("Needs 1 positive sample!")
        negative_counts = sample2
        positive_counts = sample1
    else:
        negative_counts = sample1
        positive_counts = sample2

    ratios = np.argsort(negative_counts / positive_counts)
    TPR = np.zeros(len(negative_counts) + 1, dtype=np.float64)
    FPR = np.zeros(len(negative_counts) + 1, dtype=np.float64)

    TPR[1:] = np.cumsum(negative_counts[ratios])
    FPR[1:] = np.cumsum(positive_counts[ratios])

    length = np.sqrt(
        np.diff(TPR) ** 2 + np.diff(FPR) ** 2
    ).sum()  # vectorized distance formula

    return TPR, FPR, length
