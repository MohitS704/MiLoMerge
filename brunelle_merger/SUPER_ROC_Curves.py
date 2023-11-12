import numpy as np


def length_scale_ROC(sample1, target_integral1, sample2, target_integral2):
    """This is a ROC curve that uses the length to calculate things

    Parameters
    ----------
    sample1 : numpy.ndarray
        A numpy array of the histogram bin counts for your first sample
    target_integral1 : float
        The desired integral of your sample
        
        _EXAMPLE_: un-merged interference has an absolute area of 20, but the merged interference has an absolute area of 10.
        Input 20 as target_integral1
    sample2 : numpy.ndarray
        A numpy array of the histogram bin counts for your second sample
    target_integral2 : float
        The desired integral of your sample

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray, float, float]
        The TPR and FPR arrays (to be treated as x and y),
        as well as the length score and the area score of the ROC curve.
        
        NOTE: for the positive case this area is the area __below__ the curve! 

    Raises
    ------
    ValueError
        At least one of the samples should be positive definite
    """
    sample1 = np.array(sample1, dtype=float).ravel()
    sample2 = np.array(sample2, dtype=float).ravel()
    
    if np.any(sample2 < 0):
        if np.any(sample1 < 0):
            raise ValueError("Need 1 positive-definite sample!")
        
        print("Swapping samples 1 and 2")
        sample1, sample2 = sample2.copy(), sample1.copy() #sample1 should be the negative one by construction
        target_integral1, target_integral2 = target_integral2, target_integral1
    
    print("NORMS:", np.abs(sample1).sum()/target_integral1, np.abs(sample2).sum()/target_integral2)
    
    hypo1_counts = sample1.copy()/np.abs(sample1).sum()
    hypo2_counts = sample2.copy()/np.abs(sample2).sum()
    
    print("NORMS:", np.abs(hypo1_counts).sum(), np.abs(hypo2_counts).sum())
        
    py_ratio = hypo2_counts/hypo1_counts
    
    ratios = np.argsort(py_ratio)[::-1]
    
    
    length = len(ratios) + 1 #the extra value allows for the addition of the final end point
    
    PAC = np.zeros(length) #"positive" above cutoff
    PAC_numerator = np.zeros(length)
    PBC = np.zeros(length) #"positive" below cutoff
    NAC = np.zeros(length) #"negative" above cutoff
    NAC_numerator = np.zeros(length)
    NBC = np.zeros(length) #"negative" below cutoff
    
    for n in range(length):
        above_cutoff = ratios[n:]
        below_cutoff = ratios[:n]
        
        PAC[n] = np.abs(hypo1_counts[above_cutoff]).sum() #gets the indices listed
        PAC_numerator[n] = hypo1_counts[above_cutoff].sum() #only the numerator preserves sign
        PBC[n] = np.abs(hypo1_counts[below_cutoff]).sum()
        
        NAC[n] = np.abs(hypo2_counts[above_cutoff]).sum()
        NAC_numerator[n] = hypo2_counts[above_cutoff].sum()
        NBC[n] = np.abs(hypo2_counts[below_cutoff]).sum()


    TPR = PAC_numerator/(PAC + PBC) #vectorized calculation
    FPR = NAC_numerator/(NAC + NBC)
    
    TPR[~np.isfinite(TPR)] = 0
    FPR[~np.isfinite(FPR)] = 0
    
    y = abs(np.min(TPR))
    x_plus_y = abs(np.max(TPR))
    x = abs(x_plus_y - y)
    
    yBar = abs(np.min(FPR))
    xBar_plus_yBar = abs(np.max(FPR))
    xBar = abs(xBar_plus_yBar - yBar)
    
    maximum_length = x + y + yBar + xBar #max length should be the same as the original sample given that the overall integral should be preserved
    
    norm_fac = (np.abs(sample1).sum()/target_integral1) #scales by the total absolute integral of the original sample
    TPR *= norm_fac
    
    turning_point = np.argmin(TPR)
    
    TPR_bot, TPR_top = TPR[turning_point:], TPR[:turning_point]
    FPR_bot, FPR_top = FPR[turning_point:], FPR[:turning_point]
    
    area_below_bottom = abs(np.trapz(FPR_bot, TPR_bot)) #abs because trapz is negative when going backwards but we want the actual area
    area_below_top = abs(np.trapz(FPR_top, TPR_top)) #another abs because why not! extra protection
    
    area_inside = area_below_top - area_below_bottom
    
    length_arr = np.sqrt(np.diff(TPR)**2 + np.diff(FPR)**2) #vectorized distance formula
    
    length = length_arr.sum()
    
    score = length/maximum_length
    score *= length
        
    return TPR, FPR, score, area_inside

def ROC_curve(sample1, sample2):
    """This function produces a ROC curve from an attribute like phi, cos(theta1), D_{0-}, etc.

    Parameters
    ----------
    sample1 : numpy.ndarray
        The first data sample for your attribute. This is your "True" data
    sample2 : numpy.ndarray
        The second data sample for your attribute. This if your "False" data
    

    Returns
    -------
    tuple(numpy.ndarray, numpy.ndarray, float)
        returns the true rate, the false rate, and the area under the curve (assuming true rate is the x value)
    """
    
    sample1 = np.array(sample1)
    sample2 = np.array(sample2)
    hypo1_counts = sample1.copy()/sample1.sum()
    hypo2_counts = sample2.copy()/sample2.sum()
        
    
    # print(list(g1_phi_counts))
    # print()
    # print(list(g4_phi_counts))
    
    ratios = sorted(
        list(enumerate(hypo2_counts/hypo1_counts)), key=lambda x: x[1], reverse=True
    )
    # print(ratios)
    
    ratios = np.array(ratios)[:,0].astype(int) #gets the bin indices only for the ordered ratio pairs
    ratios = ratios[np.isfinite(ratios)]
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