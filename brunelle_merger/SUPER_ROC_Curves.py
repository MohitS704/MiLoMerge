import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches
import warnings
import mplhep as hep


def length_scale_ROC(sample1, sample2):
    sample1 = np.array(sample1, dtype=float)
    sample2 = np.array(sample2, dtype=float)
    
    if np.any(sample2 < 0):
        if np.any(sample1 < 0):
            raise ValueError("Need 1 positive-definite sample!")
        
        print("Swapping samples 1 and 2")
        sample1, sample2 = sample2.copy(), sample1.copy() #sample1 should be the negative one by construction
    
    hypo1_counts = sample1.copy()/np.abs(sample1).sum()
    hypo2_counts = sample2.copy()/np.abs(sample2).sum()
    
    # ratio = hypo2_counts/hypo1_counts
    
    py_ratio = hypo2_counts/hypo1_counts
    # py_ratio = py_ratio[np.isfinite(py_ratio)]
    # py_ratio = list(enumerate(py_ratio))
    
    # py_ratio = sorted(py_ratio, key=lambda x: x[1])
    # ratios = np.array(py_ratio)[:,0].astype(int) #gets the bin indices only for the ordered ratio pairs
    
    ratios = np.argsort(py_ratio)[::-1]
    
    # plt.plot(py_ratio[ratios])
    # plt.plot(np.sort(hypo2_counts/hypo1_counts))
    # plt.title('og ratio')
    # plt.show()
    
    
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
    
    
    turning_point = np.argmin(TPR)
    
    TPR_bot, TPR_top = TPR[turning_point:], TPR[:turning_point]
    FPR_bot, FPR_top = FPR[turning_point:], FPR[:turning_point]
    
    area_below_bottom = abs(np.trapz(FPR_bot, TPR_bot)) #abs because trapz is negative when going backwards but we want the actual area
    area_below_top = abs(np.trapz(FPR_top, TPR_top))
    
    area_inside = area_below_top - area_below_bottom
    
    length_arr = np.sqrt(np.diff(TPR)**2 + np.diff(FPR)**2) #vectorized distance formula
    
    length = length_arr.sum()
    
    y = abs(np.min(TPR))
    x_plus_y = abs(np.max(TPR))
    x = abs(x_plus_y - y)
    
    yBar = abs(np.min(FPR))
    xBar_plus_yBar = abs(np.max(FPR))
    xBar = abs(xBar_plus_yBar - yBar)
    
    maximum_length = x + y + yBar + xBar
    
    print("length = {:.02f}".format(length))
    print("max length: {:.02f} = {:.02f} + {:.02f} + {:.02f} + {:.02f}".format(maximum_length, x, y, yBar, xBar))
    
    score = length/maximum_length
    
    
    
    print("Score = {:.3f}".format(score))
    
    return TPR, FPR, score, area_inside

def ROC_curve(sample1, sample2):
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