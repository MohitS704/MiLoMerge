import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches
import warnings
import mplhep as hep

class SUPER_ROC_Curves(object):
    def __init__(self) -> None:
        """A nice collection of ROC curves
        """
        self.curves = {}
    
    def add_ROC(self, sample1, sample2, name=None):
        """Generates ROC curves based off of the samples you give it and an optional name.
        Will return the requisite values, but will also store them in self.curves

        **>= 1 sample must be *completely* positive**
        
        Parameters
        ----------
        sample1 : numpy.ndarray
            This is an array of histogram bin counts
        sample2 : numpy.ndarray
            This is an array of histogram bin counts
        name : str, optional
            The name you want to give you sample. If none - the name will be an assigned number, by default None
            
        Returns
        -------
        Tuple(numpy.ndarray, numpy.ndarray, float)
            A tuple of the Sample 1 axis, Sample 2 axis, and the "score" between these 2 samples
        """
        sample1 = np.array(sample1)
        sample2 = np.array(sample2)
        
        if np.any(sample2 < 0):
            warnings.warn("Swapping sample 1 and sample 2...")
            if np.any(sample1 < 0):
                raise ValueError(">= 1 sample must be completely positive!")
            sample1, sample2 = sample2.copy(), sample1.copy() #sample1 should be the negative one by construction
        
        isNegative = False
        if np.any(sample1 < 0):
            isNegative = True
        
        hypo1_counts = sample1.copy()/np.abs(sample1).sum()
        hypo2_counts = sample2.copy()/np.abs(sample2).sum()
        
        division_terms = hypo2_counts/hypo1_counts
        
        division_terms[~np.isfinite(division_terms)] = 0
        
        ratios = np.array(
            sorted(
                list(enumerate(division_terms)), key=lambda x: x[1]
            )
        )
        print("ratios")
        print(ratios)
        indices = ratios[:,0].astype(int) #gets the bin indices only for the ordered ratio pairs
    
        length = len(indices) + 1 #the extra value allows for the addition of the final end point
        
        PAC = np.zeros(length) #"positive" above cutoff
        PAC_numerator = np.zeros(length)
        PBC = np.zeros(length) #"positive" below cutoff
        NAC = np.zeros(length) #"negative" above cutoff
        NAC_numerator = np.zeros(length)
        NBC = np.zeros(length) #"negative" below cutoff
        
        for n in range(length):
            above_cutoff = indices[n:]
            below_cutoff = indices[:n]
            
            PAC[n] = np.abs(hypo1_counts[above_cutoff]).sum() #gets the indices listed
            PAC_numerator[n] = hypo1_counts[below_cutoff].sum() #only the numerator preserves sign
            PBC[n] = np.abs(hypo1_counts[below_cutoff]).sum()
            
            NAC[n] = np.abs(hypo2_counts[above_cutoff]).sum()
            NAC_numerator[n] = hypo2_counts[below_cutoff].sum()
            NBC[n] = np.abs(hypo2_counts[below_cutoff]).sum()
        
        
        TPR = PAC_numerator/(PAC + PBC) #vectorized calculation
        FPR = NAC_numerator/(NAC + NBC)
        
        if isNegative:
            turning_point = np.where(TPR == TPR.min())[0][-1] #this is the turning point change in sign indicates turning point! Sign changes at max negative value
        else:
            turning_point = 0
        print(turning_point)
        
        negative_gradient = TPR[:turning_point], FPR[:turning_point] #now there are two curves, positive and negative
        positive_gradient = TPR[turning_point:], FPR[turning_point:]
        
        area_below_top_line = np.trapz(*positive_gradient[::-1])
        area_below_bottom_line = np.trapz(*negative_gradient[::-1])*-1
        area_inside = area_below_top_line - area_below_bottom_line
        
        print("areas:", area_below_top_line, area_below_bottom_line, area_inside)
        
        plt.plot(*negative_gradient, marker='o')
        plt.plot(*positive_gradient, marker='o')
        plt.title("OG")
        plt.show()
        
        # apex_x = TPR[turning_point]
        # apex_y = FPR[turning_point]
        # # TRANSFORM THE CURVES TO THE POSITIVE REGIME
        # negative_gradient = np.array((-apex_x + negative_gradient[0], negative_gradient[1] + apex_y))
        # positive_gradient = np.array((-apex_x + positive_gradient[0], positive_gradient[1] - apex_y))
        
        
        # #ORIGINAL PLOT
        BBB = np.abs(np.min(TPR))
        
        if isNegative:
            TPR /= -BBB
        
        if isNegative:
            score = 1 - area_inside/BBB
        else:
            score = 0.5 - area_inside
            score /= 0.5
            
        print("score=", score)
        
        plt.title("OG OG")
        plt.plot(TPR[np.isfinite(TPR) & np.isfinite(FPR)], FPR[np.isfinite(TPR) & np.isfinite(FPR)], marker='o')
        ax = plt.gca()
        ax.fill_between(*positive_gradient, 0, color='red', alpha=0.5)
        ax.fill_between(*negative_gradient, 0, color='white')
        
        rect = matplotlib.patches.Rectangle((0,0),1,1, lw=3, zorder=np.inf)
        rect.fill = False
        ax.add_patch(rect)
        
        plt.show()
        
        if name == None:
            name = str(len(self.curves.keys()))
        
        self.curves[str(name)] = TPR, FPR, score
        
        return TPR, FPR, score
    
    def plot_scores(self):
        """Plots the scores
        """
        names = []
        scores = []
        for named_curve in self.curves:
            names.append(named_curve)
            scores.append(self.curves[named_curve][2])
        plt.scatter(names, scores)
        plt.show()
    
    def plot_ROCs(self):
        """Plots every ROC Curve on a single plot
        """
        x_vals = []
        y_vals = []
        scores = []
        names = []
        for named_curves in self.curves:
            names.append(named_curves)
            x,y,s = self.curves[named_curves]
            print(named_curves, x, y)
            x_vals.append(x)
            y_vals.append(y)
            scores.append(s)
        
        for i in range(len(x_vals)):
            plt.plot(x_vals[i], y_vals[i], label=names[i] + ": {:.3f}".format(scores[i]))
            
        plt.legend()
        plt.tight_layout()
        plt.show()

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