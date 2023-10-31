import brunelle_merger.SUPER_ROC_Curves as ROC
import numpy as np
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

if __name__ == '__main__': #TEST ROC CURVES!!!
    x = np.linspace(0, 2*np.pi, 20)
    y1 = np.sin(x)
    y2 = np.sin(x)**2
    y3 = np.cos(x)**2
    
    ROC_maker = ROC.SUPER_ROC_Curves()
    # ROC_maker.add_ROC(y1.copy(), y2.copy(), "sin(x)^2 vs sin(x)")
    # ROC_maker.add_ROC(y1.copy(), y3.copy(), "sin(x) vs cos(x)^2")
    ROC_maker.add_ROC(y2.copy(), y3.copy(), "sin(x)^2 vs cos(x)^2")
    ROC_maker.plot_scores()
    ROC_maker.plot_ROCs()
    
    x,y,s = ROC.ROC_curve(y3.copy(), y2.copy())
    plt.plot(x,y, label='s')
    plt.legend()
    plt.show()