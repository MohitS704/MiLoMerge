import numpy as np

def print_msg_box(msg, indent=1, width=0, title=""):
    """returns message-box with optional title.
    Ripped from https://stackoverflow.com/questions/39969064/how-to-print-a-message-box-in-python
    
    Parameters
    ----------
    msg : str
        The message to use
    indent : int, optional
        indent size, by default 1
    width : int, optional
        box width, by default 0
    title : str, optional
        box title, by default ""
    """
    
    lines = msg.split('\n')
    space = " " * indent
    if not width:
        width = max(map(len, lines))
    box = f'╔{"═" * (width + indent * 2)}╗\n'  # upper_border
    if title:
        box += f'║{space}{title:<{width}}{space}║\n'  # title
        box += f'║{space}{"-" * len(title):<{width}}{space}║\n'  # underscore
    box += ''.join([f'║{space}{line:<{width}}{space}║\n' for line in lines])
    box += f'╚{"═" * (width + indent * 2)}╝'  # lower_border
    return box

def merge_bins(target, bins, *counts, **kwargs):
    """Merges a set of bins that are given based off of the counts provided
    Eliminates any bin with a corresponding count that is less than the target
    Useful to do merge_bins(\*np.histogram(data), ...)
    
    
    Parameters
    ----------
    counts : numpy.ndarray
        The counts of a histogram
    bins : numpy.ndarray
        The bins of a histogram
    target : int, optional
        The target value to achieve - any counts below this will be merged, by default 0
    ab_val : bool, optional
        If on, the target will consider the absolute value of the counts, not the actual value, by default True
    drop_first : bool, optional
        If on, the function will not automatically include the first bin edge, by default False

    Returns
    -------
    Tuple(numpy.ndarray, numpy.ndarray)
        A np.histogram object with the bins and counts merged

    Raises
    ------
    ValueError
        If the bins and counts are not sized properly the function will fail
    """
    
    drop_first = kwargs.get('drop_first',False)
    ab_val = kwargs.get('ab_val', True)
    
    new_counts = []
    [new_counts.append([]) for _ in counts]
    
    counts = np.vstack(counts)
    
    if any([len(bins) != len(count) + 1 for count in counts]):
        errortext = "Length of bins is {:.0f}, lengths of counts are ".format(len(bins))
        errortext += " ".join([str(len(count)) for count in counts])
        errortext += "\nlen(bins) should be len(counts) + 1!"
        raise ValueError("\n" + errortext)
    
    
    if not drop_first:
        new_bins = [bins[0]] #the first bin edge is included automatically if not explicitly stated otherwise
    else:
        new_bins = []
    
    if ab_val:
        counts = np.abs(counts)
    
    
    i = 0
    while i < len(counts[0]):
        summation = np.zeros(len(counts))
        start = i
        # print("starting iteration at:", i)
        # print("Current running sum is ( {:.3f}, {:.3f} )".format(np.sum(new_counts[0]), np.sum(new_counts[1])))
        # print("Running sum should be ( {:.3f}, {:.3f} )".format(*np.sum(counts[:,:i], axis=1)))
        while np.any(summation <= target) and (i < len(counts[0])):
            summation += counts[:,i]
            i += 1
        # print("Merged counts", start, "through", i-1)
        
        if drop_first and len(new_bins) == 0:
            first_bin = max(i - 1, 0)
            new_bins += [bins[first_bin]]
            
        if not( np.any(summation <= target) and (i == len(counts[0])) ):
            for k in range(len(counts)):
                new_counts[k] += [np.sum(counts[k][start:i])]
            new_bins += [bins[i]]
        else:
            for k in range(len(counts)):
                new_counts[k][-1] += np.sum(counts[k][start:i])
            new_bins[-1] = bins[i]
        # print("Current running sum is ( {:.3f}, {:.3f} )".format(np.sum(new_counts[0]), np.sum(new_counts[1])))
        # print("Running sum should be ( {:.3f}, {:.3f} )".format(*np.sum(counts[:,:i], axis=1)))
        # print()
        # print()
    return np.vstack(new_counts), np.array(new_bins)

def Unroll_ND_histogram(counts, *bins, bkg=False):
    counts = np.array(counts)
    integral = np.sum(counts)
    super_bin_count = 1
    for b in bins:
        super_bin_count *= len(b)
    
    filler = integral/(10*super_bin_count)
    one_D_counts = np.zeros(super_bin_count)
    _, bins = np.histogram([], super_bin_count, [0, super_bin_count])
    
    indk = 0
    for index, bin_count in np.ndenumerate(counts):
        fillable_value = bin_count
        if bin_count <= 0 and bkg:
            fillable_value = filler
        elif bin_count < 0 and not bkg:
            fillable_value = 0
        
        one_D_counts[indk] = fillable_value
        indk += 1
    
    return one_D_counts, bins