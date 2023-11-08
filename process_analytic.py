import numpy as np
import pickle as pkl
import tqdm
import matplotlib.pyplot as plt
import mplhep as hep
hep.style.use('CMS')

def process_file(fname):
    with open(fname) as f:
        top_label = f.readline().strip().split(',')
        edges = [
            
        ]
        shape = []
        for i in range(0, len(top_label), 3):
            n, min, max = map(float, top_label[i:i+3])
            n = int(n)
            shape.append(n)
            bin_edges = np.linspace(min, max, n+1)
            edges.append(bin_edges)
        
        edges = np.array(edges, dtype=np.float32)
        centers = (edges[:,1:] + edges[:,:-1])/2
        print("edges\n",edges)
        print("centers\n", centers)
        with open(fname + '.pkl', 'wb+') as p:
            pkl.dump((tuple(shape), edges, centers), p)
        
        entries = f.readlines()
        # print(entries)
        n_total = np.prod(shape)
        # shape = (len(shape), np.max(shape))
        shape = tuple(shape)
        fp = np.memmap(fname + ".mm", mode='w+', dtype=np.float64, shape=shape)
        
        entries = np.fromiter(map(float, map(str.strip, entries)), dtype=np.float64)
        indices = map(lambda x: np.unravel_index(x, shape=shape), np.arange(n_total, dtype=np.int32))
        
        # print(len(entries), len(list(indices)))
        for index, entry in tqdm.tqdm(zip(indices, entries), desc="Dumping ", total=n_total, leave=True):
            fp[index] = entry
    
    return shape, edges

if __name__ == "__main__":
    process_file('test_data/out_g1.txt')
    process_file('test_data/out_g4.txt')
    shape, edges = process_file('test_data/out_g1g4.txt')
    
    data_g1 = np.memmap('test_data/out_g1.txt.mm',  mode='r', shape=shape, dtype=np.float64)
    data_g4 = np.memmap('test_data/out_g4.txt.mm',  mode='r', shape=shape, dtype=np.float64)
    data_14 = np.memmap('test_data/out_g1g4.txt.mm',  mode='r', shape=shape, dtype=np.float64)
    
    data_int = data_14.copy() - data_g1 - data_g4
    
    values = ["MZ1", "MZ2", "Cos1", "Cos2", "Phi"]
    for i in range(5):
        indiv_axis = tuple([j for j in range(5) if j != i])
        
        g1_proj =  np.sum(data_g1, axis=indiv_axis)
        g1_sum = g1_proj.sum()
        # g1_proj *= 100/g1_sum
        
        g4_proj =  np.sum(data_g4, axis=indiv_axis)
        g4_sum = g4_proj.sum()
        # g4_proj *= 100/g4_sum
        
        
        mixed_proj = np.sum(data_14, axis=indiv_axis)
        mixed_sum = mixed_proj.sum()
        # mixed_proj *= 
        
        int_proj = np.sum(data_int, axis=indiv_axis)
        # int_proj *= 100/mixed_sum
        
        hep.histplot(g1_proj, edges[i], label=r'$0^+$')
        hep.histplot(g4_proj, edges[i], label=r'$0^-$')
        hep.histplot(mixed_proj, edges[i], label=r'$(0^+ + 0^-)^2$')
        hep.histplot(int_proj, edges[i], label="Interference")
        plt.xlabel(values[i], fontsize=20)
        plt.gca().tick_params(axis='both', labelsize=15)
        plt.legend(fontsize=20)
        # plt.yscale('symlog')
        plt.gca().axhline(0, lw=2, zorder=-1)
        plt.tight_layout()
        plt.savefig(values[i] + "_analytic_proj.png")
        plt.show()