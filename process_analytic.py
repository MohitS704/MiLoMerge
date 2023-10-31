import numpy as np
import pickle as pkl
import tqdm

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
        fp = np.memmap(fname + ".mm", mode='w+', dtype=np.float32, shape=shape)
        
        entries = np.fromiter(map(float, map(str.strip, entries)), dtype=np.float32)
        indices = map(lambda x: np.unravel_index(x, shape=shape), np.arange(n_total, dtype=np.int64))
        values = {}
        
        # print(len(entries), len(list(indices)))
        for index, entry in tqdm.tqdm(zip(indices, entries), desc="Dumping ", total=n_total, leave=True):
            fp[index] = entry

if __name__ == "__main__":
    process_file('test_data/out_g1.txt')
    process_file('test_data/out_g4.txt')
    process_file('test_data/out_g1g4.txt')