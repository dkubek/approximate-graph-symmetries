#!/usr/bin/env python
import numpy as np
import pickle
import sys
from pathlib import Path

def convert_instance(filename):
    with open(filename, 'rb') as fin:
        data = pickle.load(fin)
    
    outfile = filename.parent / (filename.stem + '.npz')
    print(outfile)
    np.savez(outfile, **{str(key): data[key] for key in data})

if __name__ == '__main__':
    convert_instance(Path(sys.argv[1]))
