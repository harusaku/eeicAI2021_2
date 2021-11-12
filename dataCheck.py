import numpy as np
import glob

files = glob.glob("filtered_data/*.npy")
for file in files:
    arr = np.load(file)
    print(file, len(arr))