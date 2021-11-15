import numpy as np
import glob
import os

files = glob.glob("../filtered_data/*.npy")
all_data = 0
full_count = 0
count = 0
for file in files:
    if os.stat(file).st_size == 0:
        continue
    arr = np.load(file)
    count += 1
    print(file.replace("../filtered_data/","").replace(".npy",""), ": the number of data is",len(arr))
    all_data += len(arr)
    if(len(arr)==1000):
        full_count += 1
    

print("sum is",all_data,count,"category is finished")
print(full_count,"/", len(files))