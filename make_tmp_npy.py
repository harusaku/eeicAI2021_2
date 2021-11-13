### A file that makes one dummy train data with sizes of 1000 from each subset
### to allow debugging of clipdraw_DCgan.py before the generation finishes

import numpy as np

ls = ['airplane', 'apple', 'banana']

for word in ls:
    data = np.load('raw_data/full_numpy_bitmap_{}.npy'.format(word))
    with open('raw_data/{}.npy'.format(word), 'wb') as f:
        np.save(f, data[0:1000])