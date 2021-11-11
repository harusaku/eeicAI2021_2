import numpy as np
from PIL import Image

data = np.load("./data/apple.npy")
for i in range(len(data)):
    img_array = data[i].reshape(28,28)
    data_img = Image.fromarray(img_array)
    data_img.save("data/apple/apple_{}.jpg".format(i))
