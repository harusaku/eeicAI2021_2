from text2list import text2list
import urllib.request
import os

categories = text2list("categories.txt")
path = "../raw_data/"
os.makedirs("raw_data",exist_ok=True)
finished = 0
all = len(categories)

for category in categories:
    url = "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/" + category.replace(" ", "%20") + ".npy"
    with urllib.request.urlopen(url) as u:
        with open(path + category.replace(" ", "_") +'.npy', 'bw') as o:
            o.write(u.read())
    finished+=1
    print(category.replace(" ","_")+".npy"+" is created! [{}/{}]".format(finished, all))