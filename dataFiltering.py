import torch
import clip
from PIL import Image
import numpy as np
from text2list import text2list
import os
import pathlib

# 使うカテゴリの名前を取得 
categories = text2list("categories.txt")

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

categories_size = len(categories)
finished = 0

for category in categories:
    if(os.path.isfile("filtered_data/" + category.replace(" ","_") + ".npy")):
        continue
    else:
        path = pathlib.Path("filtered_data/" + category.replace(" ","_") + ".npy")
        path.touch()

    dataset = np.load("raw_data/" + category.replace(" ","_") + ".npy")
    text = clip.tokenize([category]).to(device)
    good_dataset = []
    good_count = 0
    max_data_num = 1000

    # 類似度が0.25を超えるデータを抽出
    for data in dataset:
        image = preprocess(Image.fromarray(data.reshape(28,28)).convert("L").convert("RGB")).unsqueeze(0).to(device)

        with torch.no_grad():
            image_features = model.encode_image(image)
            text_features = model.encode_text(text)

            # Pick the top 5 most similar labels for the image
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            sim = (image_features @ text_features.T).cpu().numpy()
            sim = np.squeeze(sim)

        if sim >= 0.25:
            good_dataset.append(data)
            good_count += 1
        if good_count == 1000:
            break
    
    # npyファイルとして抽出したデータを保存
    os.makedirs("filtered_data", exist_ok=True)
    good_dataset = np.array(good_dataset)
    np.save("./filtered_data/" + category.replace(" ","_") + ".npy", good_dataset)

    finished += 1

    print("The number of dataset is {} (category is ".format(good_count) + category + ")")
    print("./filtered_data/" + category.replace(" ","_") + ".npy is created! [{}/{} is finished.]".format(finished, categories_size))