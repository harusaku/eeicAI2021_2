import torch
import clip
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import itertools
import os

#######################################
## 使うカテゴリの名前を代入          ##
#######################################
category = "hand"

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

dataset = np.load("./raw_data/" + category + ".npy")
text = clip.tokenize([category]).to(device)
similarity = []
good_dataset = []
bad_dataset = []
good_count = 0 # しきい値を超えなかったやつ
bad_count = 0 # しきい値を超えなかったやつ
data_count = 0

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
        print("sim:",sim)

        similarity.append(sim)

    data_count += 1
    
    if sim >= 0.25 and good_count < 16:
        good_dataset.append(data)
        good_count += 1
    elif(bad_count < 16):
        bad_dataset.append(data)
        bad_count += 1
    if good_count == 16 and bad_count == 16:
        break

    if data_count > 100000:
        break
# fig = plt.figure()
# fig.hist(similarity)
# fig.savefig("hist_"+category+"jpg")

for s in ["good", "bad"]:
    fig, ax = plt.subplots(4, 4, figsize=(6,6))
    flg = 0
    for i, j in itertools.product(range(4), range(4)):
        ax[i,j].get_xaxis().set_visible(False)
        ax[i,j].get_yaxis().set_visible(False)
    for k in range(16):
        i = k//4
        j = k%4
        ax[i,j].cla()
        if s == "good":
            if len(good_dataset) < 16:
                flg=1
            else:
                ax[i,j].imshow(good_dataset[i*4+j].reshape(28,28), cmap='Greys')
        else:
            if len(bad_dataset) < 16:
                flg = 1
            else:
                ax[i,j].imshow(bad_dataset[i*4+j].reshape(28,28), cmap='Greys')
    if flg == 1:
        continue
    label = s

    fig.text(0.5, 0.04, label, ha='center')
    os.makedirs("sample_clip_fig",exist_ok=True)
    fig.savefig("sample_clip_fig/"+category+"_"+s+"_dataset"+'.png')
print("good_count:",good_count)
