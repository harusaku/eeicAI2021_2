# 人工知能演習2021 第2ターム 2班
# Artificial Intelligence Experiment Project, UTokyo EEIC2021 Team2
(English follows Japanese)

## 概要

[Quick, Draw!](https://quickdraw.withgoogle.com/)のデータセットとGANを用いて、画像の生成を行った。

我々の班の目標としては、このデータセットにないカテゴリーに関しても人間が書いたようなお絵かき画像が生成できるようにしたいということであった。そこで、[CLIP](https://openai.com/blog/clip/)を用いて、各自然言語に対応するベクトルを出力し、それをGANの入力に加えることで出力結果をうまく区別できるのではないかと思って試してみることにした。

## 実行方法
### データの取得方法

- 全データの取得
    ```
    $ python3 ./dataprocess/dataDownloader.py
    ```
    [Quick, Draw!](https://github.com/googlecreativelab/quickdraw-dataset)のデータセットのうちNumpy bitmaps(```.npy```)の全カテゴリーを```./raw_data/```にダウンロードする。

- データのフィルタリング
    ```
    $ python3 ./dataprocess/dataFiltering.py
    ```
    CLIPを用いて、全データのうちそのカテゴリー名との類似度が閾値を超えてるもののみを抽出する。
抽出する個数は各カテゴリーにつき最大1000個で、```./filtered_data```に```.npy```ファイルとして保存する。
CLIPのインストールが必要。(インストール方法はCLIPの[github](https://github.com/openai/CLIP)を参照)


### ある一つのカテゴリーのお絵かき画像を生成する
```
$ python3 ./sample/sample_DCgan.py --category {category}
```
CLIPを用いずに、categoryオプションにあるカテゴリーを指定することによって、そのカテゴリーの画像を生成することができる。その他学習のオプションはファイル内の実装を参照。
特にload_pathオプションを```./raw_data/```にすることで元データすべてを使って学習することができる。

### CLIPを含めたDCCGAN(Deep Convolutional Conditional GAN)
```
$ python3 clipdraw_DCgan.py
```
これによりCLIPによって生成されるベクトルを入力としたDCCGANを学習させることができる。学習のオプションはファイル内の実装を参照。
(うまくいってないです、、、）

### 上で学習した生成モデルを実際に動かす
```
$ python3 test_GAN.py
```
入力で受け取った文章をCLIPに通し、それを学習済みモデルに入力することで、画像が生成される。ここで、学習時に用いなかった単語でもそれっぽいお絵かき画像が出てくるかどうか確認できる。


## Briefing

This project uses the doodling dataset available at [Quick, Draw!](https://quickdraw.withgoogle.com/) to train DCGAN/DCCGAN and generate similar images.

When the training dataset is limited to one label, our DCGAN manages to generate images similar to human-drawn doodles.

Our goal is to generate doodling images corresponding to any natural language outside the labels of the Quick, Draw! dataset using [CLIP](https://openai.com/blog/clip/) embedded vectors. Since CLIP embeddings contain both image and text information, we expect that an embedded text segment can be retrieved as an image if trained properly. 

By the end of this experiment in 2021/11, we managed to retrieve an abstract shape corresponding to simple objects (e.g. "apple" <=> a circle, "glasses" <=> something layed horizontally, etc.).


## Getting Started
### Obtaining the dataset

- Getting all data
    ```
    $ python3 ./dataprocess/dataDownloader.py
    ```
    The [Quick, Draw!](https://github.com/googlecreativelab/quickdraw-dataset) dataset will be downloaded as Numpy bitmaps(```.npy```) in ```./raw_data/``` folder.

- Data filtering
    ```
    $ python3 ./dataprocess/dataFiltering.py
    ```
    The dataset is filtered using CLIP, leaving only images matching their labels above a certain threshold as given by CLIP's cosine similarity.
    Each category has at max 1000 images saved at ```./filtered_data/``` as numpy files


### Generate doodles by training a DCGAN with images from one category
```
$ python3 ./sample/sample_DCgan.py --category {category}
```
You 
To train this DCGAN, specify one category from Quick, Draw! dataset (check [categories.txt](https://github.com/harusaku/eeicAI2021_2/blob/main/categories.txt) with the --category option and similar images will be generated. Refer to the py file for other training options.

### Train the DCCGAN with CLIP embedded vectors
```
$ python3 clipdraw_DCgan.py
```
(This model is still behaving poorly.)
You will need to install the CLIP Python package first. ([Check CLIP's github page](https://github.com/openai/CLIP))
This trains the DCCGAN with the categories embedded to vectors using CLIP. Refer to the py file for other trianing options.

### Generate images with the model trained above
```
$ python3 test_GAN.py
```
This will generate images with any text inputted by embedding the text via CLIP and sending the vector to the generator trained above.
The output images are noisy but the some very abstract outlines of both (1) trained text and (2) unknown text can be observed.
