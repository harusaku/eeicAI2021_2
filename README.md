# 人工知能演習2021 第2ターム 2班

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
