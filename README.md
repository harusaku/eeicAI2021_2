# 人工知能演習2021 第2ターム 2班

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
[CLIP](https://github.com/openai/CLIP)を用いて、全データのうちそのカテゴリー名との類似度が閾値を超えてるもののみを抽出する。
抽出する個数は各カテゴリーにつき最大1000個で、```./filtered_data```に```.npy```ファイルとして保存する。


### ある一つのカテゴリーのお絵かき画像を生成する
```
$ python3 sample/sample_DCgan.py -- category {category}
```
categoryオプションにあるカテゴリーを指定することによって、そのカテゴリーの画像を生成することができる。その他学習のオプションはファイル内の実装を参照。

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
