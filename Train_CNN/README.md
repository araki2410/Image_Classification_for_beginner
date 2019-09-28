Convolutional Neural Network
====
# フレームワークを用いてネットワークを構築する
Python3

## 環境設定
Anaconda や Docker などのこだわりがある人は自由に。

以下でできない場合は適宜python3, pip などについて調べてインストールする必要があります。

### PyTorch

'''shell
$ pip3 install torch
Collecting torch
  Downloading https://files.pythonhosted.org/packages/ba/88/7640344d841e97b9a1531385caac39d984b2c6f4abd1376e1ce0de3a0933/torch-1.2.0-cp37-none-macosx_10_7_x86_64.whl (59.9MB)
:
:
Successfully installed torch-1.2.0

$ python3
>>> import torch
>>> torch.__version__
'1.2.0'
'''

### Matplotlib
'''shell
$ pip3 install matplotlib
Collecting matplotlib
:
:
Successfully installed cycler-0.10.0 kiwisolver-1.1.0 matplotlib-3.1.1 pyparsing-2.4.2 python-dateutil-2.8.0 six-1.12.0

$ python3
>>> import matplotlib
>>> matplotlib.__version__
'3.1.1'
'''

### PIL (Python Imaging Library)
'''shell
$ pip3 install pillow
Collecting pillow
:
:
Installing collected packages: pillow
Successfully installed pillow-6.1.0

$ python3 
>>> import PIL
>>> PIL.__version__
'6.1.0'
'''

### TorchVision
'''shell
$ pip3 install torchvision
Collecting torchvision
:
:
Installing collected packages: torchvision
Successfully installed torchvision-0.4.0
'''

## 全体の流れ (例)

 - 必要なモジュールのimport
 - 初期値, 変数の設定
 - ネットワーク (model, architecture) の構築/読み込み
 - データセットの読み込み (画像パスの読み込み)
   - 学習用データ
   - 評価/テスト用データ
   - (デモ用データ)
 - 最適化手法の設定/決定
 - 損失関数 (loss function) の設定/決定
 - 学習ループの構築
 - 評価ループの構築
 - メイン関数の構築
   - イニシャライズ
   - 学習/評価ループ呼び出し
      - 学習経過出力/(保存)
      - (パラメータ調整関数構築/呼び出し)
   - 学習結果保存


順番を入れ替えたり処理を増やしたり減らしたり、効率化を計るのも大事

