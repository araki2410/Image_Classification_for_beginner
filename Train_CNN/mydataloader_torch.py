import torch
import os
from PIL import Image
import numpy

class CrossSquare_9x9(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None):
        self.image_list = []
        self.data_dir = data_dir
        self.transform = transform

        self.label_list = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]

        for label in self.label_list:
            ## クラス分けされたディレクトリからファイルの名前を読み出す
            labeled_image_list = os.listdir(os.path.join(data_dir, label))
            for image_name in labeled_image_list:
                if image_name[-4:] == ".png": ## データセットdir中に画像以外も存在するためエスケープ
                    self.image_list.append([image_name, label]) ## [画像名, クラスdir名]で一覧を保存

    def __len__(self):
        ## 外部から呼び出された際にデータの数を返す
        ## データセットディレクトリの中のファイル数がデータの数のはず
        return len(self.image_list)

    def __getitem__(self, idx):
        ## self.image_paths から画像のパスとラベルを読み出す
        filename = self.image_list[idx][0] # idx番目のファイル名
        label = self.image_list[idx][1]    # idx番目のファイルの正解ラベル
        image_path = os.path.join(self.data_dir, label, filename) # ディレクトリパスと画像名を連結

        ## 連結したパスから画像を読み出す
        image_data = Image.open(image_path)
        ## 読み込んだ画像の前処理 (左右反転やリサイズ、クロップなどやりたいことをやる)
        image_data = self.transform(image_data)
        gt_label = self.label_list.index(label) # Ground Truth. int

        ## 画像データと正解ラベルのインデックス(整数正解値)を返す
        return image_data, gt_label
