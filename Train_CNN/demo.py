import torch ## DeepLearning 用フレームワーク
import torch.nn as nn # torch.nn を何回も呼ぶので nn に省略
import numpy # 行列演算
from torchvision import transforms # 画像の前処理用関数
import os # OS インターフェース. ファイルの読み書きに用いる. 今回はディレクトリ参照に利用.
from PIL import Image
import mycnn_torch # 自分で作成したネットワーク. 同じディレクトリにmycnn_torch.pyを用意

trained_model = "mytrained.model"
input_imagefile = "cross.png"
dataset_dir = "../Dataset"
classes = [f for f in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, f))]
model = mycnn_torch.MyClassifyNet() # 自作したネットワークを読み込む

transform = transforms.Compose([
    ## 画像に施す前処理を決める
    ## (データ増強のためのパディングや画像反転, 入力を統一するためのリサイズ等)
    ## you can add other transformations in this list
    # transforms.Resize(16),
    # transforms.Pad(2),
    # transforms.RandomCrop((16)),
    transforms.ToTensor() # 必須. 読込画像を行列とする. しない場合,後のenumerate()でループ処理できない
])

#print(model)

## Dateload 自作したデータローダで、自作したデータセット (画像のパス) を読み込む
#dataset = mydataloader_torch.CrossSquare_9x9(dataset_dir, transform)

## Loss & Oprimizer
#loss_function = nn.CrossEntropyLoss()
#optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

param = torch.load(trained_model)
# create new OrderedDict that does not contain `module.`
from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in param.items():
    name = k[7:] # remove `module.`
    new_state_dict[name] = v


def demo():
    model.eval()
    with torch.no_grad():
        image = Image.open(input_imagefile)
        image = transform(image)
        image = image.unsqueeze(0)
        feature = model(image)
        predicted = feature.max(1, keepdim=True)[1]

    ##
    return predicted


#hoge = train(test_loader)
#print(hoge)

print(classes)
print(demo())
