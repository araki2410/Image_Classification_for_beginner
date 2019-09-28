import matplotlib # グラフ描画
matplotlib.use('Agg') # バックグラウンド実行, リモートでディスプレイを持ってこれない場合に便利
import torch ## DeepLearning 用フレームワーク
import torch.nn as nn # torch.nn を何回も呼ぶので nn に省略
import numpy # 行列演算
from torchvision import transforms # 画像の前処理用関数
import os # OS インターフェース. ファイルの読み書きに用いる. 今回はディレクトリ参照に利用.

import mycnn_torch # 自分で作成したネットワーク. 同じディレクトリにmycnn_torch.pyを用意
import mydataloader_torch # 自作したデータローダ.

epoch_size = 6
bs = 2   # Batch size
learning_rate = 0.0001 # Learning rate
dataset_dir = "../Dataset"
classes = [f for f in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, f))]
num_of_class = len(classes)
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

print(' epoch =', epoch_size, ', batch_size =', bs, ', learning_rate =', learning_rate,
      ', \n Data dir = "', dataset_dir, '", Num of Class =', num_of_class,
      ', \n Classes =', classes, '\n')
#print(model)

## セットアップされたNVIDIA GPUがある場合GPUを認識してから, モデルをGPUで処理できるようにしておく.
#device = 'cuda' if t.cuda.is_available() else 'cpu'
#model = model.to(device)


## Dateload 自作したデータローダで、自作したデータセット (画像のパス) を読み込む
dataset = mydataloader_torch.CrossSquare_9x9(dataset_dir, transform)
dataset_size = len(dataset) # 読み込んだデータの数

## 読み込んだデータセットを学習用とテスト用に分割。もしくはテスト用データを用意して読み込み
train_size = int(0.8 * dataset_size)  # 0 to 1 : 0.8でデータセットの8割を学習に用いる
test_size = dataset_size - train_size # 学習に使わない2割。(単純に*0.2すると小数がでる)
## torch.utils.data.random_split() を用いて読み込んだデータセットをランダムに分割
## 再現性がないことに注意
train_data, test_data = torch.utils.data.random_split(dataset, [train_size, test_size])
## パスから画像データを読み込む準備 torch.utils.data.DataLoader(input_paths, batch_size=1, shuffle=True/False, num_workers=0)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=bs)
test_loader  = torch.utils.data.DataLoader(test_data,  batch_size=bs)

## Loss & Oprimizer
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

def train(data_loader):
    ## 学習ループ
    model.train() # パラメータ更新モード
    running_loss = 0 # ?
    for idx, (images, labels) in enumerate(data_loader):
        ## idx = ループ回数, images=画像行列, labels=正解値
        ## GPU などを認識させてモデルを読み込んでいる場合は、画像と正解値もGPUに移す
        # images = images.to(device)
        # labels = labels.to(device)
        
        ## パラメータを初期化 ? zero the paramater gradient 
        optimizer.zero_grad()
        
        ## forward + backword + optimize
        features = model(images) # input the images to CNN. CNNに画像を入力して特徴量を出力
        loss = loss_function(features, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    ####
    train_loss = running_loss / len(train_loader)
    ## 学習一周分のlossを返す
    return(train_loss)


def test(data_loader):
    ## 評価ループ
    model.eval() # パラメータ固定モード
    running_loss = 0
    correct_answer = 0 # init
    num_of_answer = 0  # init 

    with torch.no_grad():
        for idx, (images, labels) in enumerate(data_loader):
            # images = images.to(device)
            # labels = labels.to(device)
            
            features = model(images)
            
            loss = loss_function(features, labels)
            running_loss += loss.item()
            
            predict_answer = features.max(1, keepdim=True)[1]
            correct_answer += predict_answer.eq(labels.view_as(predict_answer)).sum().item()
            num_of_answer  += labels.size(0)
    
    ####
    val_loss = running_loss / len(data_loader)
    val_accuracy = correct_answer / num_of_answer
    
    return val_loss, val_accuracy

    

def demo(test_loader):
    model.eval()
    predicts = {}
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            #images = images.to(device)
            #labels = labels.to(device)
            
            outputs = model(images)
            predicted = outputs.max(1, keepdim=True)[1]

            for i in range(labels.shape[0]):
                label = labels[i].data.cpu().item()
                try:    
                    predicts[label].append(predicted[i].data.cpu().item())
                except:
                    predicts[label] = []
                    predicts[label].append(predicted[i].data.cpu().item())
            #print(predicts[label])

    return predicts


#hoge = train(test_loader)
#print(hoge)


for epoch in range(epoch_size):
    loss = train(train_loader)
    val_loss, val_acc = test(test_loader)

    print('epoch %2d, loss: %.4f val_loss: %.4f val_acc: %.4f' % (epoch, loss, val_loss, val_acc))
    

print('Finished Training')
