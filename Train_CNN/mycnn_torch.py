
import torch # 基本モジュール
import torch.nn as nn # ネットワーク構築用. torch.nn は何回も呼ぶため簡略化

class MyClassifyNet(nn.Module):
    ## 自分で作った画像データのサイズは 9x9 , RGBとして扱うので (3x9x9)
    ## input size (3, 9, 9)
    def __init__(self):
        ## ネットワークを決める
        super(MyClassifyNet, self).__init__() # initialize
        ## nn.conv2d(input_chanel, output_chanel, kernel_size, stride=1, padding=0)
        ## nn.MaxPool2d((kernel_size), stride=2, padding=0, dilation=1, ceil_mode=False)
        ## nn.Linear(input_feature_size, output_feature_size, bias=True)
        ## nn.Sonfmax(dim=None)
        self.conv_1 = nn.Sequential(
            nn.Conv2d(1, 16, 3), # Chanel is 1 because images are gray_scale.
            nn.ReLU()
            )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(16, 32, 3),
            nn.ReLU()
            )
        self.fc_1 = nn.Sequential(
            nn.Linear(32*5*5, 100),
            nn.ReLU()
            )
        self.fc_2 = nn.Sequential(
            nn.Linear(100, 2),
            nn.ReLU()
            )

    def forward(self, x):
        ## __init__ で決めたネットワークにinputを投入していく。
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = x.view(x.size(0), 32*5*5) ## .view でreshape. .view(bs, num_of_feature)
        x = self.fc_1(x)
        x = self.fc_2(x)
        return x
