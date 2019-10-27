import cv2          # Open and create image. 画像を開いたり作ったりする
import numpy as np  # Calculate the matrix. 行列を計算する (画像は行列と考えることができる)

input_name = "sample.png"
output_name = "conved.png"

gray_image = cv2.imread(input_name, cv2.IMREAD_GRAYSCALE) # sample.png をグレースケールで開く
g = gray_image               # 省略表記を用意
img_h = gray_image.shape[0]  # height, 入力画像の高さ
img_w = gray_image.shape[1]  # width,  入力画像の横幅

## [[255   0 255],     [[ 1 -1  1],
##  [  0 255   0],  ->  [-1  1 -1],
##  [255   0 255]]      [ 1 -1  1]]
g = g-125.25 # (0 ~ 255) -> (-125.25 ~ 125.25)
g = (np.exp(g) - np.exp(-g)) / (np.exp(g) + np.exp(-g)) # 活性化関数tanh. (-125.25 ~ 125.25) -> (-1 ~ 1)

g = [[-1,-1,-1,-1,-1,-1,-1],
     [-1, 1,-1,-1,-1, 1,-1],
     [-1,-1, 1,-1, 1,-1,-1],
     [-1,-1,-1, 1,-1,-1,-1],
     [-1,-1, 1,-1, 1,-1,-1],
     [-1, 1,-1,-1,-1, 1,-1],
     [-1,-1,-1,-1,-1,-1,-1]
     ]
img_h = 7
img_w = 7

## 平滑化フィルタ
#filter = [[1,1,1],
#          [1,1,1],
#          [1,1,1]]

## \ 斜めエッジに反応するフィルタ
filter = [[ 1,-1,-1],
          [-1, 1,-1],
          [-1,-1, 1]]

f = np.array(filter) # 作成したフィルタをndarray型に変換 (ここではfor文を使うのでndarrayとしてはあまり活用しない)
fil_h = f.shape[0]   # filter_height, フィルタの高さ
fil_w = f.shape[1]   # filter_width,  フィルタの幅

conved_image = []    # 畳み込まれた画像をしまう場所を用意
for i in range(img_h-fil_h+1):
    ## 2重for文で実際に畳み込む
    ## 
    l = []
    for j in range(img_w-fil_w+1):
       c = g[i][j]   * f[0][0] \
          +g[i][j+1] * f[0][1] \
          +g[i][j+2] * f[0][2] \
          +g[i+1][j]   * f[1][0] \
          +g[i+1][j+1] * f[1][1] \
          +g[i+1][j+2] * f[1][2] \
          +g[i+2][j]   * f[2][0] \
          +g[i+2][j+1] * f[2][1] \
          +g[i+2][j+2] * f[2][2]
       
       c = c/f.size  # 全体ピクセル数の数で割って平均をとる (-1~1)
       #l.append(((c+1)/2)*255) # (-1~1) -> (0~255)
       l.append(c*255) # (-1~1) -> (-255~255)
    conved_image.append(l)

conved_matrix = np.array(conved_image)
print(gray_image.shape, " -> ", conved_matrix.shape)
print(input_name, " -> ", output_name)
cv2.imwrite(output_name, conved_matrix)
                
       
        
