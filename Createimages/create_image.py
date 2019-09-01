import cv2          # OpenCV2. Open and create image. 画像を開いたり作ったりする
import numpy as np  # Calculate the matrix. 行列を計算する (画像は行列と考えることができる) # OpenCV2での作業に使う

#=========
# width  = 9, height = 9

binary_image = [[ 1,1,1,1,1,1,1,1,1],
                [ 1,0,1,1,1,1,1,0,1],
                [ 1,1,0,1,1,1,0,1,1],
                [ 1,1,1,0,1,0,1,1,1],
                [ 1,1,1,1,0,1,1,1,1],
                [ 1,1,1,0,1,0,1,1,1],
                [ 1,1,0,1,1,1,0,1,1],
                [ 1,0,1,1,1,1,1,0,1],
                [ 1,1,1,1,1,1,1,1,1]]

def graying(binary_img):
    ## numpy.ndarray -> numpy.ndarray
    ## [  1   0   1] -> [255   0 255] 
    filter = 255
    gray_img = binary_img * filter
    return(gray_img)

image_matrix = np.array(binary_image)   # 行列計算を楽にするため ndarray型に変換
gray_matrix = graying(image_matrix)     # ndarray型に変換された行列を、OpenCV2 でのグレースケール表現に変換 (0~1 -> 0~255)
cv2.imwrite("sample.png", gray_matrix)  # OpenCV2 のimwrite関数を使い、変換した行列を sample.png として保存
cv2.imwrite("../Convolution/sample.png", gray_matrix)
