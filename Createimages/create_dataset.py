import cv2          # Open and create image. 画像を開いたり作ったりする
import random       # Make randomly. ランダム要素を生み出す
import string       # Control strings. 文字列を操作する
import numpy as np  # Calculate the matrix. 行列を計算する (画像は行列と考えることができる)
import subprocess   # Use shell command in this time. サブプロセス。shellコマンドを入力するのに使う
import sympy

#=========
width  = 9
height = 9
rand_rate = 0.1
save_dir = ["../Dataset/Cross/", "../Dataset/Square/"]
num_of_img = 30
name_size = 8

def rand_unit(rr=0.1):
    ## Return int -1 to 1 randomly.
    x = random.random()
    x = 1 if x > (1-rr) else  0 if x > rr else -1
    y = random.random()
    y = 1 if y > (1-rr) else  0 if y > rr else -1
    return x, y


def randXmatrix(w, h, rr=0.5):
    rama = np.ones((w, h)) # rand_matrix 行列を初期化
    x = sympy.Symbol('x')
    y = sympy.Symbol('y')
    expr_p = y - x # 変数expr_p に式を代入
    expr_m = y + x
    for i in range(w-2):
        y = sympy.solve(expr_p.subs(x, i)) # x に i を代入して y を導出
        rx, ry = rand_unit()
        rama[y[0]+1+ry][i+1+rx] = 0  # \ y = x 第1象限が右下に位置する点に注意
    for i in range(w-2):
        y = sympy.solve(expr_m.subs(x, i))
        rx, ry = rand_unit()
        rama[-y[0]+1][(h-1)-i-1] = 0 # / y = -x
    return(rama)

def rand0matrix(w, h, rr=0.5):
    rama = np.ones((w, h))
#    x = sympy.Symbol('x')
    y = sympy.Symbol('y')
    expr_t = y - 1
    expr_b = y + 2
    for i in range(w-2): # Top side, 上辺
        y = sympy.solve(expr_t.subs(1, 1))
        rx, ry = rand_unit()
        rama[y[0]+ry][i+1+rx] = 0
    for i in range(w-2): # Bottom side, 底辺
        y = sympy.solve(expr_b.subs(1, 1))
        rx, ry = rand_unit()
        rama[y[0]+ry][i+1+rx] = 0
    for i in range(w-2): # Left side, 左辺
        rx, ry = rand_unit()
        rama[i+1+ry][1+rx] = 0
    for i in range(w-2): # Right side, 右辺
        rx, ry = rand_unit()
        rama[i+1+ry][-2+rx] = 0
    return(rama)

# def coloring(binary_matrix):
#    ##  binary     =>    RGB (BGR)
#    ## [[1,1,1],       [[255,255,255],[255,255,255],[255,255,255],
#    ##  [1,0,1],  =>    [255,255,255],[  0,  0,  0],[255,255,255],
#    ##  [1,1,1]]        [255,255,255],[255,255,255],[255,255,255]]
#    rgb_matrix = []  
#    filter = 255
#    for l in binary_matrix:
#        l = l * filter # [1,0,1] => [255,  0,255]
#        l = [list(x) for x in list(zip(*[list(map(int,l)),list(map(int, l)),list(map(int, l))]))] # [255,  0,255] => [[255,255,255], [  0,  0,  0], [255,255,255]]
#        rgb_matrix.append(l)
#    return(np.array(rgb_matrix))

def graying(binary_matrix):
    ##  binary     =>    Grayscale
    ## [[1,1,1],       [[255,255,255],
    ##  [1,0,1],  =>    [255,  0,255],
    ##  [1,1,1]]        [255,255,255]]
    filter = 255
    gray_img = binary_matrix * filter
    return(gray_img)


def filename_generate(length, chars=None):
    ## Naming a image randomly.
    ## chars = ["a","2","4","1","0"]
    if chars is None:
        chars = string.digits + string.ascii_letters ## [a-zA-Z0-9]
    return ''.join([random.choice(chars) for i in range(length)])




for i in range(num_of_img):
    binary_xxx = randXmatrix(width, height, rand_rate) #  Matrix into xxx
    img = graying(binary_xxx)                         #  Convert the binary_matirix xxx with grayscaled_matrix for OpenCV2
    file_name = (save_dir[0]+filename_generate(name_size)+".png") # Create the image name and path
    cv2.imwrite(file_name, img)                 #  Write a file

    binary_ooo = rand0matrix(width, height, rand_rate) #  ooo
    img = graying(binary_ooo)                         #  Convert
    file_name = (save_dir[1]+filename_generate(name_size)+".png") # Create name
    cv2.imwrite(file_name, img)                 #  Write

