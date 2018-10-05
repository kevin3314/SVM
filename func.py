import numpy as np
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

#xに応じて色々なカーネルを返す関数。
def determine_kernel(x):
    if x == 0: return np.dot
    elif x == 1: return poly
    elif x == 2: return gauss
    elif x == 3: return sigmoid

#多項式カーネル
def poly(x1, x2):
    naiseki = np.dot(x1, x2)
    return pow( 1+naiseki, 2)

#ガウスカーネル
def gauss(x1, x2):
    V = 100
    t = np.dot( x1-x2, x1-x2)
    return np.exp( -1 * t / (2 * V))

#シグモイドカーネル
def sigmoid(x1, x2):
    return np.tanh( np.dot(x1, x2) + 2)

#ファイル名を受け取って、データのリストと各データの次元を返す関数
def get_datalist(file_name):
    """
    filename : データの入っているファイルの名前
    x_list : データのリスト
    y_list : 分類データのリスト
    """
    x_list = []
    y_list = []
    try:
        with open(file_name) as f:
            """
            n:データの次元数
            a:データを一時的に保存するベクトル
            """
            n = 0
            for s_line in f:
                s_line = s_line.split(",")
                n = len(s_line) -1
                a = np.zeros(n)
                for i, x in enumerate(s_line):
                    if i == n:
                        y_list.append(int(x))
                    else: 
                        a[i] = int(x) 
                x_list.append(a) 

    except FileNotFoundError:
        print("エラー:ファイル名が間違っています。正しいファイル名を入力してください")
        sys.exit()
    return (x_list, y_list, n)

def graph_dot(x_list, y_list, weight, shita):
    """
    Xr,Yr,Xb,Yb: ラベル付けに応じて分類したデータ群
    """
    Xr = []
    Yr = []
    Xb = []
    Yb = []

    for i, x in enumerate(x_list):
        if y_list[i] == 1.0:
            Xr.append(x[0])
            Yr.append(x[1])
        else:
            Xb.append(x[0])
            Yb.append(x[1])
    plt.scatter(Xr, Yr, c='red')
    plt.scatter(Xb, Yb, c='blue')

    X = np.linspace(np.max(x_list), np.min(x_list), 1000)
    y = (-1 * X * weight[0][0] / weight[0][1]) + shita / weight[0][1]

    plt.plot(X,y)

    plt.savefig('dot_figure.png')


def div_list(larger, mini, number):
    each_sub = (larger - mini) / number
    l = []
    for i in range(number):
        l.append(mini + each_sub * i)

    return l

def graph_ker(x_list, y_list, alpha_list, shita, kernel):
    Xr = []
    Yr = []
    Xb = []
    Yb = []

    size = 10
    large = np.max(x_list)
    mini = np.min(x_list)
    grid_x = div_list(large, mini, size)
    grid_y = div_list(large, mini, size) 

    for i in range(size):
        for j in range(size):
            x = np.array([grid_x[i], grid_y[j]])
            result = 0.0
            for m in range(len(x_list)):
                result += alpha_list[m] * y_list[m] * kernel(x, x_list[m])
            result -= shita
            if result > 0.0:
                Xr.append(x[0])
                Yr.append(x[1])
            else:
                Xb.append(x[0])
                Yb.append(x[1]) 

    plt.scatter(Xr, Yr, c='red')
    plt.scatter(Xb, Yb, c='blue')

    plt.savefig('Ker_pic.png')
