import numpy as np
import sys

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
    return np.tanh( np.dot(x1, x2) + 1)

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
