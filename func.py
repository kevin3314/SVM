import numpy as np
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import error

def set_def_dic(x):
    if x == 0: return {}
    elif x == 1: return {"p1": 1, "p2": 2}
    elif x == 2: return {"p1": 10}
    elif x == 3: return {"p1": 1, "p2": 4} 

#xに応じて色々なカーネルを返す関数。
def determine_kernel(x, p_dict):
    if x == 0: return dot_kernel
    elif x == 1: return poly_kernel(p_dict)
    elif x == 2: return gauss_kernel(p_dict)
    elif x == 3: return sigmoid_kernel(p_dict)
    else:        raise error.DetermineError

class kernel:
    def __init__(self):
        pass

    def return_fun(self):
        pass

class dot_kernel(kernel):
    def __init__(self):
        pass

    def set_pera(p_dict):
        pass
    
    def return_fun(self):
        return np.dot

#多項式カーネル
class poly_kernel(kernel):
    def __init__(self, p_dict):
        self.p1 = p_dict["p1"]
        self.p2 = p_dict["p2"]

    def set_pera(p_dict):
        self.p1 = p_dict["p1"]
        self.p2 = p_dict["p2"]

    def poly(self, x1, x2):
        naiseki = np.dot(x1, x2)
        return pow( self.p1+naiseki, self.p2)

    def return_fun(self):
        def poly(x1, x2):
            naiseki = np.dot(x1, x2)
            return pow( self.p1+naiseki, self.p2)
        return poly

#ガウスカーネル
class gauss_kernel(kernel):
    def __init__(self, p_dict):
        self.p1 = p_dict["p1"]

    def set_pera(p_dict):
        self.p1 = p_dict["p1"]

    def return_fun(self):
        V = pow(self.p1, 2) 
        def gauss(x1, x2):
            t = np.dot( x1-x2, x1-x2)
            return np.exp( -1 * t / (2 * V)) 
        return gauss

#シグモイドカーネル
class sigmoid_kernel(kernel):
    def __init__(self, p_dict):
        self.p1 = p_dict["p1"]
        self.p2 = p_dict["p2"]

    def set_pera(p_dict):
        self.p1 = p_dict["p1"]
        self.p2 = p_dict["p2"]

    def sigmoid(self, x1, x2):
        naiseki = np.dot(x1, x2)
        return np.tanh( self.p1 * np.dot(x1, x2) + self.p2)

    def return_fun(self):
        def sigmoid(x1, x2):
            naiseki = np.dot(x1, x2)
            return np.tanh( self.p1 * np.dot(x1, x2) + self.p2)
        return sigmoid

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

def graph_dot(x_list, y_list, weight, shita, write_name):
    """
    Xr,Yr,Xb,Yb: ラベル付けに応じて分類したデータ群
    """
    Xr = []
    Yr = []
    Xb = []
    Yb = []

    #各データについてラベルを確認し分類する。
    for i, x in enumerate(x_list):
        if y_list[i] == 1.0:
            Xr.append(x[0])
            Yr.append(x[1])
        else:
            Xb.append(x[0])
            Yb.append(x[1])
    plt.scatter(Xr, Yr, c='red')
    plt.scatter(Xb, Yb, c='blue')

    #識別器(直線)を表示する。
    X = np.linspace(np.max(x_list), np.min(x_list), 1000)
    y = (-1 * X * weight[0][0] / weight[0][1]) + shita / weight[0][1]
    plt.plot(X,y)

    plt.savefig(write_name + '.png')

def div_list(larger, mini, number):
    #最大値最小値の間を等分割する。each_sub:幅
    each_sub = (larger - mini) / number
    #l:等分割した各点の値
    l = []
    for i in range(number):
        l.append(mini + each_sub * i)
    return l

def graph_ker(x_list, y_list, alpha_list, shita, kernel, write_name):
    Xr = []
    Yr = []
    Xb = []
    Yb = []

    div_number = 100
    large = np.max(x_list)
    mini = np.min(x_list)
    grid_x = div_list(large, mini, div_number)
    grid_y = div_list(large, mini, div_number) 

    for i in range(div_number):
        for j in range(div_number):
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

    plt.scatter(Xr, Yr, 10, c='g')
    plt.scatter(Xb, Yb, 10, c='y')

    Xr = []
    Yr = []
    Xb = []
    Yb = []

    #各データについてラベルを確認し分類する。
    for i, x in enumerate(x_list):
        if y_list[i] == 1.0:
            Xr.append(x[0])
            Yr.append(x[1])
        else:
            Xb.append(x[0])
            Yb.append(x[1])
    plt.scatter(Xr, Yr, c='red')
    plt.scatter(Xb, Yb, c='blue')

    plt.savefig(write_name + '.png')
