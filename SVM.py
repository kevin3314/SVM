import numpy as np
import cvxopt
import func 
from cvxopt import matrix, solvers


class Svm:

    def __init__(self, x_list, y_list, kernel=0): 

        """
        x_list : データのリスト
        y_list : 分類の値のリスト
        kernel : カーネルとして何を使うかを指定.
                 0  -> 内積
                 1  -> 多項式
                 2  -> ガウス
                 3  -> シグモイド
        """

        self.x_list = x_list
        self.y_list = y_list
        self.kernel = func.determine_kernel(kernel)
        self.kernel_number = kernel
        self.N = len(x_list)

        #定数部分を定義する。
        self.q = matrix(-1.0 * np.ones(self.N))
        self.G = matrix(np.diag([-1.0]*self.N))
        self.h = matrix(np.zeros(self.N))
        self.b = matrix(0.0)

        #P,Aの設定
        self.P = np.zeros((self.N, self.N))
        for i in range(self.N):
            for j in range(self.N):
                self.P[i, j] = self.kernel( self.x_list[i], self.x_list[j]) * self.y_list[i] * self.y_list[j]
        self.P = matrix(self.P) 
        self.A = np.zeros((1, self.N))
        for i in range(self.N):
            self.A[0, i] = self.y_list[i]
        self.A = matrix(self.A)

    def solve(self):
        sol = solvers.qp(P=self.P, q=self.q, G=self.G, h=self.h, A=self.A, b=self.b)
        
        #alphaのリストを作る
        alpha_list = []
        for i in range(self.N):
            alpha_list.append(sol['x'][i,0]) 
        print("アルファの値を表示します。")
        
        for i , alpha in enumerate(alpha_list):
            print("alpha[",i,"]", alpha)
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print("\n")

        w = np.zeros((1,2))
        for i in range(self.N):
                w = w + alpha_list[i] * self.y_list[i] * self.x_list[i]
        print("重みの値を表示します。")
        print(w)

        shita = np.dot(w, self.x_list[0]) - self.y_list[0]
        print("閾値を表示します。")
        print(shita)

def get_datalist(file_name):
    """
    filename : データの入っているファイルの名前
    x_list : データのリスト
    y_list : 分類データのリスト
    """

    x_list = []
    y_list = []

    with open(file_name) as f:
        for s_line in f:
            s_line = s_line.split(",")
            x = int(s_line[0])
            y = int(s_line[1])
            z = int(s_line[2])

            x_list.append(np.array([x, y])) 
            y_list.append(z)

    return (x_list, y_list)

if __name__ == '__main__':
    file_name = "kernel_test_data.txt"
    (xlist, ylist) = get_datalist(file_name)
    svm = Svm(xlist, ylist, 2 )
    svm.solve()
