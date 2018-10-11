import numpy as np
import cvxopt
import func 
from cvxopt import matrix, solvers 
import sys
import error

class Svm:
    def __init__(self, x_list, y_list, data_dim, kernel, write_name, p_dict={}): 
        """
        x_list : データのリスト
        y_list : 分類の値のリスト
        kernel_number : カーネルとして何を使うかを指定.
                 0  -> 内積
                 1  -> 多項式
                 2  -> ガウス
                 3  -> シグモイド
        kernel_class: カーネルのインスタンスを持つ。パラメタを変えられる
        kernel: カーネル関数
        N : データの数
        data_dim : それぞれのデータの次元
        """

        self.x_list = x_list
        self.y_list = y_list
        try:
            self.kernel_class = func.determine_kernel(kernel, p_dict)
        except error.DetermineError:
            print("""エラー：第二引数の値が誤っています.
            0:カーネルなし
            1:多項式カーネル
            2:ガウスカーネル
            3:シグモイドカーネル""")
            sys.exit()            

        self.kernel = self.kernel_class.return_fun()
        self.kernel_number = kernel
        self.N = len(x_list)
        self.data_dim = data_dim
        self.write_name = write_name

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

    def set_pera(self, p_dict):
        #パラメタのハッシュを受け取ってパラメタを設定し直す。
        self.kernel_class = self.kernel_class.set_pera(p_dict)
        self.kernel = self.kernel_class.return_fun()
        self.P = np.zeros((self.N, self.N))
        for i in range(self.N):
            for j in range(self.N):
                self.P[i, j] = self.kernel( self.x_list[i], self.x_list[j]) * self.y_list[i] * self.y_list[j]
        self.P = matrix(self.P) 
        

    def solve(self):
        sol = solvers.qp(P=self.P, q=self.q, G=self.G, h=self.h, A=self.A, b=self.b)
        
        #alphaのリストを作る
        alpha_list = []
        #サポートベクタの番号を覚えておく
        sup_number = 0

        for i in range(self.N):
            alpha_list.append(sol['x'][i,0]) 
            if sol['x'][i,0] > 0.1:
                sup_number = i
        self.alpha_list = alpha_list

        #重みを計算する
        w = np.zeros((1,self.data_dim))
        for i in range(self.N):
            w = w + alpha_list[i] * self.y_list[i] * self.x_list[i]
        self.w = w[0]

        #閾値を計算する
        self.shita = self.kernel(self.w, self.x_list[sup_number]) - self.y_list[sup_number]

    def eval(self, x_list, y_list):
        all_num = 0
        cor_num = 0
        for i in range(len(x_list)):
            x = x_list[i]
            result = 0.0
            for m in range(len(self.x_list)):
                result += self.alpha_list[m] * self.y_list[m] * self.kernel(x, self.x_list[m])
            result -= self.shita
            if result > 0.0:
                if y_list[i] == 1: cor_num += 1
                else: pass
            else:
                if y_list[i] == -1: cor_num += 1
                else: pass
            all_num += 1
        print(str(cor_num) + "/" + str(all_num))
        print(float(cor_num) / float(all_num))

        return (float(cor_num) / float(all_num))

    def plot(self):
        #データの次元が2ならば2次元平面上に表示する.
        if self.data_dim == 2:
            if self.kernel_number == 0:
                func.graph_dot(self.x_list, self.y_list, w, shita, self.write_name)  
            else:
                func.graph_ker(self.x_list, self.y_list, self.alpha_list, self.shita, self.kernel, self.write_name)
