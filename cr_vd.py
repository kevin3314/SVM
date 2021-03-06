import SVM as svm
import numpy as np
import func
import sys

class val_class():
    def __init__(self, x_list, y_list, data_dim, kernel_number, write_name,cross_n):
        #初期化
        self.x_list = x_list
        self.y_list = y_list
        self.data_dim = data_dim
        self.kernel_number = kernel_number
        self.write_name = write_name
        self.cross_n = cross_n

    def validate(self, p_dict):
        """
        交差検定を行う関数。p_dictでパラメタを指定する。
        cross_n: データを分割する個数
        div_x_list: データのリストを等分割した結果得たリストを保持するリスト
        div_y_list: 同上
        learn_x_lsit: 学習を行うために用いるリスト
        learn_y_list: 同上
        cor_per: 正解率
        cor_per_list: 正解率を保持するリスト
        average: 識別率の平均を計算するための一時変数
        """
        #x_list,y_listをそれぞれ等分割
        try:
            cross_n = int(self.cross_n)
        except ValueError:
            print("分割する値は自然数である必要があります")
            sys.exit()

        div_x_list = func.div_list(self.x_list,cross_n)
        div_y_list = func.div_list(self.y_list,cross_n)
        #学習に用いるリスト、正解率のリストを作る
        learn_x_list= []
        learn_y_list= []
        cor_per_list = []

        for i in range(cross_n):
            #i番目のリストをテスト用とし他のデータで学習する。
            for m in range(cross_n):
            #i番目以外のリストを一つのリストにまとめる
                if m != i:  
                    learn_x_list.extend(div_x_list[m])
                    learn_y_list.extend(div_y_list[m])
                else: pass
            #学習データを用いて学習を行う
            inst = svm.Svm(learn_x_list,learn_y_list, self.data_dim, self.kernel_number, self.write_name, p_dict)
            inst.solve()
            #正解率を計算しリストに加える
            cor_per = inst.eval(div_x_list[i], div_y_list[i])
            cor_per_list.append(cor_per)
            #学習リストを初期化する
            learn_x_list = []
            learn_y_list = []
        #平均的な正解率を計算する
        average = 0.0
        for x in cor_per_list:
            average += x
        return (average / len(cor_per_list))
        
    def sol_pera(self):
        """
        パラメタを探索する関数。最後に一番よかったパラメタを用いて学習しその結果を出力する。
        """
        if self.kernel_number == 0:
            #内積の時
            p_dict = {} 
            score = self.validate(p_dict)
            print("スコア->" + str(score))
            inst = svm.Svm(self.x_list, self.y_list, self.data_dim, self.kernel_number, self.write_name, p_dict)
            inst.solve()
            inst.plot()

        elif self.kernel_number == 2 or self.kernel_number == 1:
            #ガウスカーネルの時
            #パラメタとスコアの列を保持しておく
            pera_list = []
            score_list = []
            for i in [ x for x in range(1, 10)]:
                p_dict = {"p1": i}
                score = self.validate(p_dict)
                score_list.append(score)
                pera_list.append(p_dict["p1"])
            #最良スコアのインデックスをxに保存
            x = score_list.index(max(score_list))
            #最良パラメタ、スコアを表示
            print("最良パラメタ:  "+ str(pera_list[x]) + "最良スコア->" + str(max(score_list)))
            #最良スコアでSVMを構成し識別器をグラフにする
            p_dict = {"p1": pera_list[x]}
            print(pera_list[x])
            inst = svm.Svm(self.x_list, self.y_list, self.data_dim, self.kernel_number, self.write_name, p_dict)
            inst.solve()
            inst.plot()
            #グラフ生成
            pera_list = np.array(pera_list)
            score_list = np.array(score_list)
            func.graph_pera(score_list, pera_list, self.write_name)

        else:
            #シグモイドカーネルの時 
            #スコアとその時のパラメタを保持するハッシュ
            score_dict = {}
            #それぞれのパラメタについて順に検定を行う
            for i in [ 0.1*x for x in range(1,10,2)]:
                for j in [0.1*x for x in range(1,10,2)]:
                    p_dict = {"p1": i, "p2": j}
                    score = self.validate(p_dict)
                    x1 = str(i)
                    x2 = str(j)
                    score_dict[x1+","+x2] = score
            #最良スコアの値とパラメタを表示する
            print("最良パラメタ:->" + max(score_dict, key=score_dict.get) + "最良スコア->" + str(max(score_dict.values()) ))
            tmp_l =(max(score_dict, key=score_dict.get)).split(",")
            #最良スコアの時のパラメタで識別器を構成する
            x1 = float(tmp_l[0])
            x2 = float(tmp_l[1])
            p_dict = {"p1": x1, "p2": x2}
            inst = svm.Svm(self.x_list, self.y_list, self.data_dim, self.kernel_number, self.write_name, p_dict)
            inst.solve()
            inst.plot() 

            pera1_list = []
            pera2_list = []
            score_list = []
            #パラメタについてのグラフを作る
            for item in score_dict.items():
                pera_list = item[0].split(",")
                pera1 = float(pera_list[0])
                pera2 = float(pera_list[1])
                pera1_list.append(pera1)
                pera2_list.append(pera2)
                score_list.append(item[1])
            func.graph_two_pera(score_list, pera1_list, pera2_list, self.write_name)
