import SVM as svm
import numpy as np
import func

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
        #x_list,y_listをそれぞれ等分割
        cross_n = int(self.cross_n)
        div_x_lsit = func.div_list(self.x_list,cross_n)
        div_y_lsit = func.div_list(self.y_list,cross_n)
        test_x_lsit = []
        test_y_list = []
        cor_per_list = []

        for i in range(cross_n):
            for m in range(cross_n):
                if m != i:  
                    test_x_lsit.extend(div_x_lsit[m])
                    test_y_list.extend(div_y_lsit[m])
                else: pass
            inst = svm.Svm(test_x_lsit, test_y_list, self.data_dim, self.kernel_number, self.write_name, p_dict)
            inst.solve()
            cor_per = inst.eval(div_x_lsit[i], div_y_lsit[i])
            cor_per_list.append(cor_per)
        average = 0.0
        for x in cor_per_list:
            average = average + x
        return (average / len(cor_per_list))
        
    def sol_pera(self):
        if self.kernel_number == 0:
            pass
            #解く必要がない
        elif self.kernel_number == 2:
            #ガウスカーネルの時
            score_list = []
            for i in range(-10, 10):
                p_dict = {"p1": pow(2, i)}
                score = self.validate(p_dict)
                score_list.append(score)
            print(score_list)
            print("最良スコア: 2^"+ str(-10 + score_list.index(max(score_list))))

        else:
            #多項式、シグモイドカーネルの時
            score_dict = {}
            for i in range(1, 10, 2):
                for j in range(1, 10, 2):
                    p_dict = {"p1": pow(2, i), "p2": pow(2,i)}
                    score = self.validate(p_dict)
                    x1 = str(i)
                    x2 = str(j)
                    score_dict[x1+x2] = score
            print("index:->" + max(score_dict, key=score_dict.get) + "最良スコア->" + str(max(score_dict.values()) ))
