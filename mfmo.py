#-*- coding: utf-8 -*-
#-*- coding: cp950 -*-

import pandas as pd
import random
import numpy as np
from sklearn import preprocessing
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import math
from sklearn.metrics import f1_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score

def load_generative_csv():
    data_out_x = []
    data_out_y = []
    title = ['gender', 'age', 'marr', 'www.facebook.com', 'www.google.com.tw', 'www.youtube.com', 'Shopping', 'Travel', 'Restaurant and Dining', 'Entertainment', 'Games', 'Education']

    csv = pd.read_csv("user_feature.csv")
    for i in range(0, len(csv), 2):
        tmp_x = []
        tmp_y = []
        for cate in title:
            tmp_x.append(csv[cate][i])
            tmp_y.append(csv[cate][i+1])
        data_out_x.append(tmp_x)
        data_out_y.append(tmp_y)

    return data_out_x, data_out_y

def value2label(true, tr_pred, te_pred):
    one_count = 0
    for value in true:
        if value == 1:
            one_count += 1

    for i in range(one_count):
        if i == one_count - 1:
            line = tr_pred[tr_pred.index(max(tr_pred))]
            tr_pred[tr_pred.index(max(tr_pred))] = -100

        else:
            tr_pred[tr_pred.index(max(tr_pred))] = -100

    for value in range(len(tr_pred)):
        if tr_pred[value] == -100:
            tr_pred[value] = 1

        else:
            tr_pred[value] = 0

    for value in range(len(te_pred)):
        if te_pred[value] >= line:
            te_pred[value] = 1

        else:
            te_pred[value] = 0

    return tr_pred, te_pred

def feature_one_hard(data_y):

    feature_x = []

    for i in range(len(data_y)):
        feature = []

        #gender
        if data_y[i][0] == 0:
            feature.append(1)
            feature.append(0)
            feature.append(0)

        elif data_y[i][0] == 1:
            feature.append(0)
            feature.append(1)
            feature.append(0)

        else:
            feature.append(0)
            feature.append(0)
            feature.append(1)

        #age
        if data_y[i][1] == 0:
            feature.append(1)
            feature.append(0)
            feature.append(0)
            feature.append(0)

        elif data_y[i][1] == 1:
            feature.append(0)
            feature.append(1)
            feature.append(0)
            feature.append(0)

        elif data_y[i][1] == 2:
            feature.append(0)
            feature.append(0)
            feature.append(1)
            feature.append(0)

        else:
            feature.append(0)
            feature.append(0)
            feature.append(0)
            feature.append(1)

        #marr
        if data_y[i][2] == 0:
            feature.append(1)
            feature.append(0)
            feature.append(0)
            feature.append(0)

        elif data_y[i][2] == 1:
            feature.append(0)
            feature.append(1)
            feature.append(0)
            feature.append(0)

        elif data_y[i][2] == 2:
            feature.append(0)
            feature.append(0)
            feature.append(1)
            feature.append(0)

        else:
            feature.append(0)
            feature.append(0)
            feature.append(0)
            feature.append(1)

        feature.append(data_y[i][3])
        feature.append(data_y[i][4])
        feature.append(data_y[i][5])

        feature.append(data_y[i][6])
        feature.append(data_y[i][7])
        feature.append(data_y[i][8])
        feature.append(data_y[i][9])
        feature.append(data_y[i][10])
        feature.append(data_y[i][11])

        feature_x.append(feature)

    return feature_x


def mfmo(tr_f, tr_t, te_f, te_t):


    matrix = 1
    #latent_p, latent_q 產生初始值
    latent_p = []
    for i in range(len(tr_f[0])):
        temp = []
        for j in range(matrix):
            temp.append(random.uniform(0.0, 1.0))
        latent_p.append(temp)

    latent_q = []
    for i in range(matrix):
        temp = []
        for j in range(len(tr_t[0])):
            temp.append(random.uniform(0.0, 1.0))
        latent_q.append(temp)

    alpha = 0.01
    latent_p = np.asarray(latent_p)
    latent_q = np.asarray(latent_q)

    #MFMO
    for iters in range(30):
        print(latent_p[0][0])
        for u in range(len(tr_f)):
            for n in range(len(tr_t[0])):
                for k in range(len(latent_p)):
                    for l in range(len(latent_q)):

                        #算pre
                        w = []
                        pre_l = 0
                        for a in range(len(latent_p)):
                            temp = 0
                            for b in range(len(latent_q)):
                                temp = temp + latent_p[a][b] * latent_q[b][n]
                            w.append(temp)
                            #在這裡會有w[a][c]
                            pre_l += tr_f[u][a] * w[a]

                        latent_p[k][l] = -(alpha * 2.0 * (pre_l - tr_t[u][n]) * (tr_f[u][k] * latent_q[l][n]) - latent_p[k][l])

                        #算pre
                        w = []
                        pre_l = 0
                        for a in range(len(latent_p)):
                            temp = 0
                            for b in range(len(latent_q)):
                                temp = temp + latent_p[a][b] * latent_q[b][n]
                            w.append(temp)
                            #在這裡會有w[a][c]
                            pre_l += tr_f[u][a] * w[a]

                        latent_q[l][n] = -(alpha * 2.0 * (pre_l - tr_t[u][n]) * (tr_f[u][k] * latent_p[k][l]) - latent_q[l][n])

    #train 結果
    tr_w = np.matmul(latent_p, latent_q)

    tr_label = np.matmul(tr_f, tr_w)

    #test 運算
    te_label = np.matmul(te_f, tr_w)

    #把train結果分別拉出來
    tr_s_list = []
    tr_t_list = []
    tr_r_list = []
    tr_ent_list = []
    tr_g_list = []
    tr_e_list = []

    tr_s_tl = []
    tr_t_tl = []
    tr_r_tl = []
    tr_ent_tl = []
    tr_g_tl = []
    tr_e_tl = []

    for i in range(len(tr_label)):
        tr_s_list.append(tr_label[i][0])
        tr_t_list.append(tr_label[i][1])
        tr_r_list.append(tr_label[i][2])
        tr_ent_list.append(tr_label[i][3])
        tr_g_list.append(tr_label[i][4])
        tr_e_list.append(tr_label[i][5])

        tr_s_tl.append(tr_t[i][0])
        tr_t_tl.append(tr_t[i][1])
        tr_r_tl.append(tr_t[i][2])
        tr_ent_tl.append(tr_t[i][3])
        tr_g_tl.append(tr_t[i][4])
        tr_e_tl.append(tr_t[i][5])

    #把test結果分別拉出來
    te_s_list = []
    te_t_list = []
    te_r_list = []
    te_ent_list = []
    te_g_list = []
    te_e_list = []

    te_s_tl = []
    te_t_tl = []
    te_r_tl = []
    te_ent_tl = []
    te_g_tl = []
    te_e_tl = []

    for i in range(len(te_label)):
        te_s_list.append(te_label[i][0])
        te_t_list.append(te_label[i][1])
        te_r_list.append(te_label[i][2])
        te_ent_list.append(te_label[i][3])
        te_g_list.append(te_label[i][4])
        te_e_list.append(te_label[i][5])


        te_s_tl.append(te_t[i][0])
        te_t_tl.append(te_t[i][1])
        te_r_tl.append(te_t[i][2])
        te_ent_tl.append(te_t[i][3])
        te_g_tl.append(te_t[i][4])
        te_e_tl.append(te_t[i][5])
    '''
    #算train各種類auc
    tr_s_list = np.array(tr_s_list)
    tr_s_tl = np.array(tr_s_tl)
    tr_s_fpr, tr_s_tpr, _ = roc_curve(tr_s_tl, tr_s_list, pos_label=1)
    tr_s_auc = float("%.3f" % auc(tr_s_fpr, tr_s_tpr))

    tr_t_list = np.array(tr_t_list)
    tr_t_y = np.array(tr_t_tl)
    tr_t_fpr, tr_t_tpr, tr_t_thresholds = roc_curve(tr_t_y, tr_t_list, pos_label=1)
    tr_t_auc = float("%.3f" % auc(tr_t_fpr, tr_t_tpr))

    tr_r_list = np.array(tr_r_list)
    tr_r_y = np.array(tr_r_tl)
    tr_r_fpr, tr_r_tpr, tr_r_thresholds = roc_curve(tr_r_y, tr_r_list, pos_label=1)
    tr_r_auc = float("%.3f" % auc(tr_r_fpr, tr_r_tpr))

    tr_ent_list = np.array(tr_ent_list)
    tr_ent_y = np.array(tr_ent_tl)
    tr_ent_fpr, tr_ent_tpr, tr_ent_thresholds = roc_curve(tr_ent_y, tr_ent_list, pos_label=1)
    tr_ent_auc = float("%.3f" % auc(tr_ent_fpr, tr_ent_tpr))

    tr_g_list = np.array(tr_g_list)
    tr_g_y = np.array(tr_g_tl)
    tr_g_fpr, tr_g_tpr, tr_g_thresholds = roc_curve(tr_g_y, tr_g_list, pos_label=1)
    tr_g_auc = float("%.3f" % auc(tr_g_fpr, tr_g_tpr))

    tr_e_list = np.array(tr_e_list)
    tr_e_y = np.array(tr_e_tl)
    tr_e_fpr, tr_e_tpr, tr_e_thresholds = roc_curve(tr_e_y, tr_e_list, pos_label=1)
    tr_e_auc = float("%.3f" % auc(tr_e_fpr, tr_e_tpr))

    #算test各種類auc
    te_s_list = np.array(te_s_list)
    te_s_y = np.array(te_s_tl)
    te_s_fpr, te_s_tpr, te_s_thresholds = roc_curve(te_s_y, te_s_list, pos_label=1)
    te_s_auc = float("%.3f" % auc(te_s_fpr, te_s_tpr))

    te_t_list = np.array(te_t_list)
    te_t_y = np.array(te_t_tl)
    te_t_fpr, te_t_tpr, te_t_thresholds = roc_curve(te_t_y, te_t_list, pos_label=1)
    te_t_auc = float("%.3f" % auc(te_t_fpr, te_t_tpr))

    te_r_list = np.array(te_r_list)
    te_r_y = np.array(te_r_tl)
    te_r_fpr, te_r_tpr, te_r_thresholds = roc_curve(te_r_y, te_r_list, pos_label=1)
    te_r_auc = float("%.3f" % auc(te_r_fpr, te_r_tpr))

    te_ent_list = np.array(te_ent_list)
    te_ent_y = np.array(te_ent_tl)
    te_ent_fpr, te_ent_tpr, te_ent_thresholds = roc_curve(te_ent_y, te_ent_list, pos_label=1)
    te_ent_auc = float("%.3f" % auc(te_ent_fpr, te_ent_tpr))

    te_g_list = np.array(te_g_list)
    te_g_y = np.array(te_g_tl)
    te_g_fpr, te_g_tpr, te_g_thresholds = roc_curve(te_g_y, te_g_list, pos_label=1)
    te_g_auc = float("%.3f" % auc(te_g_fpr, te_g_tpr))

    te_e_list = np.array(te_e_list)
    te_e_y = np.array(te_e_tl)
    te_e_fpr, te_e_tpr, te_e_thresholds = roc_curve(te_e_y, te_e_list, pos_label=1)
    te_e_auc = float("%.3f" % auc(te_e_fpr, te_e_tpr))

    #印出所有train結果
    print("training_auc")
    print("s: " + str("%.3f" % tr_s_auc))
    print("t: " + str("%.3f" % tr_t_auc))
    print("r: " + str("%.3f" % tr_r_auc))
    print("ent: " + str("%.3f" % tr_ent_auc))
    print("g: " + str("%.3f" % tr_g_auc))
    print("e: " + str("%.3f" % tr_e_auc))

    #印出所有test結果
    print("test_auc")
    print("s: " + str("%.3f" % te_s_auc))
    print("t: " + str("%.3f" % te_t_auc))
    print("r: " + str("%.3f" % te_r_auc))
    print("ent: " + str("%.3f" % te_ent_auc))
    print("g: " + str("%.3f" % te_g_auc))
    print("e: " + str("%.3f" % te_e_auc))
    '''
    tr_s_list, te_s_list = value2label(tr_s_tl, tr_s_list, te_s_list)
    tr_s_score = f1_score(tr_s_tl, tr_s_list, average='macro')

    tr_t_list, te_t_list = value2label(tr_t_tl, tr_t_list, te_t_list)
    tr_t_score = f1_score(tr_t_tl, tr_t_list, average='macro')

    tr_r_list, te_r_list = value2label(tr_r_tl, tr_r_list, te_r_list)
    tr_r_score = f1_score(tr_r_tl, tr_r_list, average='macro')

    tr_ent_list, te_ent_list = value2label(tr_ent_tl, tr_ent_list, te_ent_list)
    tr_ent_score = f1_score(tr_ent_tl, tr_ent_list, average='macro')

    tr_g_list, te_g_list = value2label(tr_g_tl, tr_g_list, te_g_list)
    tr_g_score = f1_score(tr_g_tl, tr_g_list, average='macro')

    tr_e_list, te_e_list = value2label(tr_e_tl, tr_e_list, te_e_list)
    tr_e_score = f1_score(tr_e_tl, tr_e_list, average='macro')

    #test
    te_s_score = f1_score(te_s_tl, te_s_list, average='macro')
    te_t_score = f1_score(te_t_tl, te_t_list, average='macro')
    te_r_score = f1_score(te_r_tl, te_r_list, average='macro')
    te_ent_score = f1_score(te_ent_tl, te_ent_list, average='macro')
    te_g_score = f1_score(te_g_tl, te_g_list, average='macro')
    te_e_score = f1_score(te_e_tl, te_e_list, average='macro')

    print('tr:')
    print(tr_s_score)
    print(tr_t_score)
    print(tr_r_score)
    print(tr_ent_score)
    print(tr_g_score)
    print(tr_e_score)
    print('')
    print('te:')
    print(te_s_score)
    print(te_t_score)
    print(te_r_score)
    print(te_ent_score)
    print(te_g_score)
    print(te_e_score)

    #return tr_s_auc, te_s_auc, tr_t_auc, te_t_auc, tr_r_auc, te_r_auc, tr_ent_auc, te_ent_auc, tr_g_auc, te_g_auc, tr_e_auc, te_e_auc
    return tr_s_score, te_s_score, tr_t_score, te_t_score, tr_r_score, te_r_score, tr_ent_score, te_ent_score, tr_g_score, te_g_score, tr_e_score, te_e_score

def avg_count(y, x):
    #y ~ 12/12
    #x 12/12 ~ 12/25
    label = []
    for i in range(len(x)):
        will = []
        for j in range(6, len(x[0]), 1):
            if x[i][j] - y[i][j] > 0:
                will.append(1)
            if x[i][j] - y[i][j] <= 0:
                will.append(0)
        label.append(will)

    return label

def increase_or_not(label):
    s_count_will = 0
    s_count_not = 0
    t_count_will = 0
    t_count_not = 0
    r_count_will = 0
    r_count_not = 0
    ent_count_will = 0
    ent_count_not = 0
    g_count_will = 0
    g_count_not = 0
    e_count_will = 0
    e_count_not = 0

    for i in range(len(label)):
        if label[i][0] == 1:
            s_count_will += 1
        else:
            s_count_not += 1

        if label[i][1] == 1:
            t_count_will += 1
        else:
            t_count_not += 1

        if label[i][2] == 1:
            r_count_will += 1
        else:
            r_count_not += 1

        if label[i][3] == 1:
            ent_count_will += 1
        else:
            ent_count_not += 1

        if label[i][4] == 1:
            g_count_will += 1
        else:
            g_count_not += 1

        if label[i][5] == 1:
            e_count_will += 1
        else:
            e_count_not += 1

    print("Shopping_Proportionate increase: " + str(s_count_will))
    print("Shopping_Proportionate decreased: " + str(s_count_not))
    print("Travel_Proportionate increase: " + str(t_count_will))
    print("Travel_Proportionate decreased: " + str(t_count_not))
    print("Restaurant and Dining_Proportionate increase: " + str(r_count_will))
    print("Restaurant and Dining_Proportionate decreased: " + str(r_count_not))
    print("Entertainment_Proportionate increase: " + str(ent_count_will))
    print("Entertainment_Proportionate decreased: " + str(ent_count_not))
    print("Games_Proportionate increase: " + str(g_count_will))
    print("Games_Proportionate decreased: " + str(g_count_not))
    print("Education_Proportionate increase: " + str(e_count_will))
    print("Education_Proportionate decreased: " + str(e_count_not))

def cross_validation(feature, label, run, cv):

    total_item = len(label)
    sitems = math.floor(total_item / cv)
    litems = math.ceil(total_item / cv)
    l = (total_item % cv)
    s = run - (total_item % cv)

    cv_feature = []
    cv_label = []

    #lase_run
    if run >= total_item % cv :
        for item in range(sitems):
            cv_feature.append(feature[l*litems + s*sitems + item])
            cv_label.append(label[l*litems + s*sitems + item])


    else:
        for item in range(litems):
            cv_feature.append(feature[run*litems + item])
            cv_label.append(label[run*litems + item])


    return cv_feature, cv_label

def main():

    data_x = []
    data_y = []

    print("csv loading...")
    data_x, data_y = load_generative_csv()
    #data_x ~ 12/12
    #data_y 12/12 ~ 12/25
    #labele改成上升下降
    label = avg_count(data_x, data_y)

    #算有多少上升多少下降
    increase_or_not(label)

    #feature取出來
    feature = []
    for i in range(len(data_x)):
        feature.append([data_x[i][0], data_x[i][1], data_x[i][2], data_x[i][3], data_x[i][4], data_x[i][5], data_x[i][6], data_x[i][7], data_x[i][8], data_x[i][9], data_x[i][10], data_x[i][11]])

    #打散資料集
    random.seed(2)
    tmp = list(zip(feature, label))
    random.shuffle(tmp)
    feature, label = zip(*tmp)

    feature = feature_one_hard(feature)

    cv = 5
    tr_s_auc = []
    tr_t_auc = []
    tr_r_auc = []
    tr_ent_auc = []
    tr_g_auc = []
    tr_e_auc = []

    te_s_auc = []
    te_t_auc = []
    te_r_auc = []
    te_ent_auc = []
    te_g_auc = []
    te_e_auc = []

    for test in range(cv):
        print("------------------ " + str(test+1) + " ------------------")
        tr_feature = []
        tr_label = []
        te_feature = []
        te_label = []

        for cvi in range(cv):
            cv_feature = []
            cv_label = []
            cv_feature, cv_label = cross_validation(feature, label, cvi, cv=cv)

            if cvi == test:
                te_feature = te_feature + cv_feature
                te_label = te_label + cv_label
            else:
                tr_feature = tr_feature + cv_feature
                tr_label = tr_label + cv_label

        tr_s_au, te_s_au, tr_t_au, te_t_au, tr_r_au, te_r_au, tr_ent_au, te_ent_au, tr_g_au, te_g_au, tr_e_au, te_e_au = mfmo(tr_feature, tr_label, te_feature, te_label)


        tr_s_auc.append(tr_s_au)
        tr_t_auc.append(tr_t_au)
        tr_r_auc.append(tr_r_au)
        tr_ent_auc.append(tr_ent_au)
        tr_g_auc.append(tr_g_au)
        tr_e_auc.append(tr_e_au)

        te_s_auc.append(te_s_au)
        te_t_auc.append(te_t_au)
        te_r_auc.append(te_r_au)
        te_ent_auc.append(te_ent_au)
        te_g_auc.append(te_g_au)
        te_e_auc.append(te_e_au)

    print('')
    print("mf_training_auc")
    print("Shopping:  " + str("%.3f" % np.mean(tr_s_auc)))
    print("Travel:  " + str("%.3f" % np.mean(tr_t_auc)))
    print("Restaurant and Dining:  " + str("%.3f" % np.mean(tr_r_auc)))
    print("Entertainment:  " + str("%.3f" % np.mean(tr_ent_auc)))
    print("Games:  " + str("%.3f" % np.mean(tr_g_auc)))
    print("Education:  " + str("%.3f" % np.mean(tr_e_auc)))
    print('')
    print("mf_test_auc")
    print("Shopping:  " + str("%.3f" % np.mean(te_s_auc)))
    print("Travel:  " + str("%.3f" % np.mean(te_t_auc)))
    print("Restaurant and Dining:  " + str("%.3f" % np.mean(te_r_auc)))
    print("Entertainment:  " + str("%.3f" % np.mean(te_ent_auc)))
    print("Games:  " + str("%.3f" % np.mean(te_g_auc)))
    print("Education:  " + str("%.3f" % np.mean(te_e_auc)))




if __name__ == '__main__':
    main()
