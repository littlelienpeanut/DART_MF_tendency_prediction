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
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error
from sklearn.metrics import f1_score
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

def avg_pred(tr_label, te_label):
    one_count = 0
    zero_count = 0
    for l in tr_label:
        if l == 1:
            one_count += 1

        else:
            zero_count += 1

    target = max(one_count, zero_count)
    if target == one_count:
        target = 1
    else:
        target = 0

    tr_pred = [target] * len(tr_label)
    te_pred = [target] * len(te_label)


    return tr_pred, te_pred

def main():

    data_x = []
    data_y = []

    print("csv loading...")
    data_x, data_y = load_generative_csv()
    #data_x ~ 12/12
    #data_y 12/12 ~ 12/25
    label = avg_count(data_x, data_y)

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

        #knn
        tr_s_tl = []
        tr_t_tl = []
        tr_r_tl = []
        tr_ent_tl = []
        tr_g_tl = []
        tr_e_tl = []

        for i in range(len(tr_label)):
            tr_s_tl.append(tr_label[i][0])
            tr_t_tl.append(tr_label[i][1])
            tr_r_tl.append(tr_label[i][2])
            tr_ent_tl.append(tr_label[i][3])
            tr_g_tl.append(tr_label[i][4])
            tr_e_tl.append(tr_label[i][5])

        te_s_tl = []
        te_t_tl = []
        te_r_tl = []
        te_ent_tl = []
        te_g_tl = []
        te_e_tl = []

        for i in range(len(te_label)):
            te_s_tl.append(te_label[i][0])
            te_t_tl.append(te_label[i][1])
            te_r_tl.append(te_label[i][2])
            te_ent_tl.append(te_label[i][3])
            te_g_tl.append(te_label[i][4])
            te_e_tl.append(te_label[i][5])


        #training預測
        tr_s_pred, te_s_pred = avg_pred(tr_s_tl, te_s_tl)
        tr_t_pred, te_t_pred = avg_pred(tr_t_tl, te_t_tl)
        tr_r_pred, te_r_pred = avg_pred(tr_r_tl, te_r_tl)
        tr_ent_pred, te_ent_pred = avg_pred(tr_ent_tl, te_ent_tl)
        tr_g_pred, te_g_pred = avg_pred(tr_g_tl, te_g_tl)
        tr_e_pred, te_e_pred = avg_pred(tr_e_tl, te_e_tl)


        tr_s_auc.append(f1_score(tr_s_tl, tr_s_pred, average='macro'))
        tr_t_auc.append(f1_score(tr_t_tl, tr_t_pred, average='macro'))
        tr_r_auc.append(f1_score(tr_r_tl, tr_r_pred, average='macro'))
        tr_ent_auc.append(f1_score(tr_ent_tl, tr_ent_pred, average='macro'))
        tr_g_auc.append(f1_score(tr_g_tl, tr_g_pred, average='macro'))
        tr_e_auc.append(f1_score(tr_e_tl, tr_e_pred, average='macro'))



        te_s_auc.append(f1_score(te_s_tl, te_s_pred, average='macro'))
        te_t_auc.append(f1_score(te_t_tl, te_t_pred, average='macro'))
        te_r_auc.append(f1_score(te_r_tl, te_r_pred, average='macro'))
        te_ent_auc.append(f1_score(te_ent_tl, te_ent_pred, average='macro'))
        te_g_auc.append(f1_score(te_g_tl, te_g_pred, average='macro'))
        te_e_auc.append(f1_score(te_e_tl, te_e_pred, average='macro'))




    print("avg_training_auc")
    print("Shopping:  " + str("%.3f" % np.mean(tr_s_auc)))
    print("Travel:  " + str("%.3f" % np.mean(tr_t_auc)))
    print("Restaurant and Dining:  " + str("%.3f" % np.mean(tr_r_auc)))
    print("Entertainment:  " + str("%.3f" % np.mean(tr_ent_auc)))
    print("Games:  " + str("%.3f" % np.mean(tr_g_auc)))
    print("Education:  " + str("%.3f" % np.mean(tr_e_auc)))
    print("")
    print("avg_test_auc")
    print("Shopping:  " + str("%.3f" % np.mean(te_s_auc)))
    print("Travel:  " + str("%.3f" % np.mean(te_t_auc)))
    print("Restaurant and Dining:  " + str("%.3f" % np.mean(te_r_auc)))
    print("Entertainment:  " + str("%.3f" % np.mean(te_ent_auc)))
    print("Games:  " + str("%.3f" % np.mean(te_g_auc)))
    print("Education:  " + str("%.3f" % np.mean(te_e_auc)))








if __name__ == '__main__':
    main()
