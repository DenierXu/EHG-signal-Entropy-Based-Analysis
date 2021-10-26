import numpy as np
import pandas as pd
import warnings
from sklearn.preprocessing import scale
from sklearn.model_selection import cross_val_score,StratifiedKFold,train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import ADASYN,SMOTE
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.utils import shuffle
from sklearn.feature_selection import SelectKBest,SelectPercentile,chi2
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import  AdaBoostClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import KFold


'''
1.  对EHG信号做STFT， 并在EHG信号的频率分量上计算熵特征， 应用SMOTE解决样本不平衡问题：
    先Partition, 再对测试集与训练集分别进行合成采样
    
    本实验进行特征选择，观察gnb分类器在不同主成分特征维度下的分类效果
'''

#100次5折交叉验证
N = 1
# 一次STFT的窗口大小
windowSize = 200
#Hz, raw signal sample frequency
sampleFrequency = 20
#EHG信号经过STFT后产生的频率分量个数
N_components = int(4 / (sampleFrequency / windowSize) + 1)
#ID + 3个通道特征个数 + Preterm
N_features =1 + N_components * 6 + 1
#主成分特征的个数
n_components = N_components


#分类器
gnb = GaussianNB(priors=[0.45,0.55],var_smoothing=0.001)
svm = SVC(probability=True, kernel='rbf', C=2,degree=1)  # SVM模型
lr = LogisticRegression(C=10, penalty='l2',tol=0.1)  # 逻辑回归模型
knn = KNeighborsClassifier(n_neighbors=5,leaf_size = 1,p=1)

#使用的分类器
model = gnb
#文件名称
fileNumber = 14
#读取特征文件
data = pd.read_csv('F:/Entropy_STFT/1_1/ApEn_TF'+ str(fileNumber) +'_300.csv',index_col=0)
#随机打乱顺序
random_state = 2021
data = data.sample(frac=1.0,random_state=6)

#数据存储
columns = ['Component','Accuracy','Sensitivity','Specificity','AUC']
result = pd.DataFrame(columns=columns)
#用来存储实验结果
co = []
ac = []
se = []
sp = []
au = []


#5折交叉验证
C = 5
for h in range(n_components):
    #去除文件中ID和标签Preterm
    X = data.drop(data.columns[[0, N_features - 1]], axis=1)
    #标签Label
    Y = data['Preterm']

    # 转化为ndarray
    X = np.array(X)
    Y = np.array(Y)

    # 所使用的特征索引
    # S1通道，无tr
    #index_1 = np.arange(0,N_components * 2,2)
    # S1通道，有tr
    # index_1 = np.arange(1,N_components * 2,2)

    # S2通道，无tr
    #index_1 = np.arange(N_components * 2,N_components * 4,2)
    # S1通道，有tr
    # index_1 = np.arange(N_components * 2 + 1,N_components * 4,2)

    #第3通道，无tr
    index_1 = np.arange(N_components * 4, N_components * 6, 2)
    # 第3通道，有tr
    #index_1 = np.arange(N_components * 4 + 1, N_components * 6, 2)

    #存储实验结果，我们取平均值
    Accuracys = np.zeros(C * N)
    AUCs = np.zeros(C * N)
    Specificitys = np.zeros(C * N)
    Sensitivitys = np.zeros(C * N)

    for l in range(N):
        # 表示5折交叉验证实验进行的次数，从0次开始计算
        i = 0
        #5折交叉验证
        skf = StratifiedKFold(n_splits=C)
        # 用来记录画平均ROC曲线的信息
        mean_tpr = 0.0
        mean_fpr = np.linspace(0, 1, 101)

        #5折交叉验证
        for train_index, test_index in skf.split(X, Y):
            #训练样本
            X_train = X[train_index, :]
            Y_train = Y[train_index]
            #测试样本
            X_test = X[test_index, :]
            Y_test = Y[test_index]

            # 选择第三通道信号训练的特征
            X_train = X_train[:, index_1]
            X_test = X_test[:, index_1]

            # 进行特征选择，在不同主成分维度下的分类效果
            selector = PCA(n_components=h+1)
            X_train = selector.fit_transform(X_train, Y_train)
            X_test = selector.transform(X_test)

            #分别进行过采样
            smote = SMOTE(k_neighbors=6,random_state=random_state)
            X_train, Y_train = smote.fit_sample(X_train, Y_train)
            X_test, Y_test = smote.fit_sample(X_test, Y_test)

            #机器学习训练
            model.fit(X_train, Y_train)
            y_pred = model.predict(X_test)  # <class 'numpy.ndarray'>
            probas_ = model.predict_proba(X_test)  # 训练模型后预测每条样本得到两种结果的概率
            fpr, tpr, thresholds = roc_curve(Y_test, probas_[:, 1])  # 该函数得到伪正例、真正例、阈值，这里只使用前两个
            mean_tpr += np.interp(mean_fpr, fpr, tpr)  # 插值函数 interp(x坐标,每次x增加距离,y坐标)  累计每次循环的总值后面求平均值
            #预测的结果
            pred_preterm = y_pred
            #实际的结果
            real_preterm = Y_test


            TP = (pred_preterm == 1) & (real_preterm == 1)  # True positive
            FP = (pred_preterm != real_preterm) & (pred_preterm == 1)  # prediction is positive, while real is negative
            TN = (pred_preterm == real_preterm) & (pred_preterm == 0)  # True negative
            FN = (pred_preterm != real_preterm) & (pred_preterm == 0)

            Accuracy = np.sum(y_pred == Y_test) / len(Y_test)
            Precision = np.sum(TP) / (np.sum(TP) + np.sum(FP))
            Recall = np.sum(TP) / (np.sum(TP) + np.sum(FN))
            F1_score = 2 * Precision * Recall / (Precision + Recall)
            AUC = auc(fpr, tpr)  # 求auc面积
            plt.plot(fpr, tpr, lw=1, label='ROC fold {0:.2f} (area = {1:.2f})'.format(i, AUC))  # 画出当前分割数据的ROC曲线
            TPR = np.sum(TP) / (np.sum(TP) + np.sum(FN))  # True Positive Rate,sensitivity
            sensitivity = TPR
            FPR = np.sum(FP) / (np.sum(TN) + np.sum(FP))  # False Positive Rate
            TNR = np.sum(TN) / (np.sum(FP) + np.sum(TN))  # True  Negative rate, specificity
            specificity = TNR

            Accuracys[i + l * C] = Accuracy
            AUCs[i + l * C] = AUC
            Sensitivitys[i + l * C] = sensitivity
            Specificitys[i + l * C] = specificity
            i += 1

    co.append(h+1)
    Accuracy = round(Accuracys.mean(), 2)
    print("Accuracy:", Accuracy)
    Accuracy_std = round(Accuracys.std(),3)
    print("Accuracy_std:", round(Accuracy_std, 2))
    ac.append(Accuracy)

    AUC = round(AUCs.mean(), 2)
    print("AUC:", AUC)
    AUC_std = round(AUCs.std(),3)
    print("AUC_std:", round(AUC_std, 2))
    au.append(AUC)

    Sensitivity = round(Sensitivitys.mean(), 2)
    print("Sensitivity:", Sensitivity)
    Sensitivity_std = round(Sensitivitys.std(),3)
    print("Sensitivity_std:", round(Sensitivity_std, 2))
    se.append(Sensitivity)

    Specificity = round(Specificitys.mean(), 2)
    print("Specificity:", Specificity)
    Specificity_std = round(Specificitys.std(),3)
    print("Specificity_std:", round(Specificity_std, 2))
    sp.append(Specificity)

result['Component'] = co
result['Accuracy'] = ac
result['Sensitivity'] = se
result['Specificity'] = sp
result['AUC'] = au

#基于时频域近似熵的特征选择结果
result.to_csv('ApEn_TF_FS.csv')



'''
S3通道近似熵改进的特征个数选择
'''
data = pd.read_csv('F:/Entropy_STFT/1_1/ApEn_TF'+ str(fileNumber) +'_300.csv',index_col=0)

#随机打乱顺序A
random_state = 6
data = data.sample(frac=1.0,random_state=6)

columns = ['Component','Accuracy','Sensitivity','Specificity','AUC']
result = pd.DataFrame(columns=columns)

co = []
ac = []
se = []
sp = []
au = []


#SampEn:6,4    ApEn:6,6

C = 5   #表示实验进行的总次数

for h in range(n_components):
    X = data.drop(data.columns[[0, N_features - 1]], axis=1)
    Y = data['Preterm']
    # print(X.shape)  #(228, 21)

    # 转化为ndarray
    X = np.array(X)
    Y = np.array(Y)

    # 所使用的特征索引
    # index_1 = np.arange(0,N_components * 2,2)
    #index_1 = np.arange(1,N_components * 2,2)

    # index_1 = np.arange(N_components * 2,N_components * 4,2)
    #index_1 = np.arange(N_components * 2 + 1,N_components * 4,2)

    #index_1 = np.arange(N_components * 4, N_components * 6, 2)
    index_1 = np.arange(N_components * 4 + 1, N_components * 6, 2)


    Accuracys = np.zeros(C * N)
    AUCs = np.zeros(C * N)
    Specificitys = np.zeros(C * N)  # 5折交叉验证存储5次实验的结果，最后我们取平均值
    Sensitivitys = np.zeros(C * N)

    for l in range(N):
        # 5折交叉验证
        i = 0
        skf = StratifiedKFold(n_splits=C)
        # 用来记录画平均ROC曲线的信息
        mean_tpr = 0.0
        mean_fpr = np.linspace(0, 1, 101)
        for train_index, test_index in skf.split(X, Y):
            # train_index与test_index为下标
            X_train = X[train_index, :]
            X_test = X[test_index, :]

            Y_train = Y[train_index]
            Y_test = Y[test_index]

            # 选择近似熵或者近似熵的改进，默认是2者的组合
            X_train = X_train[:, index_1]
            X_test = X_test[:, index_1]


            # 进行特征选择
            selector = PCA(n_components=h+1)
            #selector = SelectKBest(k=h+1)
            X_train = selector.fit_transform(X_train, Y_train)
            X_test = selector.transform(X_test)

            # adasyn()方法
            # svm,random_state=888,sampen;random_state=4,apen
            #Ada = ADASYN(random_state=random_state)  # SampEn:2,14,20ApEn:14
            Ada = SMOTE(k_neighbors=6,random_state=random_state)  # SampEn:2,14,20ApEn:14
            X_train, Y_train = Ada.fit_sample(X_train, Y_train)
            X_test, Y_test = Ada.fit_sample(X_test, Y_test)

            sc = StandardScaler()
            # sc = MinMaxScaler()
            X_train = sc.fit_transform(X_train)
            X_test = sc.transform(X_test)

            model.fit(X_train, Y_train)
            y_pred = model.predict(X_test)  # <class 'numpy.ndarray'>
            probas_ = model.predict_proba(X_test)  # 训练模型后预测每条样本得到两种结果的概率
            fpr, tpr, thresholds = roc_curve(Y_test, probas_[:, 1])  # 该函数得到伪正例、真正例、阈值，这里只使用前两个

            mean_tpr += np.interp(mean_fpr, fpr, tpr)  # 插值函数 interp(x坐标,每次x增加距离,y坐标)  累计每次循环的总值后面求平均值

            pred_preterm = y_pred
            real_preterm = Y_test

            TP = (pred_preterm == 1) & (real_preterm == 1)  # True positive
            FP = (pred_preterm != real_preterm) & (pred_preterm == 1)  # prediction is positive, while real is negative
            TN = (pred_preterm == real_preterm) & (pred_preterm == 0)  # True negative
            FN = (pred_preterm != real_preterm) & (pred_preterm == 0)
            #print(sum(TP), sum(FP), sum(TN), sum(FN))  # 0 4 43 13

            Accuracy = np.sum(y_pred == Y_test) / len(Y_test)
            Precision = np.sum(TP) / (np.sum(TP) + np.sum(FP))
            Recall = np.sum(TP) / (np.sum(TP) + np.sum(FN))
            F1_score = 2 * Precision * Recall / (Precision + Recall)

            AUC = auc(fpr, tpr)  # 求auc面积
            plt.plot(fpr, tpr, lw=1, label='ROC fold {0:.2f} (area = {1:.2f})'.format(i, AUC))  # 画出当前分割数据的ROC曲线

            TPR = np.sum(TP) / (np.sum(TP) + np.sum(FN))  # True Positive Rate,sensitivity
            sensitivity = TPR
            FPR = np.sum(FP) / (np.sum(TN) + np.sum(FP))  # False Positive Rate

            TNR = np.sum(TN) / (np.sum(FP) + np.sum(TN))  # True  Negative rate, specificity
            specificity = TNR

            Accuracys[i + l * C] = Accuracy
            AUCs[i + l * C] = AUC
            Sensitivitys[i + l * C] = sensitivity
            Specificitys[i + l * C] = specificity

            i += 1

    co.append(h+1)

    Accuracy = round(Accuracys.mean(), 2)
    print("Accuracy:", Accuracy)
    Accuracy_std = round(Accuracys.std(),3)
    print("Accuracy_std:", round(Accuracy_std, 2))
    ac.append(Accuracy)

    AUC = round(AUCs.mean(), 2)
    print("AUC:", AUC)
    AUC_std = round(AUCs.std(),3)
    print("AUC_std:", round(AUC_std, 2))
    au.append(AUC)

    Sensitivity = round(Sensitivitys.mean(), 2)
    print("Sensitivity:", Sensitivity)
    Sensitivity_std = round(Sensitivitys.std(),3)
    print("Sensitivity_std:", round(Sensitivity_std, 2))
    se.append(Sensitivity)

    Specificity = round(Specificitys.mean(), 2)
    print("Specificity:", Specificity)
    Specificity_std = round(Specificitys.std(),3)
    print("Specificity_std:", round(Specificity_std, 2))
    sp.append(Specificity)

result['Component'] = co
result['Accuracy'] = ac
result['Sensitivity'] = se
result['Specificity'] = sp
result['AUC'] = au

#基于时频域近似熵改进的特征选择结果
result.to_csv('ApEn_TFS_FS.csv')


'''
S3通道样本熵的特征个数选择
'''
data = pd.read_csv('F:/Entropy_STFT/1_1/SampEn_TF'+ str(fileNumber) +'_300.csv',index_col=0)
#随机打乱顺序A
random_state = 4
data = data.sample(frac=1.0,random_state=6)

columns = ['Component','Accuracy','Sensitivity','Specificity','AUC']
result = pd.DataFrame(columns=columns)

co = []
ac = []
se = []
sp = []
au = []

C = 5   #表示实验进行的总次数

for h in range(n_components):
    X = data.drop(data.columns[[0, N_features - 1]], axis=1)
    Y = data['Preterm']
    # print(X.shape)  #(228, 21)

    # 转化为ndarray
    X = np.array(X)
    Y = np.array(Y)

    # 所使用的特征索引
    #index_1 = np.arange(0,N_components * 2,2)
    # index_1 = np.arange(1,N_components * 2,2)

    #index_1 = np.arange(N_components * 2,N_components * 4,2)
    # index_1 = np.arange(N_components * 2 + 1,N_components * 4,2)

    index_1 = np.arange(N_components * 4, N_components * 6, 2)
    #index_1 = np.arange(N_components * 4 + 1, N_components * 6, 2)


    Accuracys = np.zeros(C * N)
    AUCs = np.zeros(C * N)
    Specificitys = np.zeros(C * N)  # 5折交叉验证存储5次实验的结果，最后我们取平均值
    Sensitivitys = np.zeros(C * N)

    for l in range(N):

        i = 0  # 表示实验进行的次数，从0次开始计算
        skf = StratifiedKFold(n_splits=C)
        mean_tpr = 0.0  # 用来记录画平均ROC曲线的信息
        mean_fpr = np.linspace(0, 1, 101)

        # sc = StandardScaler()
        # X = sc.fit_transform(X)
        for train_index, test_index in skf.split(X, Y):
            #print('train_index', len(train_index), 'test_index', len(test_index))
            # train_index与test_index为下标
            X_train = X[train_index, :]
            X_test = X[test_index, :]

            Y_train = Y[train_index]
            Y_test = Y[test_index]

            # 选择近似熵或者近似熵的改进，默认是2者的组合
            X_train = X_train[:, index_1]
            X_test = X_test[:, index_1]

            # 进行特征选择，在不同主成分维度下的分类效果
            selector = PCA(n_components=h+1)
            X_train = selector.fit_transform(X_train, Y_train)
            X_test = selector.transform(X_test)


            # adasyn()方法
            # svm,random_state=888,sampen;random_state=4,apen
            #Ada = ADASYN(random_state = random_state)  # SampEn:2,14,20ApEn:14
            Ada = SMOTE(k_neighbors=5,random_state=4)
            X_train, Y_train = Ada.fit_sample(X_train, Y_train)
            X_test, Y_test = Ada.fit_sample(X_test, Y_test)

            sc = StandardScaler()
            X_train = sc.fit_transform(X_train)
            X_test = sc.transform(X_test)

            model.fit(X_train, Y_train)
            y_pred = model.predict(X_test)  # <class 'numpy.ndarray'>
            probas_ = model.predict_proba(X_test)  # 训练模型后预测每条样本得到两种结果的概率
            fpr, tpr, thresholds = roc_curve(Y_test, probas_[:, 1])  # 该函数得到伪正例、真正例、阈值，这里只使用前两个
            #print(fpr.shape)

            mean_tpr += np.interp(mean_fpr, fpr, tpr)  # 插值函数 interp(x坐标,每次x增加距离,y坐标)  累计每次循环的总值后面求平均值

            pred_preterm = y_pred
            real_preterm = Y_test
            #print('X_test的长度：', X_test.shape)

            TP = (pred_preterm == 1) & (real_preterm == 1)  # True positive
            FP = (pred_preterm != real_preterm) & (pred_preterm == 1)  # prediction is positive, while real is negative
            TN = (pred_preterm == real_preterm) & (pred_preterm == 0)  # True negative
            FN = (pred_preterm != real_preterm) & (pred_preterm == 0)
            #print(sum(TP), sum(FP), sum(TN), sum(FN))  # 0 4 43 13

            Accuracy = np.sum(y_pred == Y_test) / len(Y_test)
            Precision = np.sum(TP) / (np.sum(TP) + np.sum(FP))
            Recall = np.sum(TP) / (np.sum(TP) + np.sum(FN))
            F1_score = 2 * Precision * Recall / (Precision + Recall)

            AUC = auc(fpr, tpr)  # 求auc面积
            plt.plot(fpr, tpr, lw=1, label='ROC fold {0:.2f} (area = {1:.2f})'.format(i, AUC))  # 画出当前分割数据的ROC曲线

            TPR = np.sum(TP) / (np.sum(TP) + np.sum(FN))  # True Positive Rate,sensitivity
            sensitivity = TPR
            FPR = np.sum(FP) / (np.sum(TN) + np.sum(FP))  # False Positive Rate

            TNR = np.sum(TN) / (np.sum(FP) + np.sum(TN))  # True  Negative rate, specificity
            specificity = TNR

            Accuracys[i + l * C] = Accuracy
            AUCs[i + l * C] = AUC
            Sensitivitys[i + l * C] = sensitivity
            Specificitys[i + l * C] = specificity

            i += 1

    co.append(h+1)

    Accuracy = round(Accuracys.mean(), 2)
    print("Accuracy:", Accuracy)
    Accuracy_std = round(Accuracys.std(),3)
    print("Accuracy_std:", round(Accuracy_std, 2))
    ac.append(Accuracy)

    AUC = round(AUCs.mean(), 2)
    print("AUC:", AUC)
    AUC_std = round(AUCs.std(),3)
    print("AUC_std:", round(AUC_std, 2))
    au.append(AUC)

    Sensitivity = round(Sensitivitys.mean(), 2)
    print("Sensitivity:", Sensitivity)
    Sensitivity_std = round(Sensitivitys.std(),3)
    print("Sensitivity_std:", round(Sensitivity_std, 2))
    se.append(Sensitivity)

    Specificity = round(Specificitys.mean(), 2)
    print("Specificity:", Specificity)
    Specificity_std = round(Specificitys.std(),3)
    print("Specificity_std:", round(Specificity_std, 2))
    sp.append(Specificity)

result['Component'] = co
result['Accuracy'] = ac
result['Sensitivity'] = se
result['Specificity'] = sp
result['AUC'] = au
#基于时频域样本熵的特征选择结果
result.to_csv('SampEn_TF_FS.csv')


'''
S3通道样本熵改进的特征个数选择
'''
data = pd.read_csv('F:/Entropy_STFT/1_1/SampEn_TF'+ str(fileNumber) +'_300.csv',index_col=0)

#随机打乱顺序A
random_state = 6
data = data.sample(frac=1.0,random_state=6)

columns = ['Component','Accuracy','Sensitivity','Specificity','AUC']
result = pd.DataFrame(columns=columns)

co = []
ac = []
se = []
sp = []
au = []

C = 5   #表示实验进行的总次数

for h in range(n_components):
    X = data.drop(data.columns[[0, N_features - 1]], axis=1)
    Y = data['Preterm']
    # print(X.shape)  #(228, 21)

    # 转化为ndarray
    X = np.array(X)
    Y = np.array(Y)

    # 所使用的特征索引
    # index_1 = np.arange(0,N_components * 2,2)
    #index_1 = np.arange(1,N_components * 2,2)

    # index_1 = np.arange(N_components * 2,N_components * 4,2)
    #index_1 = np.arange(N_components * 2 + 1,N_components * 4,2)

    #index_1 = np.arange(N_components * 4, N_components * 6, 2)
    index_1 = np.arange(N_components * 4 + 1, N_components * 6, 2)


    Accuracys = np.zeros(C * N)
    AUCs = np.zeros(C * N)
    Specificitys = np.zeros(C * N)  # 5折交叉验证存储5次实验的结果，最后我们取平均值
    Sensitivitys = np.zeros(C * N)

    for l in range(N):

        i = 0  # 表示实验进行的次数，从0次开始计算
        skf = StratifiedKFold(n_splits=C)
        mean_tpr = 0.0  # 用来记录画平均ROC曲线的信息
        mean_fpr = np.linspace(0, 1, 101)

        # sc = StandardScaler()
        # X = sc.fit_transform(X)
        for train_index, test_index in skf.split(X, Y):
            #print('train_index', len(train_index), 'test_index', len(test_index))
            # train_index与test_index为下标
            X_train = X[train_index, :]
            X_test = X[test_index, :]

            Y_train = Y[train_index]
            Y_test = Y[test_index]

            # 选择近似熵或者近似熵的改进，默认是2者的组合
            X_train = X_train[:, index_1]
            X_test = X_test[:, index_1]
            # X_train = X_train[:,index_2]
            # X_test = X_test[:,index_2]

            # 进行特征选择
            selector = PCA(n_components=h+1)
            #selector = SelectKBest(k=h+1)
            X_train = selector.fit_transform(X_train, Y_train)
            X_test = selector.transform(X_test)

            # adasyn()方法
            # svm,random_state=888,sampen;random_state=4,apen
            #Ada = ADASYN(random_state=random_state)  # SampEn:2,14,20ApEn:14
            Ada = SMOTE(k_neighbors=5,random_state=1314)  # SampEn:2,14,20ApEn:14
            X_train, Y_train = Ada.fit_sample(X_train, Y_train)
            X_test, Y_test = Ada.fit_sample(X_test, Y_test)

            sc = StandardScaler()
            # sc = MinMaxScaler()
            X_train = sc.fit_transform(X_train)
            X_test = sc.transform(X_test)

            model.fit(X_train, Y_train)
            y_pred = model.predict(X_test)  # <class 'numpy.ndarray'>
            probas_ = model.predict_proba(X_test)  # 训练模型后预测每条样本得到两种结果的概率
            fpr, tpr, thresholds = roc_curve(Y_test, probas_[:, 1])  # 该函数得到伪正例、真正例、阈值，这里只使用前两个
            #print(fpr.shape)

            mean_tpr += np.interp(mean_fpr, fpr, tpr)  # 插值函数 interp(x坐标,每次x增加距离,y坐标)  累计每次循环的总值后面求平均值

            pred_preterm = y_pred
            real_preterm = Y_test
            #print('X_test的长度：', X_test.shape)

            TP = (pred_preterm == 1) & (real_preterm == 1)  # True positive
            FP = (pred_preterm != real_preterm) & (pred_preterm == 1)  # prediction is positive, while real is negative
            TN = (pred_preterm == real_preterm) & (pred_preterm == 0)  # True negative
            FN = (pred_preterm != real_preterm) & (pred_preterm == 0)
            #print(sum(TP), sum(FP), sum(TN), sum(FN))  # 0 4 43 13

            Accuracy = np.sum(y_pred == Y_test) / len(Y_test)
            Precision = np.sum(TP) / (np.sum(TP) + np.sum(FP))
            Recall = np.sum(TP) / (np.sum(TP) + np.sum(FN))
            F1_score = 2 * Precision * Recall / (Precision + Recall)

            AUC = auc(fpr, tpr)  # 求auc面积
            plt.plot(fpr, tpr, lw=1, label='ROC fold {0:.2f} (area = {1:.2f})'.format(i, AUC))  # 画出当前分割数据的ROC曲线

            TPR = np.sum(TP) / (np.sum(TP) + np.sum(FN))  # True Positive Rate,sensitivity
            sensitivity = TPR
            FPR = np.sum(FP) / (np.sum(TN) + np.sum(FP))  # False Positive Rate

            TNR = np.sum(TN) / (np.sum(FP) + np.sum(TN))  # True  Negative rate, specificity
            specificity = TNR

            Accuracys[i + l * C] = Accuracy
            AUCs[i + l * C] = AUC
            Sensitivitys[i + l * C] = sensitivity
            Specificitys[i + l * C] = specificity

            i += 1

    co.append(h+1)

    Accuracy = round(Accuracys.mean(), 2)
    print("Accuracy:", Accuracy)
    Accuracy_std = round(Accuracys.std(),3)
    print("Accuracy_std:", round(Accuracy_std, 2))
    ac.append(Accuracy)

    AUC = round(AUCs.mean(), 2)
    print("AUC:", AUC)
    AUC_std = round(AUCs.std(),3)
    print("AUC_std:", round(AUC_std, 2))
    au.append(AUC)

    Sensitivity = round(Sensitivitys.mean(), 2)
    print("Sensitivity:", Sensitivity)
    Sensitivity_std = round(Sensitivitys.std(),3)
    print("Sensitivity_std:", round(Sensitivity_std, 2))
    se.append(Sensitivity)

    Specificity = round(Specificitys.mean(), 2)
    print("Specificity:", Specificity)
    Specificity_std = round(Specificitys.std(),3)
    print("Specificity_std:", round(Specificity_std, 2))
    sp.append(Specificity)

result['Component'] = co
result['Accuracy'] = ac
result['Sensitivity'] = se
result['Specificity'] = sp
result['AUC'] = au
#基于时频域样本熵改进的特征选择结果
result.to_csv('SampEn_TFS_FS.csv')


