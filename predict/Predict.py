import numpy as np
import pandas as pd
import warnings
from imblearn.over_sampling import SMOTE
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



#100次5折交叉验证
N = 100
# STFT的窗口大小
windowSize = 200
# Hz, raw signal sample frequency
sampleFrequency = 20
#EHG信号产生的特征个数
N_components = int(4 / (sampleFrequency / windowSize) + 1)
#特征个数 + 2
N_features =1 + N_components * 6 + 1
#训练好的实验数据
fileNumber = 14


#读取数据
data = pd.read_csv('F:/Entropy_STFT/1_1/SampEn_TF' + str(fileNumber) + '_300.csv',index_col=0)


#随机打乱顺序A
random_state = 4
data = data.sample(frac=1.0,random_state=6)    #SampEn:6,4    ApEn:6,6


#画折线图
#columns = ['fpr','tpr']

columns = ['Accuracy','Precision','Recall','F1_score','AUC']
result = pd.DataFrame(columns=columns)

#5折交叉验证
C = 5
#特征
X = data.drop(data.columns[[0,N_features - 1]],axis=1)
#Label
Y = data['Preterm']

#转化为ndarray
X = np.array(X)
Y = np.array(Y)

#所使用的特征索引
# 选取S1通道熵特征
#index_1 = np.arange(0,N_components * 2,2)
# 选取S1通道熵特征的改进
#index_1 = np.arange(1,N_components * 2,2)

# 选取S2通道熵特征
#index_1 = np.arange(N_components * 2,N_components * 4,2)
# 选取S2通道熵特征的改进
#index_1 = np.arange(N_components * 2 + 1,N_components * 4,2)

# 选取S3通道熵特征
#index_1 = np.arange(N_components * 4,N_components * 6,2)
# 选取S3通道熵特征的改进
index_1 = np.arange(N_components * 4 + 1,N_components * 6,2)


#分类器
gnb = GaussianNB(priors = [0.45,0.55], var_smoothing= 0.001)
svm = SVC(probability=True, kernel='rbf', C=2,degree=1)  # SVM模型
lr = LogisticRegression(C=10, penalty='l2',tol=0.1)  # 逻辑回归模型
knn = KNeighborsClassifier(n_neighbors=2,leaf_size = 1,p=1)
forest = RandomForestClassifier(n_estimators=200,max_depth=20)  # 随机森林

#使用的分类器
model = gnb

#存储实验的结果
Accuracys = np.zeros(C*N)
Precisions = np.zeros(C*N)
Recalls = np.zeros(C*N)
F1_scores = np.zeros(C*N)
AUCs = np.zeros(C*N)
Specificitys = np.zeros(C*N)     #5折交叉验证存储5次实验的结果，最后我们取平均值
Sensitivitys = np.zeros(C*N)

#5折交叉验证
skf = StratifiedKFold(n_splits=C)

for l in range(N):

    i = 0  # 表示实验进行的次数，从0次开始计算
    skf = StratifiedKFold(n_splits=C)
    mean_tpr = 0.0  # 用来记录画平均ROC曲线的信息
    mean_fpr = np.linspace(0, 1, 101)

    for train_index, test_index in skf.split(X, Y):
        print('train_index', len(train_index), 'test_index', len(test_index))
        # train_index与test_index为下标
        X_train = X[train_index, :]
        X_test = X[test_index, :]

        Y_train = Y[train_index]
        Y_test = Y[test_index]

        #选择近似熵或者近似熵的改进，默认是2者的组合
        X_train = X_train[:, index_1]
        X_test = X_test[:, index_1]

        # 进行特征选择
        selector = PCA(n_components='mle')
        X_train = selector.fit_transform(X_train, Y_train)
        print('特征个数',X_train.shape)
        X_test = selector.transform(X_test)

        smote = SMOTE(k_neighbors=5,random_state=1314)
        X_train, Y_train = smote.fit_sample(X_train, Y_train)
        X_test, Y_test = smote.fit_sample(X_test, Y_test)


        model.fit(X_train, Y_train)
        y_pred = model.predict(X_test)  # <class 'numpy.ndarray'>
        # 训练模型后预测每条样本得到两种结果的概率
        probas_ = model.predict_proba(X_test)
        # 该函数得到伪正例、真正例、阈值，这里只使用前两个
        fpr, tpr, thresholds = roc_curve(Y_test, probas_[:, 1])
        print(fpr.shape)
        # 插值函数 interp(x坐标,每次x增加距离,y坐标)  累计每次循环的总值后面求平均值
        mean_tpr += np.interp(mean_fpr, fpr, tpr)

        pred_preterm = y_pred
        real_preterm = Y_test
        print('X_test的长度：', X_test.shape)

        TP = (pred_preterm == 1) & (real_preterm == 1)  # True positive
        FP = (pred_preterm != real_preterm) & (pred_preterm == 1)  # prediction is positive, while real is negative
        TN = (pred_preterm == real_preterm) & (pred_preterm == 0)  # True negative
        FN = (pred_preterm != real_preterm) & (pred_preterm == 0)
        print(sum(TP), sum(FP), sum(TN), sum(FN))  # 0 4 43 13

        Accuracy = np.sum(y_pred == Y_test) / len(Y_test)
        Precision = np.sum(TP) / (np.sum(TP) + np.sum(FP))
        Recall = np.sum(TP) / (np.sum(TP) + np.sum(FN))
        F1_score = 2 * Precision * Recall / (Precision + Recall)
        # 求auc面积
        AUC = auc(fpr, tpr)
        # 画出当前分割数据的ROC曲线
        plt.plot(fpr, tpr, lw=1, label='ROC fold {0:.2f} (area = {1:.2f})'.format(i, AUC))
        # True Positive Rate,sensitivity
        TPR = np.sum(TP) / (np.sum(TP) + np.sum(FN))
        sensitivity = TPR
        # False Positive Rate
        FPR = np.sum(FP) / (np.sum(TN) + np.sum(FP))
        # True  Negative rate, specificity
        TNR = np.sum(TN) / (np.sum(FP) + np.sum(TN))
        specificity = TNR

        Accuracys[i+l*C] = Accuracy
        Precisions[i+l*C] = Precision
        Recalls[i+l*C] = Recall
        F1_scores[i+l*C] = F1_score
        AUCs[i+l*C] = AUC
        Sensitivitys[i+l*C] = sensitivity
        Specificitys[i+l*C] = specificity

        i += 1

#输出实验结果
Accuracy = round(Accuracys.mean(),2)
print(Accuracys)
print("Accuracy:",Accuracy)
Accuracy_std = Accuracys.std()
print("Accuracy_std:",round(Accuracy_std,2))

Precision = round(Precisions.mean(),2)
print("Precision:",Precision)
Precision_std = Precisions.std()
print("Precision_std:",round(Precision_std,2))

Recall = round(Recalls.mean(),2)
print("Recall:",Recall)
Recall_std = Recalls.std()
print("Recall_std:",round(Recall_std,2))

F1_score = round(F1_scores.mean(),2)
print("F1_score:",F1_score)
F1_score_std = F1_scores.std()
print("F1_score_std:",round(F1_score_std,2))

AUC = round(AUCs.mean(),2)
print("AUC:",AUC)
AUC_std = AUCs.std()
print("AUC_std:",round(AUC_std,2))

Specificity = round(Specificitys.mean(),2)
print("Specificity:",Specificity)
Specificity_std = Specificitys.std()
print("Specificity_std:",round(Specificity_std,2))

Sensitivity = round(Sensitivitys.mean(),2)
print("Sensitivity:",Sensitivity)
Sensitivity_std = Sensitivitys.std()
print("Sensitivity_std:",round(Sensitivity_std,2))





