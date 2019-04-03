import os
import numpy as np
from sklearn.decomposition import PCA
from sklearn import neighbors
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn import tree
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics
from data.dataset import HSI_readall

def train():

    # -----------------------step1: 模型---------------------------
    clf_KNN = neighbors.KNeighborsClassifier()  # K近邻（K Nearest Neighbor）
    clf_LDA = LinearDiscriminantAnalysis()  # 线性鉴别分析（Linear Discriminant Analysis）
    clf_SVM = svm.SVC() # 支持向量机（Support Vector Machine）
    clf_LR = LogisticRegression() # 逻辑回归（Logistic Regression）
    clf_RF = RandomForestClassifier() # 随机森林（Random Forest）
    clf_DTC = tree.DecisionTreeClassifier() # 决策树
    # clf_GBDT = GradientBoostingClassifier(n_estimators = 200) # GBDT
    print('------------Finishing Loading Model----------------')

    # -----------------------step2: 数据----------------------------
    train_data,train_label = HSI_readall("train")
    test_data,test_label = HSI_readall("test")
    print('------------Finishing Loading Data----------------')

    # -----------------------step3: 训练 ----------------------------
    clf_KNN.fit(train_data, train_label)
    clf_LDA.fit(train_data, train_label)
    clf_SVM.fit(train_data, train_label)
    clf_LR.fit(train_data, train_label)
    clf_RF.fit(train_data, train_label)
    clf_DTC.fit(train_data, train_label)
    # clf_GBDT.fit(train_data, train_label)
    print('------------Finishing Model Training----------------')

    # -----------------------step4: 预测 ----------------------------
    test_acc = np.zeros(7)

    test_acc[0] = metrics.accuracy_score(clf_KNN.predict(test_data), test_label)
    test_acc[1] = metrics.accuracy_score(clf_LDA.predict(test_data), test_label)
    test_acc[2] = metrics.accuracy_score(clf_SVM.predict(test_data), test_label)
    test_acc[3] = metrics.accuracy_score(clf_LR.predict(test_data), test_label)
    test_acc[4] = metrics.accuracy_score(clf_RF.predict(test_data), test_label)
    test_acc[5] = metrics.accuracy_score(clf_DTC.predict(test_data), test_label)
    # test_acc[6] = metrics.accuracy_score(clf_GBDT.predict(test_data), test_label)

    print("K Nearest Neighbor recognition rate: %f "% test_acc[0])
    print("Linear Discriminant Analysis recognition rate: %f "% test_acc[1])
    print("Support Vector Machine recognition rate: %f "% test_acc[2])
    print("Logistic Regression recognition rate: %f "% test_acc[3])
    print("Random Forest recognition rate %f "% test_acc[4])
    print("Decision Tree Classifier recognition rate: %f "% test_acc[5])
    # print("Gradient Boosting Decision Tree recognition rate:%f "% test_acc[6])

if __name__ == '__main__':
    train()