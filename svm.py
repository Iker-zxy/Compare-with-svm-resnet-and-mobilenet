import numpy as np
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn import metrics
from data.dataset import HSI_Scenes_dataset_SVM(

def train():

    # -----------------------step1: 模型---------------------------
    clf_SVM = svm.SVC() # 支持向量机（Support Vector Machine）

    # -----------------------step2: 数据----------------------------
    train_data,train_label,test_data,test_label = HSI_Scenes_dataset_SVM()
    pca?

    # -----------------------step3: 训练 ----------------------------
    clf_SVM.fit(train_data, train_label)

    # -----------------------step4: 预测 ----------------------------
    test_acc = metrics.accuracy_score(clf_SVM.predict(test_data), test_label)

    print("Support Vector Machine recognition rate: %f "% test_acc[2])

if __name__ == '__main__':
    train()