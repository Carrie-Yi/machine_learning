# 关于iris数据集
# 鸢尾花（Iris）数据集:  3类共150条记录，每个记录有4个特征。通过4个特征预测3分类。
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
# sklearn.svm将被弃用，移动到.multiclass 多分类模块中
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc  ###计算roc和auc
# from sklearn import cross_validation  ##这个将被弃用,被移动在model_selection模块内
from sklearn.model_selection import train_test_split
# ovr（一对多）： 多分类问题
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize

from itertools import cycle
#from scipy import interp (interp被移动到np.interp)
from sklearn.metrics import roc_auc_score

iris = datasets.load_iris()
X = iris.data
y = iris.target

# 二分类
# y = label_binarize(y,classes=[0,1])
# n_classes = y.shape[1]

# 对于多分类，二值化标签
# Binarize the output
y = label_binarize(y, classes=[0, 1, 2])
n_classes = y.shape[1]

random_state = np.random.RandomState(0)
n_samples, n_features = X.shape
X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]
# shuffle and split training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=0)
# X_train,X_test,y_train,y_test = cross_validation.train_test_split(X,y,test_size=.3,random_state=0)

# #定义分类器
# classifier = svm.SVC(kernel='linear',probability=True,random_state=random_state)
# # 测试集Y_test的打分
# # y_score = classifier.fit(X_train,y_train).decision_function(X_test)

classifier = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True, random_state=random_state))
y_score = classifier.fit(X_train, y_train).decision_function(X_test)

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
# iris是3类
n_classes1 = len(y_test[0])
for i in range(n_classes1):
    # 第i类的fpr和tpr和auc
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
# ravel 化成水平
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
roc_auc['micro'] = auc(fpr['micro'], tpr['micro'])
# ------------------------------绘制特定类别的roc图---------------------------------------------
# ------------------二分类roc------------------
# 二分类，第2类的roc曲线和面积（第一类0.95,第二类0.6）
plt.figure()
lw = 2
# fpr[i],tpr[i],roc_auc[i]
plt.plot(fpr[1], tpr[1], color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[1])
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()
# ------------------多分类roc------------------
# 多分类，第3类的roc曲线和面积（第一类0.95，第二类0.6，第三类0.78）
plt.figure()
# 线的粗细>0
lw = 2
plt.plot(fpr[2], tpr[2], color='pink',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
plt.plot([0, 1], [0, 1], color='red', lw=lw, linestyle='--')
# plt.plot(fpr[2], tpr[2], color='darkorange',
#          lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
# plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
# xlim,ylim x和y的范围
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()
# ------------------------------绘制多标签问题的ROC曲线（计算宏观平均roc和aoc面积）
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    # interp 被移动到np.interp
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
plt.figure()
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Some extension of Receiver operating characteristic to multi-class')
plt.legend(loc="lower right")
plt.show()
