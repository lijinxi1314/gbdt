import numpy as np
# Load libraries
from sklearn.ensemble import AdaBoostClassifier
from sklearn import datasets
# Import train_test_split function
from sklearn.model_selection import train_test_split
#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingClassifier

from os import rename, listdir
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.externals import joblib
import graphviz # doctest: +SKIP
from sklearn import tree
import pydotplus
from sklearn.externals.six import StringIO
from IPython.display import Image
import pydot
import os
import time
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import GridSearchCV
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/graphviz-2.38/release/bin/'

def load_data():

    feature_num=2295
    pos_num=523
    neg_num=1499


    #파일에 접근하기

    a = 'C:/Users/viplab/PycharmProjects/GBDT_classifier_same/pedestrian_feature/three_patch_upper1_lower2/MIT/neg/'
    files = listdir("C:/Users/viplab/PycharmProjects/GBDT_classifier_same/pedestrian_feature/three_patch_upper1_lower2/MIT/neg/")


    x_fail=[]
    for f in files:
        # print(f)
        x_fail.append(np.loadtxt(a+f, dtype=np.int))



    x_fail=np.array([x_fail])
    #print(x_fail)


    a = 'C:/Users/viplab/PycharmProjects/GBDT_classifier_same/pedestrian_feature/three_patch_upper1_lower2/MIT/pos/'
    files = listdir("C:/Users/viplab/PycharmProjects/GBDT_classifier_same/pedestrian_feature/three_patch_upper1_lower2/MIT/pos/")

    x_real=[]
    for f in files:
        # print(f)
        x_real.append(np.loadtxt(a+f, dtype=np.int))


    x_real=np.array([x_real])
    #print(x_real)


    print("pos")
    print(x_real.shape)

    print("neg")
    print(x_fail.shape)

#shape 맞춰주기
    x_real=x_real.reshape(-1, feature_num) #pos data 수
    print("pos2")
    print(x_real.shape)

    x_fail=x_fail.reshape(-1, feature_num) #neg data 수
    print("neg2")

    print(x_fail.shape)



    y_real=np.loadtxt('C:/Users/viplab/PycharmProjects/GBDT_classifier_same/pedestrian_feature/three_patch_upper1_lower2/MIT/pos_label_origin.txt', dtype=np.int)
    y_fail=np.loadtxt('C:/Users/viplab/PycharmProjects/GBDT_classifier_same/pedestrian_feature/three_patch_upper1_lower2/MIT/neg_label_origin.txt', dtype=np.int)


#################### training 923


    print(y_real.shape)
    #print(y_good.shape)
    print(y_fail.shape)

    X=np.vstack([x_real,x_fail]) #X=np.vstack([x_real, x_good, x_fail])
    print("X.shape",X.shape)
    y=np.hstack([y_real, y_fail]) #y=np.hstack([y_real, y_good, y_fail])
    print("y.shape",y.shape)



    return X, y

def plot_feature_importance(dataset, model_bst):
    list_feature_name=[]
    for i in range (len(dataset[0])):
        list_feature_name.append(i)

    print("list_feature_name",len(list_feature_name))
    # list_feature_importance = list(model_bst.feature_importance(importance_type='split', iteration=-1))
    list_feature_importance = list(model_bst.feature_importances_)
    print("list_feature_importance",len(list_feature_importance))
    dataframe_feature_importance = pd.DataFrame(
        {'feature_name': list_feature_name, 'importance': list_feature_importance})
    dataframe_feature_importance20 = dataframe_feature_importance.sort_values(by='importance', ascending=False)[:20]
    print(dataframe_feature_importance20)
    x = range(len(dataframe_feature_importance20['feature_name']))
    plt.xticks(x, dataframe_feature_importance20['feature_name'], rotation=90, fontsize=8)
    plt.plot(x, dataframe_feature_importance20['importance'])
    plt.xlabel("Feature name")
    plt.ylabel("Importance")
    plt.title("The importance of features")
    plt.show()



def train():
    X,y=load_data()
    X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.3, random_state=0)#stratify=y
    #70%는 train셋, 30%는 테스트 셋
    #adaboost 분류기 생성하기
  ############################################
#######################################   0     ############################################
    # gbm0 = GradientBoostingClassifier(random_state=10)
    # gbm0.fit(X_train, y_train)
    #
    # y_predb=gbm0.predict(X_test)
    # y_predprob = gbm0.predict_proba(X_test)[:,1]
    # # print(y_predb)
    # # print("Accuracy_b", metrics.accuracy_score(y_predb, y_test))
    # sub_tree = gbm0.estimators_[69, 0]
    # list_feature_name=[]
    # for i in range (len(X_train[0])):
    #     list_feature_name.append(i)
    # print("fit finsh")
    #
    # # tree.export_graphviz(gbrt_b, out_file=dot_data,feature_names=list_feature_name,class_names=np.unique(y)) # doctest: +SKIP
    # dot_data = tree.export_graphviz(sub_tree,out_file=None, filled=True,rounded=True,
    # special_characters=True, proportion=True)
    # graph = pydotplus.graph_from_dot_data(dot_data)
    # graph.write_pdf("best_depth_pdf/0813_test/MIT_three_upper1_lower2_nonono.pdf")
    # # Image(graph.create_png())
    #
    # plot_feature_importance(X_train,gbm0)
    # joblib.dump(gbm0, 'best_depth_model/0813_test/MIT_three_upper1_lower2_nonono.pkl')
    #
    # #
    # # errors = [mean_squared_error(y_test, y_pred) for y_pred in gbm0.staged_predict(X_test)]
    # # bst_n_estimators=np.argmin(errors)
    # # print("node",bst_n_estimators)
    # print ("Accuracy : %.4g" % metrics.accuracy_score(y_test, y_predb))
    # print ("AUC Score (Train): %f" % metrics.roc_auc_score( y_test, y_predprob))
    #
    #
    # false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_predb)
    # roc_auc = auc(false_positive_rate, true_positive_rate)
    # print("roc_auc",roc_auc)

###############################################################################################

#######################################   1     ############################################
    # param_test1 = {'n_estimators':range(20,120,10)}
    # gsearch = GridSearchCV(estimator= GradientBoostingClassifier(learning_rate=0.1,max_depth=7,random_state=10), param_grid= param_test1, scoring='roc_auc', iid= False, cv= 5)
    # gsearch.fit(X_train, y_train)
    # print("gsearch.best_params_")
    # print(gsearch.best_params_)
    # print("gsearch.best_score_")
    # print(gsearch.best_score_)

####################################    max_depths   ##########################################################
    # max_depths = np.linspace(2, 9, 8, endpoint=True)
    # acc_results = []
    # for max_depth in max_depths:
    #    model = GradientBoostingClassifier(max_depth=max_depth,n_estimators=90)
    #    model.fit(X_train, y_train)
    #    y_pred = model.predict(X_test)
    #    y_predprob = model.predict_proba(X_test)[:,1]
    #
    #    print("max_depth: %d" %max_depth )
    #    print ("Accuracy : %.4g" % metrics.accuracy_score(y_test, y_pred))
    #    acc_results.append(metrics.accuracy_score(y_test, y_pred))
    #
    #    print ("AUC Score (Train): %f" % metrics.roc_auc_score( y_test, y_predprob))
    #
    # print(acc_results)
#######################################   2     ############################################
    # param_test2 = {'max_depth':[3,4,5,6,7,8,9]}
    # gsearch2 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, n_estimators=90, min_samples_leaf=20, max_features='sqrt', subsample=0.8, random_state=10), param_grid = param_test2, scoring='roc_auc',iid=False, cv=5)
    # gsearch2.fit(X_train,y_train)
    # print(gsearch2.best_params_, gsearch2.best_score_)
    #
    # return gsearch2

#

# #######################################   3    ############################################
# #
#     gbrt_b = GradientBoostingClassifier(learning_rate=0.1,max_depth=4, n_estimators=90, random_state=1)
#     gbrt_b.fit(X_train, y_train)
#    ################### change estimators_[]##############################
#     sub_tree = gbrt_b.estimators_[89, 0]
#     list_feature_name=[]
#     for i in range (len(X_train[0])):
#         list_feature_name.append(i)
#     print("fit finsh")
#     # tree.export_graphviz(gbrt_b, out_file=dot_data,feature_names=list_feature_name,class_names=np.unique(y)) # doctest: +SKIP
#     dot_data = tree.export_graphviz(sub_tree,out_file=None, filled=True,rounded=True,
#     special_characters=True, proportion=True)
#     graph = pydotplus.graph_from_dot_data(dot_data)
#     graph.write_pdf("best_depth_pdf/0813_test/MIT_three_upper1_lower2_depth4.pdf")
#     # Image(graph.create_png())
#
#     joblib.dump(gbrt_b, 'best_depth_model/0813_test/MIT_three_upper1_lower2_depth4.pkl')
#
#     plot_feature_importance(X_train,gbrt_b)
#
#
#     y_predb=gbrt_b.predict(X_test)
#     y_predprob = gbrt_b.predict_proba(X_test)[:,1]
#
#     print ("Accuracy : %.4g" % metrics.accuracy_score(y_test, y_predb))
#     print ("AUC Score (Train): %f" % metrics.roc_auc_score( y_test, y_predprob))
#     print(X_test.shape)
#
#     return gbrt_b

#   # #######################################   4   max_features ############################################
#     #
#
    # param_test4 = {'max_features':range(7,20,2)}
    # gsearch4 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, n_estimators=80,max_depth=2,
    #             # min_samples_leaf =60, min_samples_split =1200,
    #                     subsample=0.8, random_state=10),
    #                    param_grid = param_test4, scoring='roc_auc',iid=False, cv=5)
    # gsearch4.fit(X_train, y_train)
    # print(gsearch4.best_params_)
    # print(gsearch4.best_score_)
    # # #
#      # #######################################   5 subsample   ############################################
# # #
#     param_test5 = {'subsample':[0.6,0.7,0.75,0.8,0.85,0.9]}
#     gsearch5 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, n_estimators=80,max_depth=2,
#                 # min_samples_leaf =60, min_samples_split =1200,
#                 max_features=11, random_state=10),
#                        param_grid = param_test5, scoring='roc_auc',iid=False, cv=5)
#     gsearch5.fit(X_train, y_train)
#     print(gsearch5.best_params_)
#     print(gsearch5.best_score_)


#     # #######################################   6     ############################################
#     param_gbdt3 = {'learning_rate':[0.06,0.08,0.1],
#                'n_estimators':[75,80,85,90,95]}
#     gsearch3 = GridSearchCV(estimator=GradientBoostingRegressor( max_depth=4,max_features='sqrt',subsample=0.8,random_state=75),n_jobs=3,
#                             param_grid=param_gbdt3,scoring='neg_mean_squared_error',iid=False,cv=5)
#     gsearch3.fit(X_train, y_train)
#     print(gsearch3.best_params_)
#     print(gsearch3.best_score_)
#
#
#
#     # #######################################     final  ############################################
    gbrt_b = GradientBoostingClassifier(learning_rate=0.1,max_depth=2, n_estimators=160,
                                        max_features='sqrt',
                                        # subsample=0.8,
                                        # min_samples_leaf=20,
                                        random_state=1)
    gbrt_b.fit(X_train, y_train)
   ################### change estimators_[]##############################
    sub_tree = gbrt_b.estimators_[159, 0]
    list_feature_name=[]
    for i in range (len(X_train[0])):
        list_feature_name.append(i)
    print("fit finsh")
    # tree.export_graphviz(gbrt_b, out_file=dot_data,feature_names=list_feature_name,class_names=np.unique(y)) # doctest: +SKIP
    dot_data = tree.export_graphviz(sub_tree,out_file=None, filled=True,rounded=True,
    special_characters=True, proportion=True)
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.write_pdf("best_depth_pdf/0813_test/MIT_three_upper1_lower2_depth2_final.pdf")
    # Image(graph.create_png())

    joblib.dump(gbrt_b, 'best_depth_model/0813_test/MIT_three_upper1_lower2_final.model')

    plot_feature_importance(X_train,gbrt_b)


    y_predb=gbrt_b.predict(X_test)
    y_predprob = gbrt_b.predict_proba(X_test)[:,1]

    print ("Accuracy : %.4g" % metrics.accuracy_score(y_test, y_predb))
    print ("AUC Score (Train): %f" % metrics.roc_auc_score( y_test, y_predprob))
    print(X_test.shape)

    return gbrt_b




def test(model, test):


    y_predb=model.predict(test)
 #   print(y_predb)
    #accuracy 출력
    return y_predb

start=time.time()
model=train()
end=time.time()
print (end-start)
