import numpy as np
import shutil
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
from local_patch_gbdt_train import load_data, train, test
from shutil import copyfile
#
# # adaboost 라이브러리, naive 라이브러리 찾아보기.


model=train()
#prediction대로 폴더 분류
# fail = 'C:/Users/viplab/PycharmProjects/GBDT_classifier/pedestrian_result/all_local_patch_584/MIT_test_neg/'
# real = 'C:/Users/viplab/PycharmProjects/GBDT_classifier/pedestrian_result/all_local_patch_584/MIT_test_pos/'
#
# # ##Test용
#
# a = 'C:/Users/viplab/PycharmProjects/GBDT_classifier/pedestrian_test/all_local_patch_584/MIT_test_feature/pos/'
# files = listdir("C:/Users/viplab/PycharmProjects/GBDT_classifier/pedestrian_test/all_local_patch_584/MIT_test_feature/pos/")
# test_image_dir="C:/Users/viplab/PycharmProjects/GBDT_classifier/pedestrian_test/all_local_patch_584/MIT_test_image/test_pos/"
#
# # a = 'C:/Users/viplab/PycharmProjects/GBDT_classifier/pedestrian_test/combine_pos_neg_feature/'
# # files = listdir("C:/Users/viplab/PycharmProjects/GBDT_classifier/pedestrian_test/combine_pos_neg_feature/")
# # test_image_dir="C:/Users/viplab/PycharmProjects/GBDT_classifier/pedestrian_test/combine_pos_neg_img/"
#
#
# test_data=[]
# file_list=[]
# for f in files:
#     print(f)
#     file_list.append(f)
#     test_data.append(np.loadtxt(a+f, dtype=np.float))
#
# test_data=np.array([test_data])
# print(test_data.shape)
# test_num=400
# test_data=test_data.reshape(test_num,-1)# test 데이터의 수대로 파라메터 변경
# print("teststestsetsetsetset")
# print(test_data.shape)
# test_prediction=test(model, test_data)
#
# #그림
# for i in range(len(file_list)):
#     idx=file_list[i].split(".")
#
#     nfile=test_image_dir+idx[0]+".bmp"
#     print(nfile)
#     if test_prediction[i]==-1:
#         shutil.copy(nfile, fail)
#
#     if test_prediction[i]==1:
#         shutil.copy(nfile, real)
#
#
# #정확도
# count=0;
# for i in range(len(file_list)):
#     idx=file_list[i].split(".")
#
#     # if test_prediction[i]==-1:
#     #     count+=1;
#
#
#     if test_prediction[i]==1:
#         count+=1;
#
# print("acc");
# print(count/test_num);
