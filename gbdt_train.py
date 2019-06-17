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


def load_data():

    feature_num=15768
    pos_num=523
    neg_num=1499


    #파일에 접근하기

    # a = 'C:/Users/viplab/PycharmProjects/GBDT_classifier/pedestrian_feature/neg/'
    # files = listdir("C:/Users/viplab/PycharmProjects/GBDT_classifier/pedestrian_feature/neg/")
    a = 'C:/Users/viplab/PycharmProjects/GBDT_classifier/pedestrian_feature/all_local_patch_584/neg/'
    files = listdir("C:/Users/viplab/PycharmProjects/GBDT_classifier/pedestrian_feature/all_local_patch_584/neg/")
    x_fail=[]
    for f in files:
        # print(f)
        x_fail.append(np.loadtxt(a+f, dtype=np.int))



    x_fail=np.array([x_fail])
    #print(x_fail)


    # a = 'C:/Users/viplab/PycharmProjects/GBDT_classifier/pedestrian_feature/pos/'
    # files = listdir("C:/Users/viplab/PycharmProjects/GBDT_classifier/pedestrian_feature/pos/")
    a = 'C:/Users/viplab/PycharmProjects/GBDT_classifier/pedestrian_feature/all_local_patch_584/pos/'
    files = listdir("C:/Users/viplab/PycharmProjects/GBDT_classifier/pedestrian_feature/all_local_patch_584/pos/")
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


    # y_real=np.loadtxt('C:/Users/viplab/PycharmProjects/GBDT_classifier/pedestrian_feature/pos_label.txt', dtype=np.int)
    # #y_good=np.loadtxt('C:/Users/IVP Lab/source/repos/GAN_daseul/GAN_daseul/result/0528_finaldata/good_label.txt', dtype=np.int)
    # y_fail=np.loadtxt('C:/Users/viplab/PycharmProjects/GBDT_classifier/pedestrian_feature/neg_label.txt', dtype=np.int)

    y_real=np.loadtxt('C:/Users/viplab/PycharmProjects/GBDT_classifier/pedestrian_feature/all_local_patch_584/pos_label_origin.txt', dtype=np.int)
    #y_good=np.loadtxt('C:/Users/IVP Lab/source/repos/GAN_daseul/GAN_daseul/result/0528_finaldata/good_label.txt', dtype=np.int)
    y_fail=np.loadtxt('C:/Users/viplab/PycharmProjects/GBDT_classifier/pedestrian_feature/all_local_patch_584/neg_label_origin.txt', dtype=np.int)

    print(y_real.shape)
    #print(y_good.shape)
    print(y_fail.shape)

    X=np.vstack([x_real,x_fail]) #X=np.vstack([x_real, x_good, x_fail])
    print(X.shape)
    y=np.hstack([y_real, y_fail]) #y=np.hstack([y_real, y_good, y_fail])
    print(y.shape)

    # iris=datasets.load_iris()
    # x_iris=iris.data
    # y_iris=iris.target
    # print(y_iris.shape)
    # print(x_iris.shape)

    return X, y

def train():
    X,y=load_data()
    X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.3, random_state=0)#stratify=y
    #70%는 train셋, 30%는 테스트 셋
    #adaboost 분류기 생성하기

###############################################################################################
    gbrt = GradientBoostingClassifier(max_depth=2, n_estimators=120, random_state=1)
    #분류기 학습하기
    #model=abc.fit(X_train, y_train)
    gbrt.fit(X_train, y_train)

    #테스트 데이터셋으로 예측하기
    y_preda=gbrt.predict(X_test)

    print("Accuracy_nb", metrics.accuracy_score(y_preda, y_test))

    errors = [mean_squared_error(y_test, y_pred) for y_pred in gbrt.staged_predict(X_test)]
    bst_n_estimators=np.argmin(errors)
    print(bst_n_estimators)

###############################################################################################
    #Regression이기 때문에 predict를 하면 [0.83, 1.5. ,,, ..]이렇게 나옴.
    #우리는 분류기를 사용해야 되기 때문에 gradientBosstingClassifer를 사용한다.
    #최적의 estimators를 사용한 경우와 사용하지 않은 경우를 비교한다.
    gbrt_b = GradientBoostingClassifier(max_depth=2, n_estimators=54, random_state=1)
    gbrt_b.fit(X_train, y_train)
    y_predb=gbrt_b.predict(X_test)
    print(y_predb)
    print("Accuracy_b", metrics.accuracy_score(y_predb, y_test))
    print(X_test.shape)

    return gbrt_b

def test(model, test):


    y_predb=model.predict(test)
    print(y_predb)
    #accuracy 출력
    return y_predb

