#include "xgboost/c_api.h"
using namespace std;
#include<iostream>
////
////////// return 0 when success, -1 when failure happens
//////int main() {
//////	
//////
//////
//////
//////	BoosterHandle handle=nullptr;
//////	const char *fileName = "0919_xgboost/xgboost_realtime_predict_test-master/xgb_test.model";
//////	int x = XGBoosterLoadModel(handle, fileName);
//////	
//////	if (x == 0) {
//////		printf("Successfully Loaded Model\n");
//////	}
//////	else {
//////		cout << "fail" << endl;
//////	}
//////	return 0;
//////}
////
////



#include "xgboost/c_api.h"
#include <iostream>
#include <fstream>
#include<iostream>
#include <vector>
#include<conio.h>
#include<stdio.h>
using namespace std;

const int cols = 2295; //d
const int rows = 2022; //num  pos + neg
//#define rows  100 //num  pos + neg

int main() {

	//// create the train data
	ifstream fin("0924_xgboost_traindata/three_upper2_lower1/train_xgboost.txt");
	// double train[rows][cols] = { 0, };
	
	float **train = new float*[rows];
	for (int i = 0; i < rows; i++) {
		train[i] = new float[cols];
		for (int j = 0; j < cols; j++) {
			train[i][j] = 0.0f;
		}
	}
	
	float train_labels[rows] = { 0, };

	double value=0;

	if (!fin) {
		cout << "fail" << endl;

		
	}
	else {
		int count = 0;
			for (int i =0; i < rows; i++) 
			{
				for (int j =0; j <cols ; j++) 
				{
					
					fin >> train[i][j];
					//cout << train[i][j] << endl;
					count++;
				}
				
				

			}
			cout << count << endl;
		}

	const int postive_num = 523;
	for (int i = 0; i < postive_num; i++) {
		train_labels[i] = 1;
		
	}

	for (int i = postive_num; i < rows; i++) {
		train_labels[i] = 0;
	}





	////////////////////////////////////////////////////////////////////////////
	// convert to DMatrix

	DMatrixHandle h_train[1];
	float *train_flatten = new float[cols * rows];
	for (int i = 0; i < cols*rows; i++) {
		train_flatten[i] = train[i / cols][i % cols];
	}
	int matCreated = XGDMatrixCreateFromMat(train_flatten, rows, cols, -1, &h_train[0]);

	// load the labels
	XGDMatrixSetFloatInfo(h_train[0], "label", train_labels, rows);

	// read back the labels, just a sanity check
	bst_ulong bst_result;
	const float *out_floats;
	XGDMatrixGetFloatInfo(h_train[0], "label", &bst_result, &out_floats);
	//for (unsigned int i = 0; i<bst_result; i++)
	//	std::cout << "label[" << i << "]=" << out_floats[i] << std::endl;

	// create the booster and load some parameters
	BoosterHandle h_booster;
	
	//XGBoosterCreate(0, 0, &h_booster);
	//XGBoosterCreate(h_train, 1, &h_booster);
	//XGBoosterSetParam(h_booster, "booster", "gbtree");
	//XGBoosterSetParam(h_booster, "objective", "reg:linear");
	//XGBoosterSetParam(h_booster, "max_depth", "3");
	//XGBoosterSetParam(h_booster, "eta", "0.1");
	//XGBoosterSetParam(h_booster, "min_child_weight", "1");
	//XGBoosterSetParam(h_booster, "subsample", "0.5");
	//XGBoosterSetParam(h_booster, "colsample_bytree", "1");
	//XGBoosterSetParam(h_booster, "num_parallel_tree", "1");

	XGBoosterCreate(h_train, 1, &h_booster);
	XGBoosterSetParam(h_booster, "booster", "gbtree");
	XGBoosterSetParam(h_booster, "objective", "binary:logistic");
	//XGBoosterSetParam(h_booster, "eval_metric", "error");
	XGBoosterSetParam(h_booster, "silent", "1");
	XGBoosterSetParam(h_booster, "max_depth", "3");
	XGBoosterSetParam(h_booster, "eta", "0.1");
	XGBoosterSetParam(h_booster, "min_child_weight", "5");
	XGBoosterSetParam(h_booster, "gamma", "0.4");
	XGBoosterSetParam(h_booster, "colsample_bytree", "1");
	XGBoosterSetParam(h_booster, "subsample", "0.6");
	//XGBoosterSetParam(h_booster, "reg_alpha", "10");



	// perform 200 learning iterations
	for (int iter = 0; iter<200; iter++)  
		XGBoosterUpdateOneIter(h_booster, iter, h_train[0]);

	// predict
	const int sample_rows = 400;
	/*float test[sample_rows][cols];
	for (int i = 0; i<sample_rows; i++)
		for (int j = 0; j<cols; j++)
			test[i][j] = (i + 1) * (j + 1);*/
	/////////////////////     load test feature     ////////////////////////////

	ifstream fin1("0924_xgboost_traindata/three_upper2_lower1/test_xgboost.txt");
	float **test = new float*[rows];
	for (int i = 0; i < rows; i++) {
		test[i] = new float[cols];
		for (int j = 0; j < cols; j++) {
			test[i][j] = 0;
		}
	}

	if (!fin) {
		cout << "fail" << endl;
	}
	else {
		int count1 = 0;
		for (int i = 0; i < sample_rows; i++)
		{
			for (int j = 0; j <cols; j++)
			{
				fin1 >> test[i][j];
				count1++;
			}
		}
		cout << count1 << endl;
	}

	DMatrixHandle h_test;

	float *test_flatten = new float[sample_rows * cols];
	for (int i = 0; i < sample_rows * cols; i++) {
		test_flatten[i] = test[i / cols][i % cols];
	}
	XGDMatrixCreateFromMat(test_flatten, sample_rows, cols, -1, &h_test);
	bst_ulong out_len;
	const float *f;
	XGBoosterPredict(h_booster, h_test, 0, 0, &out_len, &f);

	int cnt_over_half = 0;
	for (unsigned int i = 0; i < out_len; i++)
	{
		if (f[i] >= 0.5f) {
			cnt_over_half++;
		}
		std::cout << "prediction[" << i << "]=" << f[i] << std::endl;
	}
	std::cout << cnt_over_half << '/' << out_len << std::endl;

///////////////////////////////////////////////////////////////////
	// free xgboost internal structures
	XGDMatrixFree(h_train[0]);
	XGDMatrixFree(h_test);
	XGBoosterFree(h_booster);

	cout << "s" << endl;

	for (int i = 0; i < rows; i++) {
		delete[] train[i];
		delete[] test[i];
	}
	delete train;
	delete test;
	delete[] test_flatten;
	delete[] train_flatten;

	return 0;
}
