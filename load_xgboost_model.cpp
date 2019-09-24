//////////使用c++加载xgboost的model///////////

#include "xgboost/c_api.h"
#include <assert.h>
using namespace std;
#include<iostream>
#include <fstream>
////////////////////////////////load sucessful///////////////////////////////////
int main() {
	BoosterHandle booster;
	const char *model_path = "0919_xgboost/xgboost_realtime_predict_test-master/xgb_test.model";

	// create booster handle first
	XGBoosterCreate(NULL, 0, &booster);

	// by default, the seed will be set 0
	XGBoosterSetParam(booster, "seed", "0");

	// load model
	
	int x=XGBoosterLoadModel(booster, model_path);
	if (x == 0) {
		printf("Successfully Loaded Model\n");
	}



	const int feat_size = 2295;
	const int num_row = 10;
	//float feat[num_row][feat_size];

	//// create some fake data for predicting
	//for (int i = 0; i < num_row; ++i) {
	//	for (int j = 0; j < feat_size; ++j) {
	//		feat[i][j] = (i + 1) * (j + 1);
	//	}
	//}


	ifstream fin("0919_xgboost/xgboost_realtime_predict_test-master/10test_three_patch_upper1_lower2.txt");
	double feat[num_row][feat_size];

	double value = 0;

	if (!fin) {
		cout << "fail" << endl;


	}
	else {
		int count = 0;
		for (int i = 0; i < num_row; i++)
		{
			for (int j = 0; j <feat_size; j++)
			{

				fin >> feat[i][j];
				count++;
			}


		}
		cout << count << endl;
	}



	// convert 2d array to DMatrix
	DMatrixHandle dtest;
	XGDMatrixCreateFromMat(reinterpret_cast<float*>(feat),
		num_row, feat_size, NAN, &dtest);
	// predict
	bst_ulong out_len;
	const float *f;
	XGBoosterPredict(booster, dtest, 0, 0, &out_len, &f);
	cout << out_len << endl;
	assert(out_len == num_row);
	for (int i = 0; i < out_len; i++) {
		std::cout << "prediction["<<i<<"]=" << f[i] << std::endl;
	}


	// free memory
	XGDMatrixFree(dtest);
	XGBoosterFree(booster);
}
