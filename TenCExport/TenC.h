#pragma once


#ifndef COMPILER_MSVC
#define COMPILER_MSVC
#endif //COMPILER_MSVC

#ifndef NOMINMAX
#define NOMINMAX
#endif //NOMINMAX
#ifndef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS
#endif //_CRT_SECURE_NO_WARNINGS

#define D_SCL_SECURE_NO_WARNINGS

#ifndef _TENC_H_
#define _TENC_H_
#include <eigen/Dense>
#include <opencv2/opencv.hpp>
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/public/session_options.h"
#include "tinyxml.h"
#include <sstream>
#include <io.h>
#include <iostream>
#include <stdlib.h>
#include <windows.h>
#include <queue>
#include <map>
#include <cmath>
#include <direct.h>
#include <numeric>
//#include "xgboost/c_api.h"
#include <algorithm>
#include <stdio.h>
#include <stdlib.h>
//#include "TenInteface.h"
#define MY_WIDTH 1216
#define MY_HEIGHT 1936
using namespace std;
using namespace tensorflow;
using namespace cv;

#ifdef TenCExport
#define TenC_API __declspec(dllexport)
#else 
#define TenC_API __declspec(dllimport)
#endif // !TenCExport

typedef unsigned long long *myHandler;

//为输出设置一个结构体
struct myPoint
{
	int x;//x坐标
	int y;//y坐标
};
struct RegionResult
{
	float score;//视野块得分
	vector<float> score1;//model1得分
	vector<float> score2;//model2替换得分
	vector<myPoint> points;//区域坐标
};
//extern "C" TenC_API void feature23(float*p,std::string iniPath);//输入是配置文件路径，输出是23个特征
extern "C" TenC_API myHandler initializeConfig(string, string);
extern "C" TenC_API RegionResult regionCompute(cv::Mat &inMat, myHandler handler);
extern "C" TenC_API void freeHandler(myHandler handler);

struct modelResult
{
	float score;
	Tensor mask;
	vector<vector<Point>> contours;
	vector<vector<Point>> regionPoints;//得到的每一个区域的所需要返回的左上角的点
									   //保存一个model2输出的结果，就不用为model2单独设置一个结构了
									   //每个regionPoints中的Point的得分都保存在score2中
	vector<vector<float>> score2;//保存model2的得分

};
struct modelResult2
{
	float score1;//保存model1的预测
	Tensor mask;//保存model1的掩码
	vector<Point> points;//保存model1每个区域的最大值点的坐标
	vector<float> score2;//保存model2的预测结果
};
//设置一个结构体保存一个subImg的相关信息
struct subImgInfo
{
	float score;//一个subImg的得分(结合model1和model2综合算出结果)
	Point point;//在整张大图中的坐标
	map<int, modelResult> modelResults;//保存model1和model2的所有结果
};

class  TF_RES {
public:
	float score;
	int index;
	std::string path;

	bool operator<(const TF_RES& rt) {
		return this->score > rt.score;
	};
};
//结构体写的有问题
struct slideResult2
{
	float score;
	vector<float> score2;
	vector<cv::Point> points;
};
//正确的结构体
struct slideResult
{
	vector<float> score1;
	map<int, vector<float>> score2;
	vector<cv::Point> points;
};

struct node
{
	float value;
	int index;
};

struct features
{
	float max1;
	float max2;
	float max3;
	float max4;
	float max5;
	float m1_avg;
	float m1_median;
	float m1_std;

	float m1_thresh1_num;//大于0.05的个数
	float m1_thresh2_num;//0.5
	float m1_thresh3_num;//0.8
	float m2_thresh4_num;//大于0.9的个数(用model2替换计算的值)
	float m2_thresh5_num;//大于0.95的个数(用model2替换计算的值)

	float m1_thresh1_sum;//大于0.05的总分
	float m1_thresh2_sum;
	float m1_thresh3_sum;
	float m2_thresh4_sum;
	float m2_thresh5_sum;

	float m1_thresh1_avg;//大于0.05的平均
	float m1_thresh2_avg;
	float m1_thresh3_avg;
	float m2_thresh4_avg;
	float m2_thresh5_avg;
};

struct ImgCVMeta
{
	int index;
	cv::Mat mat;
	std::string path;
};
typedef std::list<TF_RES> TfResultList;
struct Config {
	//int tfWorkers;
	//int imgReaders;
	//int filterNum;

	//std::string dir_prefix;

	//multi thread ancess
	std::unique_ptr<tensorflow::Session> session;

	//std::queue<ImgCVMeta*> imgQue;
	//std::mutex mtxImgQue;

	//TfResultList resList;
	//std::mutex mtxResOut;

	//volatile int runningWorkers;
	//volatile int curDirTotalFrams;

	int img_width;
	int img_height;
	int img_channel;

	//tf graph congif
	std::string GRAPH;
	std::string opsInput;
	std::vector<std::string> opsOutput;

	//tf batch config
	int batch_size;
	tensorflow::Tensor tensorInput;
	tensorflow::TensorShape shapeInput;
	tensorflow::SessionOptions* sessionOptions;
	/////////////////////////////////////////////
};

class /*_declspec(dllexport)*/ MyExport
{
public:
	//要测试的图片路径
	string m_imgPath;
	//保存的路径
	string m_savePath;
	bool m_savePathFlag;
	string m_model1Path;
	string m_model2Path;
	string m_xgModelPath;
	vector<cv::Rect> m_myRects;
	Config m_tfConfig1;
	Config m_tfConfig2;
	MyExport() {}
	MyExport(string model1, string model2, string imgPath = "", string savePath = ""){
		m_imgPath = imgPath;
		m_savePath = savePath;
		m_model1Path = model1;
		m_model2Path = model2;
		initializeConfigMember();
		m_myRects = getRects(MY_WIDTH, MY_HEIGHT, 512, 512, 3, 5);
	}
	MyExport(string iniFile)
	{
		getIniConf(iniFile);
		m_savePathFlag = false;
		//创建保存的文件夹
		CreateSaveDir();
		//开始初始化两个Config
		initializeConfigMember();
		
		//vector<string> opsOutput2;
		//opsOutput2.push_back("dense_4/Sigmoid:0");
		//tfConfig(
		//	&m_tfConfig2, 256, 256, 3, 1, "input_2:0", opsOutput2,
		//	m_model2Path
		//);
	}
	void getIniConf(string iniFile);
	//接下来拷贝函数名即可
	bool tfConfig(
		Config *tfConfig, int width, int height, int channel,
		int batchsize, string opsInput, vector<string> opsOutput, string modelPath
	);
	void splitString(string str, string c, vector<string> *outStr);
	vector<Rect> getRects(int srcImgWidth, int srcImgHeight, int dstImgWidth, int dstImgHeight, int m, int n);
	void getSubImg(Mat *srcImg, Mat *dstImg, Rect myRect);
	Tensor getTensorInput(Config *tfConfig);
	Status readTensorFromMat(Config *tfConfig, vector<Mat> *resizedImg, Tensor *outTensor, int convert = 0);
	void TensorToMat(Tensor mask, Mat *dst);
	vector<cv::Point> getRegionPoints2(Mat *mask, float threshold);
	void getFiles(string path, vector<string> &files, string suffix);
	void TensorToString(Tensor mask, string *out, int channel);
	int imgResize(Mat *srcImg, Mat *dstImg, int newW, int newH);
	Tensor getTensorInput(Config *tfConfig, int batchSize);
	string getFileNamePrefix(string *path);

	void TestMulImg();
	void TestOneImg(Config *tfConfig, cv::Mat *inMat, map<int, modelResult2> *modelResults, string imgPath);
	void testImageOnModel2(Config* tfConfig,Mat *inMat, map<int, modelResult2> *modelResults, string path);
	void saveModelResult(string savePath,map<int, modelResult2> *modelResults);

	void getResult2(cv::Mat *inMat, slideResult *result);
	void getResult(cv::Mat *inMat, RegionResult *result);
	void ConvertResult(slideResult *result1,RegionResult *result2);

	features xgBoost(float *predict);
	void CreateSaveDir();
	
	void initializeConfigMember();

	//features xgBoostPredict(features *myFeatures);

	/*bool cmpUp(node a, node b);
	bool cmpDown(node a, node b);
	void sort_indexes(vector<float> &v, vector<int> &idx, char order);*/
};


#endif // !_TENC_H_


