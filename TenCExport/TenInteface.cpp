//#include "TenInteface.h"

#include "TenC.h"

void feature23(float* p ,string iniPath)
{
	MyExport myexport(iniPath);
	float predict = 0;
	features myFeatures;
	myFeatures = myexport.xgBoost(&predict);
	/*float *p = new float[23];*/
	memcpy(p, &myFeatures, 23 * sizeof(float));
	/*return p;*/
	//return 0.0;
}

//��һ�������õ����һ��ʵ��
myHandler initializeConfig(string model1Path, string model2Path){
	MyExport *myexport = new MyExport(model1Path, model2Path);
	myHandler handler = (myHandler)myexport;//ǿת
	return handler;
}

RegionResult regionCompute(cv::Mat &inMat,myHandler handler)
{
	MyExport *myexport = (MyExport*)handler;
	//Ȼ��Ϳ��Ե��ú�����
	RegionResult result;
	myexport->getResult(&inMat, &result);
	//RegionResult result2;
	//myexport->ConvertResult(&result1, &result2);
	return result;
}
void freeHandler(myHandler handler)
{
	MyExport *myexport = (MyExport*)handler;
	delete myexport;
}