#include "TenC.h"



//
////template<typename T>
//bool MyExport::cmpUp(node a, node b)
//{
//	if (a.value < b.value)
//	{
//		return true;
//	}
//	return false;
//}
////template<typename T>
//bool MyExport::cmpDown(node a, node b)
//{
//	if (a.value > b.value)
//	{
//		return true;
//	}
//	return false;
//}
////使用sort()函数来实现排序并返回索引,BubbleSort速度太慢
////template<typename T>
//void MyExport::sort_indexes(vector<float> &v, vector<int> &idx, char order) {
//	node *a = new node[v.size()];
//	for (int i = 0; i < v.size(); i++) {
//		a[i].value = v[i];
//		a[i].index = i;
//	}
//	//将a转存到vector中
//	vector<node> vectorA(a, a + v.size());
//	if (order == 'a') {
//		sort(vectorA.begin(), vectorA.end(), MyExport::cmpUp);
//	}
//	if (order == 'd') {
//		sort(vectorA.begin(), vectorA.end(), MyExport::cmpDown);
//	}
//	for (int i = 0; i < v.size(); i++) {
//		idx.push_back(vectorA[i].index);
//	}
//	delete[] a;
//}

vector<cv::Point> MyExport::getRegionPoints2(Mat *mask, float threshold)
{
	//cout << "enter getRegionPoints2" <<endl;
	//先直接进行筛选操作
	double minVal;
	double maxVal;
	Point minLoc;
	Point maxLoc;
	minMaxLoc(*mask, &minVal, &maxVal, &minLoc, &maxLoc);
	//cout << "maxVal:" << maxVal << endl;
	//对图像进行过滤，大于阈值的等于原图像
	cv::threshold(*mask, *mask, threshold*maxVal, maxVal, THRESH_TOZERO);
	//cout << "after thresHold ,the mask is" << *mask << endl;
	//归一化到0-255
	cv::normalize(*mask, *mask, 0, 255, NORM_MINMAX, CV_8UC1);
	//cout << "after normalize ,the mask is" <<endl<< *mask << endl;
	//寻找连通域的lable图
	cv::Mat labels;
	//conn知道到底有几个连通域，其中0代表的是背景，1-(conn-1)，则是前景的部分
	int conn = cv::connectedComponents(*mask, labels, 8, CV_32S);
	//cout << "the lables is:"<<endl << labels << endl;
	//求每个连通域最大值的坐标，若有多个最大值，取第一个最大值
	vector<int> maxValueConn(conn, 0);//保存每个连通域的最大值
	vector<cv::Point> points(conn, cv::Point(0, 0));

	for (int i = 0; i < labels.rows; i++) {
		int *LinePtr = (int*)labels.ptr(i);
		auto LinePtrMask = (*mask).ptr(i);
		for (int j = 0; j < labels.cols; j++) {
			//查看这个点属于哪一个连通域(1-(conn-1))
			int label = *(LinePtr + j);
			if (label == 0) {
				continue;
			}
			int value = *(LinePtrMask + j);
			//只有大于的时候，才会记录，等于的时候，不保存，为了避免以后会有重复的最大值，只取第一个最大值
			if (value > maxValueConn[label]) {
				maxValueConn[label] = value;//保留最大值
				points[label].x = j;//保留最大值的下标
				points[label].y = i;
			}
		}
	}
	//还有将points转为512*512中的点
	for (int i = 0; i < points.size(); i++) {
		points[i].x = int((points[i].x + 0.5) * (512 / 16));
		points[i].y = int((points[i].y + 0.5) * (512 / 16));
	}
	return points;//记住，第一个点不代表什么东西
}

void MyExport::getIniConf(string iniFile)
{
	string strGroup = "INFO";
	string imgPathName = "imgPath";
	string savePathName = "savePath";
	string model1PathName = "model1Path";
	string model2PathName = "model2Path";
	string xgmodelPathName = "xgModelPath";

	char c1[MAX_PATH];
	char c2[MAX_PATH];
	char c3[MAX_PATH];
	char c4[MAX_PATH];
	char c5[MAX_PATH];
	string var;
	string imgPath;
	string savePath;
	string model1Path;
	string model2Path;
	string xgmodelPath;
	GetPrivateProfileString(strGroup.c_str(), imgPathName.c_str(), "default", c1, MAX_PATH, iniFile.c_str());
	imgPath = c1;
	GetPrivateProfileString(strGroup.c_str(), savePathName.c_str(), "default", c2, MAX_PATH, iniFile.c_str());
	savePath = c2;
	GetPrivateProfileString(strGroup.c_str(), model1PathName.c_str(), "default", c3, MAX_PATH, iniFile.c_str());
	model1Path = c3;
	GetPrivateProfileString(strGroup.c_str(), model2PathName.c_str(), "default", c4, MAX_PATH, iniFile.c_str());
	model2Path = c4;
	GetPrivateProfileString(strGroup.c_str(), xgmodelPathName.c_str(), "default", c5, MAX_PATH, iniFile.c_str());
	xgmodelPath = c5;
	cout << "getIniConf model2Path " << model2Path << endl;
	m_imgPath = imgPath;
	m_savePath = savePath;
	m_model1Path = model1Path;
	m_model2Path = model2Path;
	m_xgModelPath = xgmodelPath;

}
//通过路径得到一个文件的前缀名
string MyExport::getFileNamePrefix(string *path)
{
	if ((*path) == ""){
		cout << "getFileNamePrefix: the path should not be null!" << endl;
		return "";
	}
	vector<string> pathSplit;
	splitString(*path, "\\", &pathSplit);
	//带有后缀名的文件名
	string imgName = pathSplit[pathSplit.size() - 1];
	//分解掉.
	vector<string> imgSplit;
	splitString(imgName, ".", &imgSplit);
	string prefix = imgSplit[imgSplit.size() - 2];
	return prefix;
}

Tensor MyExport::getTensorInput(Config *tfConfig, int batchSize)
{
	auto shapeInput = tensorflow::TensorShape();
	shapeInput.AddDim(batchSize);//here
	shapeInput.AddDim(tfConfig->img_height);
	shapeInput.AddDim(tfConfig->img_width);
	shapeInput.AddDim(tfConfig->img_channel);
	Tensor tensorInput = Tensor(tensorflow::DT_UINT8, shapeInput);
	return tensorInput;
}

int MyExport::imgResize(Mat *srcImg, Mat *dstImg, int newW, int newH)
{
	if ((*srcImg).empty())
	{
		cout << "srcImg is empty" << endl;
		return -1;
	}
	Size dsize = Size(newW, newH);
	resize(*srcImg, *dstImg, dsize);
	return 0;
}

void MyExport::TensorToString(Tensor mask, string *out, int channel)
{
	auto output_c = mask.tensor<float, 4>();
	for (int j = 0; j < mask.dim_size(1); j++) {
		for (int k = 0; k < mask.dim_size(2); k++) {
			float tmp = output_c(0, j, k, channel);
			*out = *out + " " + to_string(tmp);
		}
	}
}

void MyExport::getFiles(string path, vector<string> &files, string suffix)
{
	intptr_t hFile = 0;
	struct _finddata_t fileinfo;
	string p;
	//printf("enter getFiles\n");
	if ((hFile = _findfirst(p.assign(path).append("\\*").c_str(), &fileinfo)) != -1)
	{
	//	printf("enter if\n");
		do
		{
			//如果是目录,迭代之  
			//如果不是,加入列表  
			if ((fileinfo.attrib &  _A_SUBDIR))
			{
				if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0)
					getFiles(p.assign(path).append("\\").append(fileinfo.name), files, suffix);
			}
			else
			{
				string tempFilename = string(fileinfo.name);

				if (!tempFilename.substr(tempFilename.find("."), 4).compare(suffix))
				{
					files.push_back(p.assign(path).append("\\").append(fileinfo.name));
				}

			}
		} while (_findnext(hFile, &fileinfo) == 0);
		_findclose(hFile);
	}
}

void MyExport::TensorToMat(Tensor mask, Mat *dst)
{
	float *data = new float[(mask.dim_size(1))*(mask.dim_size(2))];
	auto output_c = mask.tensor<float, 4>();
	//cout << "data 1 :" << endl;
	for (int j = 0; j < mask.dim_size(1); j++) {
		for (int k = 0; k < mask.dim_size(2); k++) {
			data[j * mask.dim_size(1) + k] = output_c(0, j, k, 1);
			//cout << data[j * mask.dim_size(1) + k] << " ";
		}
	}
	Mat myMat = Mat(mask.dim_size(1), mask.dim_size(2), CV_32FC1, data);
	//cout << endl;
	//cout << "myMat" << endl;
	//cout << myMat << endl;
	*dst = myMat.clone();
	delete[]data;
}

Status MyExport::readTensorFromMat(Config *tfConfig, vector<Mat> *resizedImg, Tensor *outTensor, int convert)
{
	if (resizedImg->size() < 1)
	{
		return Status(tensorflow::error::Code::INVALID_ARGUMENT,
			"arg_list_mat->batch.size() < 1");
	}
	int tem_size = resizedImg->size();
	int w = tfConfig->img_width;
	int h = tfConfig->img_height;
	int ch = tfConfig->img_channel;
	tensorflow::Tensor tem_tensor_res(tensorflow::DataType::DT_FLOAT,
		tensorflow::TensorShape({ tem_size, h, w, ch }));

	auto mapTensor = tem_tensor_res.tensor<float, 4>();
	//auto resPointer = tem_tensor_res.tensor<float, 4>().data();

	int countNum = 0;
	for (auto &temImg : (*resizedImg)) {
		for (int i = 0; i < h; ++i) {
			auto temLinePtr = temImg.ptr(i);//temImg->mat.ptr(i);
			for (int j = 0; j < w; ++j) {
				//在这里做一下归一化
				//本来顺序是0,1,2,换成2,1,0的时候就偏偏正确了，解释是与rgb通道有关系，不过到底在哪里搞反的还有待斟酌
				if (convert == 1)
				{
					mapTensor(countNum, i, j, 2) = (((*(temLinePtr + j * 3))*1.0f) / 255 - 0.5) * 2;
					mapTensor(countNum, i, j, 1) = (((*(temLinePtr + j * 3 + 1))*1.0f) / 255 - 0.5) * 2;
					mapTensor(countNum, i, j, 0) = (((*(temLinePtr + j * 3 + 2))*1.0f) / 255 - 0.5) * 2;
				}
				if (convert == 0)
				{
					mapTensor(countNum, i, j, 0) = (((*(temLinePtr + j * 3))*1.0f) / 255 - 0.5) * 2;
					mapTensor(countNum, i, j, 1) = (((*(temLinePtr + j * 3 + 1))*1.0f) / 255 - 0.5) * 2;
					mapTensor(countNum, i, j, 2) = (((*(temLinePtr + j * 3 + 2))*1.0f) / 255 - 0.5) * 2;
				}

				//mapTensor(countNum, i, j, 0) = (*(temLinePtr + j * 3))*1.0f;// / 255. * 2 - 1;
				//mapTensor(countNum, i, j, 1) = (*(temLinePtr + j * 3 + 1))*1.0f;// / 255. * 2 - 1;
				//mapTensor(countNum, i, j, 2) = (*(temLinePtr + j * 3 + 2))*1.0f;// / 255. * 2 - 1;

				/**(resPointer + i) = (*(temLinePtr + j * 3)) / 255. * 2 - 1;
				*(resPointer + i + 1) = (*(temLinePtr + j * 3 + 1)) / 255. * 2 - 1;
				*(resPointer + i + 2) = (*(temLinePtr + j * 3 + 2)) / 255. * 2 - 1;*/
			}
		}
		++countNum;
	}
	outTensor->CopyFrom(tem_tensor_res,
		tensorflow::TensorShape({ tem_size, h, w, ch }));

	return Status::OK();
}

Tensor MyExport::getTensorInput(Config *tfConfig)
{
	auto shapeInput = tensorflow::TensorShape();
	shapeInput.AddDim(tfConfig->batch_size);//here
	shapeInput.AddDim(tfConfig->img_height);
	shapeInput.AddDim(tfConfig->img_width);
	shapeInput.AddDim(tfConfig->img_channel);
	Tensor tensorInput = Tensor(tensorflow::DT_UINT8, shapeInput);
	return tensorInput;
}



void MyExport::getSubImg(Mat *srcImg, Mat *dstImg, Rect myRect)
{
	if ((*srcImg).empty())
	{
		cout << "srcImg is null" << endl;
		return;
	}
	*dstImg = (*srcImg)(myRect)/*.clone()*/;//不进行clone，会影响效率
}


//有冗余的裁剪
vector<Rect> MyExport::getRects(int srcImgWidth, int srcImgHeight, int dstImgWidth, int dstImgHeight, int m, int n)
{
	vector<Rect> myRects;
	//计算每次裁剪的间隔(hDirect,wDirect)
	int wDirect = (srcImgWidth - dstImgWidth) / (m - 1);
	int hDirect = (srcImgHeight - dstImgHeight) / (n - 1);

	for (int i = 0; i < n; i++) {
		for (int j = 0; j < m; j++) {
			int topValue = i*hDirect;
			int leftValue = j*wDirect;
			Rect myRect(leftValue, topValue, dstImgWidth, dstImgHeight);
			myRects.push_back(myRect);
		}
	}
	return myRects;

}

void MyExport::splitString(string str, string c, vector<string> *outStr)
{
	int pos = 0;

	while (str.find(c) != -1)
	{
		pos = str.find(c);
		string tmp = str.substr(0, pos);
		(*outStr).push_back(tmp);
		str = str.substr(pos + c.size());
	}
	(*outStr).push_back(str);
}


bool MyExport::tfConfig(
	Config *tfConfig, int width, int height, int channel,
	int batchsize, string opsInput, vector<string> opsOutput, string modelPath
)
{
	tfConfig->img_width = width;
	tfConfig->img_height = height;
	tfConfig->img_channel = channel;
	tfConfig->batch_size = batchsize;
	tfConfig->opsInput = opsInput;
	(tfConfig->opsOutput).insert((tfConfig->opsOutput).end(), opsOutput.begin(), opsOutput.end());

	//string modelPath = "Z:\\HanWei\\model12\\weights\\new\\model1_340_10x_adapt.pb";
	
	tfConfig->GRAPH = modelPath;

	tfConfig->sessionOptions = new tensorflow::SessionOptions;
	//sessionOptions->config.set_log_device_placement(true);
	tfConfig->sessionOptions->config.mutable_gpu_options()->set_allow_growth(true);
	tfConfig->sessionOptions->config.mutable_gpu_options()->set_force_gpu_compatible(true);
	tfConfig->sessionOptions->config.mutable_device_count()->insert({ "GPU",1 });
	//printf("set cpu to 0\n");
	//tfConfig->sessionOptions->config.mutable_device_count()->insert({ "CPU",0 });
	tensorflow::GraphDef graph_def;
	Status load_graph_status =
		ReadBinaryProto(tensorflow::Env::Default(),
			tfConfig->GRAPH,
			&graph_def);
	//graph::SetDefaultDevice("/gpu:0", &graph_def);
	if (!load_graph_status.ok()) {
		cout << "[LoadGraph] load graph failed!\n";
		return false;
	}
	tfConfig->session.reset(tensorflow::NewSession(*(tfConfig->sessionOptions)));
	auto status_creat_session = tfConfig->session.get()->Create(graph_def);
	cout << "create session success\n";
	if (!status_creat_session.ok()) {
		cout << "[LoadGraph] creat session failed!\n" << endl;
		return false;
	}
	return true;

}

void MyExport::TestMulImg()
{
	vector<string> files;
	cout << "getFiles" << endl;
	getFiles(m_imgPath, files, ".tif");
	cout << "after getFiles" << endl;
	//在这里就应该初始化tffunc
	//图像已经resize完成，开始初始化tffun，batchsize设置为15
	Config tffuncM1;
	vector<string> opsOutput;
	opsOutput.push_back("dense_2/Sigmoid:0");
	opsOutput.push_back("conv2d_1/truediv:0");
	tfConfig(
		&tffuncM1, 512, 512, 3, 1, "input_1:0", opsOutput,
		m_model1Path
	);
	for (int i = 0; i < files.size(); i++) {
		cout << "the "<< i <<" image is running\n";
		

		string imgNameWithoutSuffix = getFileNamePrefix(&files[i]);
		map<int, modelResult2> modelResults;
		Mat Tmp = imread(files[i]);
		TestOneImg(&tffuncM1, &Tmp, &modelResults,files[i]);
		//已经获得了points之后，就可以传入model2进行计算了
		testImageOnModel2(&m_tfConfig2,&Tmp, &modelResults,files[i]);
		//得到model2的结果后，保存结果
		string fileNamePrefix = getFileNamePrefix(&files[i]);
		string savePath = m_savePath + fileNamePrefix;
		CreateDirectoryA(savePath.c_str(), NULL);
		savePath = savePath + "/" + fileNamePrefix + ".xml";
		saveModelResult(savePath, &modelResults);

	}
}

//float MyExport::xgBoostPredict(features *myFeatures)
//{
//	float *test = new float[23];
//	memcpy(test, myFeatures, 23 * sizeof(float));
//	BoosterHandle booster;
//	XGBoosterCreate(NULL, 0, &booster);
//	XGBoosterSetParam(booster, "seed", "0");
//	XGBoosterLoadModel(booster, m_xgModelPath.c_str());
//	DMatrixHandle h_test;
//	XGDMatrixCreateFromMat((float *)test, 1, 23, -1, &h_test);
//	bst_ulong out_len;
//	const float *f;
//	XGBoosterPredict(booster, h_test, 0, 0, &out_len, &f);
//
//	delete[]test;
//	return *f;
//}

void MyExport::CreateSaveDir()
{
	if (m_savePath == ""){
		cout << "savePath is null,so didn't save the result\n";
		return;
	}
	DWORD dirType = GetFileAttributesA(m_savePath.c_str());
	if (dirType == INVALID_FILE_ATTRIBUTES){
		//如果不存在，则创建文件夹
		m_savePathFlag = CreateDirectoryA(m_savePath.c_str(), NULL);
		if (!m_savePathFlag){
			cout << "create savePath failed" << endl;
		}
	}
	if (dirType == FILE_ATTRIBUTE_DIRECTORY){
		m_savePathFlag = true;
	}
}

features MyExport::xgBoost(float *predict)
{
	vector<string> files;
	getFiles(m_imgPath, files, ".tif");

	vector<slideResult> results;
	for (int i = 0; i < files.size(); i++)
	{
		cout << "the " << i << " image is running" << endl;
		cv::Mat srcImg = cv::imread(files[i]);
		slideResult result;
		getResult2(&srcImg, &result);
		results.push_back(result);
	}
	cout << "result complete\n" << endl;
	//暂时输出model2的得分以方便观察
	//map<int, vector<float>> model2Scores;

	vector<slideResult> results2;//用model2替换了之后的版本
	for (int i = 0; i < results.size(); i++){
		//一个result代表一张1216*1936的大图

		//vector<float> model2Score;//暂时用来存储一张1216*1936的图像的model2的值

		slideResult result = results[i];
		for (int j = 0; j < result.score1.size(); j++){
			float score = 0;
			int key = j;
			map<int, vector<float>>::iterator iter = result.score2.find(key);
			if (iter != results[i].score2.end()){
				if (iter->second.size() > 0){//在model1的一个512*512的小块中，有model2的预测值，则用这个预测值进行替换					
					float maxValue = *max_element(iter->second.begin(), iter->second.end());

					//model2Score.insert(model2Score.end(), iter->second.begin(), iter->second.end());

					if (maxValue > score) {
						result.score1[j] = maxValue;
					}
				}
			}
		}
		results2.push_back(result);
		//model2Scores.insert(pair<int, vector<float>>(i, model2Score));
	}
	cout << "results2 ini end\n";
	//开始计算23个特征
	vector<float> scores1;
	vector<float> scores2;
	vector<int> ind;
	for (vector<slideResult>::iterator iter = results.begin(); iter != results.end(); iter++){
		for (int j = 0; j < iter->score1.size(); j++){
			scores1.push_back(iter->score1[j]);
		}	
	}
	for (vector<slideResult>::iterator iter = results2.begin(); iter != results2.end(); iter++) {
		for (int j = 0; j < iter->score2.size(); j++) {
			scores2.push_back(iter->score1[j]);
		}
	}
	cout << "scores1 and scores2 complete\n" << endl;
	//存储结果方便观察
	if (m_savePathFlag){
		string saveScores1 = m_savePath + "/scores1.txt";
		FILE *file = fopen(saveScores1.c_str(), "w+");
		for (int i = 0; i < scores1.size(); i++) {
			fprintf(file, "%f\n", scores1[i]);
		}
		fclose(file);
		string saveScores2 = m_savePath + "/scores2.txt";
		FILE *file2 = fopen(saveScores2.c_str(), "w+");
		for (int i = 0; i < scores2.size(); i++) {
			fprintf(file2, "%f\n", scores2[i]);
		}
		fclose(file2);
	}

	//string model2savepath = m_savePath + "/model2.txt";
	//FILE *file3 = fopen(model2savepath.c_str(), "w+");
	//for (map<int,vector<float>>::iterator iter = model2Scores.begin(); iter != model2Scores.end(); iter++)
	//{
	//	int i = iter->first;
	//	fprintf(file3, "the %d image is: \n", i);
	//	for (int j = 0; j < iter->second.size(); j++)
	//	{
	//		fprintf(file3, "%f\n", iter->second[j]);
	//	}
	//}
	//fclose(file3);
	cout << "write file complete\n";
	std::sort(scores1.begin(), scores1.end());//升序索引
	std::sort(scores2.begin(), scores2.end());
	//printf("sort complete\n");
	features myFeatures;
	if (scores1.size() == 0)
	{
		cout << "scores1 size is null, the result is incorrect" << endl;
		return myFeatures;
	}
	myFeatures.max1 = scores2[scores2.size() - 1];
	myFeatures.max2 = scores2[scores2.size() - 2];
	myFeatures.max3 = scores2[scores2.size() - 3];
	myFeatures.max4 = scores2[scores2.size() - 4];
	myFeatures.max5 = scores2[scores2.size() - 5];

	float sum = std::accumulate(std::begin(scores1), std::end(scores1), 0.0);

	float mean = sum / scores1.size();
	float accum = 0.0;
	std::for_each(std::begin(scores1), std::end(scores1), [&](const float d) {
		accum += (d - mean)*(d - mean);
	});
	accum = accum / scores1.size();
	accum = pow(accum, 0.5);

	myFeatures.m1_avg = mean;
	myFeatures.m1_std = accum;
	myFeatures.m1_median = scores1[scores1.size()/2];

	float num1 = 0, num2 = 0, num3 = 0, num4 = 0, num5 = 0;
	float sum1 = 0, sum2 = 0, sum3 = 0, sum4 = 0, sum5 = 0;
	float avg1 = 0, avg2 = 0, avg3 = 0, avg4 = 0, avg5 = 0;
	for (int i = 0; i < scores1.size(); i++){//对的是这三个阈值
		if (scores1[i] > 0.05){
			num1++;
			sum1 = sum1 + scores1[i];
		}
		if (scores1[i] > 0.5){
			num2++;
			sum2 = sum2 + scores1[i];
		}
		if (scores1[i] > 0.8){
			num3++;
			sum3 = sum3 + scores1[i];
		}
	}
	for (int i = 0; i < scores2.size(); i++){//对应的是剩下来的两个阈值
		if (scores2[i] > 0.9){
			num4++;
			sum4 = sum4 + scores2[i];
		}
		if (scores2[i] > 0.95){
			num5++;
			sum5 = sum5 + scores2[i];
		}
	}
	avg1 = (num1 == 0 ? 0 : sum1 / num1);
	avg2 = (num2 == 0 ? 0 : sum2 / num2);
	avg3 = (num3 == 0 ? 0 : sum3 / num3);
	avg4 = (num4 == 0 ? 0 : sum4 / num4);
	avg5 = (num5 == 0 ? 0 : sum5 / num5);

	myFeatures.m1_thresh1_avg = avg1;
	myFeatures.m1_thresh2_avg = avg2;
	myFeatures.m1_thresh3_avg = avg3;
	myFeatures.m2_thresh4_avg = avg4;
	myFeatures.m2_thresh5_avg = avg5;

	myFeatures.m1_thresh1_num = num1;
	myFeatures.m1_thresh2_num = num2;
	myFeatures.m1_thresh3_num = num3;
	myFeatures.m2_thresh4_num = num4;
	myFeatures.m2_thresh5_num = num5;

	myFeatures.m1_thresh1_sum = sum1;
	myFeatures.m1_thresh2_sum = sum2;
	myFeatures.m1_thresh3_sum = sum3;
	myFeatures.m2_thresh4_sum = sum4;
	myFeatures.m2_thresh5_sum = sum5;
	return myFeatures;
	//*predict = xgBoostPredict(&myFeatures);
}

void MyExport::ConvertResult(slideResult *result1, RegionResult *result2)
{
	result2->score1.insert(result2->score1.end(), result1->score1.begin(), result1->score1.end());//初始化score1
	//初始化point
	//result2->points.insert(result2->points.end(), result1->points.begin(), result1->points.end());
	for (int i = 0; i < result1->points.size(); i++){
		myPoint point;
		point.x = result1->points[i].x;
		point.y = result1->points[i].y;
		result2->points.push_back(point);
	}
	for (int i = 0; i < result1->score1.size(); i++){
		float maxValue = result1->score1[i];
		map<int, vector<float>>::iterator iter = result1->score2.find(i);
		if (iter != result1->score2.end()){
			if (iter->second.size() > 0){
				float maxScore2 = *max_element(iter->second.begin(), iter->second.end());
				maxValue = maxScore2;
			}		
		}
		result2->score2.push_back(maxValue);
	}
	//得到一个视野块的得分
	result2->score = *max_element(result2->score2.begin(), result2->score2.end());
}

void MyExport::getResult2(cv::Mat *inMat, slideResult *result)
{
	//集合之前所写的函数，很快就可以搞定
	map<int, modelResult2> modelResults;
	TestOneImg(&m_tfConfig1, inMat, &modelResults,"");
	//cout << "after TestOneImg" << endl;
	testImageOnModel2(&m_tfConfig2,inMat,&modelResults,"");
	//cout << "after testImageOnModel2" << endl;
	//对modelResult进行处理得到结果
	map<int, modelResult2>::iterator iter;
	for (iter = modelResults.begin(); iter != modelResults.end(); iter++){
		int i = iter->first;
		result->score1.push_back(iter->second.score1);
		//将score2也推入result中
		result->score2.insert(pair<int, vector<float>>(i, iter->second.score2));
		//result->score2.insert(result->score2.end(), iter->second.score2.begin(), iter->second.score2.end());
		result->points.insert(result->points.end(), iter->second.points.begin(), iter->second.points.end());
	}
}

void MyExport::getResult(cv::Mat *inMat, RegionResult *result)
{
	map<int, modelResult2> modelResults;
	TestOneImg(&m_tfConfig1, inMat, &modelResults, "");
	//cout << "after TestOneImg" << endl;
	testImageOnModel2(&m_tfConfig2, inMat, &modelResults, "");
	map<int, modelResult2>::iterator iter;
	for (iter = modelResults.begin(); iter != modelResults.end(); iter++)
	{
		int i = iter->first;
		result->score1.push_back(iter->second.score1);
		float maxValue = iter->second.score1;
		if (iter->second.score2.size() > 0)
		{
			maxValue = *max_element(iter->second.score2.begin(), iter->second.score2.end());
		}
		result->score2.push_back(maxValue);
		for (int j = 1; j < iter->second.points.size(); j++)
		{
			myPoint point;
			cv::Point cvPoint;
			cvPoint = iter->second.points[j] + m_myRects[i].tl();
			//cout << "cvPoint x:" << cvPoint.x << "cvPoint y:" << cvPoint.y << endl;
			point.x = cvPoint.x;
			point.y = cvPoint.y;
			result->points.push_back(point);
		}
		//result->points.insert(result->points.end(), iter->second.points.begin(), iter->second.points.end());
	}
	result->score = *max_element(result->score2.begin(), result->score2.end());
}

void MyExport::TestOneImg(Config *tfConfig,  cv::Mat *inMat, map<int, modelResult2> *modelResults,string imgPath)
{
	//先传入一张1216*1936的图像
	Mat rawImg;
	rawImg = *inMat;
	//cout << rawImg << endl;
	vector<Mat> resizedImg;
	vector<Rect>  myRects;

	//这是要保存的文件，已注释
	//string imgNameWithoutSuffix = getFileNamePrefix(&imgPath);
	//string savedPath = m_savePath + imgNameWithoutSuffix;
	//CreateDirectoryA(savedPath.c_str(), NULL);

	//myRects = getRects(MY_WIDTH, MY_HEIGHT, 512, 512, 3, 5);
	for (int i = 0; i < 15; i++) {
		Rect myRect = m_myRects[i];
		Mat tmp;
		//cout << "TestOneImg getSubImg" << endl;
		getSubImg(&rawImg, &tmp, myRect);
		resizedImg.push_back(tmp);
		//写入图片
		//imwrite(savedPath + "\\" + to_string(i) + ".tif", resizedImg[i]);
	}

	auto tensorInput = getTensorInput(tfConfig);
	//resOutPuts是一个batchsize*16*16*2的输出，将结果一个个的保存起来
	//偶数是输出的预测值，奇数是输出的mask
	for (int i = 0; i < resizedImg.size(); i++)
	{
		vector<Mat> tmpMatImg;
		tmpMatImg.push_back(resizedImg[i]);

		Status readTensorStatus = readTensorFromMat(tfConfig, &tmpMatImg, &tensorInput, 0);

		std::vector<tensorflow::Tensor> resOutputs;
		//printf("before enter Run\n");
		auto status_run = tfConfig->session->Run({ { tfConfig->opsInput,tensorInput } },
			tfConfig->opsOutput, {}, &resOutputs);
		//printf("after Run\n");
		Tensor predict = resOutputs[0];
		Tensor mask = resOutputs[1];
		modelResult2 modelResult_example;
		auto output_pre = predict.tensor<float, 2>();
		modelResult_example.score1 = float(output_pre(0, 0));
		modelResult_example.mask = mask;

		if (float(output_pre(0)) > 0.5) {
			auto output_c = mask.tensor<float, 4>();
			Mat maskMat;
			TensorToMat(mask, &maskMat);
			//采用getRegionPoint2
			modelResult_example.points = getRegionPoints2(&maskMat, 0.7);
			(*modelResults).insert(pair<int, modelResult2>(i, modelResult_example));

		}
		else {
			auto output_c = mask.tensor<float, 4>();
			Mat maskMat;
			TensorToMat(mask, &maskMat);
			//采用getRegionPoint2
			//对于小于0.5的，不予计算
			//modelResult_example.points = getRegionPoints2(&maskMat, 0.7);
			(*modelResults).insert(pair<int, modelResult2>(i, modelResult_example));
		}
	}
}

void MyExport::testImageOnModel2(Config* tfConfig,Mat *inMat,map<int, modelResult2> *modelResults, string path)
{
	Mat srcImg = *inMat;
	//cout << "testImageOnModel2" << endl;
	//保存一下传入model2里的图像
	//string fileNamePrefix = getFileNamePrefix(&path);

	//vector<Rect> myRects = getRects(MY_WIDTH, MY_HEIGHT, 512, 512, 3, 5);
	map<int, modelResult2>::iterator iter;
	for (iter = modelResults->begin(); iter != modelResults->end(); iter++) {
		int i = iter->first;
		modelResult2 result = (*modelResults)[i];

		//先将512*512中的坐标转为1216*1936中的坐标
		vector<cv::Point> points = result.points;
		for (int m = 1; m < points.size(); m++) {
			vector<Mat> tmpMatImg;
			cv::Point center = points[m] + m_myRects[i].tl();
			//先计算在1216*1936中裁掉的图
			int top = 0, bottom = 0, left = 0, right = 0;
			top = (center.y - 64) > 0 ? (center.y - 64) : 0;
			bottom = (center.y + 63) >= 1936 ? 1935 : (center.y + 63);
			left = (center.x - 64) > 0 ? (center.x - 64) : 0;
			right = (center.x + 63) >= 1216 ? 1215 : (center.x + 63);
			//这几个就是在原图中裁掉的四个点的坐标
			cv::Rect rectMat(left, top, right - left + 1, bottom - top + 1);
			Mat tmp;
			//cout << "testImageOnModel2 getSubImg" << endl;
			getSubImg(&srcImg, &tmp, rectMat);
			//计算需要贴在128*128白图上的坐标
			int topStick = (center.y - 64) > 0 ? 0 : abs(center.y - 64);
			int leftStick = (center.x - 64) > 0 ? 0 : abs(center.x - 64);
			cv::Mat WhiteMat(128, 128, CV_8UC3, Scalar(0, 0, 0));
			tmp.copyTo(WhiteMat(Rect(leftStick, topStick, tmp.cols, tmp.rows)));//得到图像
			imgResize(&WhiteMat, &WhiteMat, 256, 256);
			tmpMatImg.push_back(WhiteMat);
			//(*tfConfig)
			//将tmpMatImg转为tensor然后传入model2得到结果
			auto tensorInput = getTensorInput(tfConfig, tmpMatImg.size());//好像有bug，只有传入单个tensorInput才能得到结果的，传入多个tensorInput，也只会得到一个tensorOutput
			Status readTensorStatus = readTensorFromMat(tfConfig, &tmpMatImg, &tensorInput, 0);
			std::vector<tensorflow::Tensor> resOutputs;
			auto status_run = (*tfConfig).session->Run({ { (*tfConfig).opsInput,tensorInput } },
				(*tfConfig).opsOutput, {}, &resOutputs);
			//保存结果
			Tensor predict = resOutputs[0];
			auto output_pre = predict.tensor<float, 2>();
			float score = float(output_pre(0, 0));
			(*modelResults)[i].score2.push_back(score);

			//将image写入文件
			//string savePath = m_savePath + fileNamePrefix + "/model2";
			//CreateDirectoryA(savePath.c_str(), NULL);
			//savePath = savePath + "/" + to_string(i) + "_" + to_string(m) + ".tif";
			//cv::imwrite(savePath, tmpMatImg[0]);

		}

	}
}

void MyExport::saveModelResult(string savePath,map<int, modelResult2> *modelResults)
{
	TiXmlDocument *writeDoc = new TiXmlDocument;//xml文档指针												
	TiXmlDeclaration *decl = new TiXmlDeclaration("1.0", "UTF-8", "yes");//文档格式声明
	writeDoc->LinkEndChild(decl);//写入文档
	int n = (*modelResults).size();//父节点个数
	TiXmlElement *RootElement = new TiXmlElement("Root");//根元素
	RootElement->SetAttribute("num", n);//设置属性
	writeDoc->LinkEndChild(RootElement);//将根元素写入xml
	vector<Rect> myRects = getRects(MY_WIDTH, MY_HEIGHT, 512, 512, 3, 5);
	map<int, modelResult2>::iterator iter;
	for (iter = modelResults->begin(); iter != modelResults->end(); iter++)
	{
		int i = iter->first;
		modelResult2 result = (*modelResults)[i];
		TiXmlElement *subImg = new TiXmlElement("subImg");//建立一个元素
		subImg->SetAttribute("id", i);
		RootElement->LinkEndChild(subImg);//将subImg节点连接到根节点

		TiXmlElement *score = new TiXmlElement("score");//建立score节点
		subImg->LinkEndChild(score);
		string scoreStr = to_string(result.score1);
		TiXmlText *scoreText = new TiXmlText(scoreStr.c_str());
		score->LinkEndChild(scoreText);

		TiXmlElement *center = new TiXmlElement("centerPoint");
		subImg->LinkEndChild(center);
		string centerCoor = "";
		//保存regionPoint里的point
		for (int j = 1; j < result.points.size(); j++) {
			result.points[j] = result.points[j] + myRects[i].tl();
			centerCoor = centerCoor + " " + to_string(result.points[j].x) + "," + to_string(result.points[j].y);
		}
		TiXmlText *centerText = new TiXmlText(centerCoor.c_str());
		center->LinkEndChild(centerText);

		//建立mask_1节点，存储mask的第一通道
		TiXmlElement *mask_1 = new TiXmlElement("mask_1");
		subImg->LinkEndChild(mask_1);
		string mask1Str;
		TensorToString(result.mask, &mask1Str, 0);
		TiXmlText *mask1Text = new TiXmlText(mask1Str.c_str());
		mask_1->LinkEndChild(mask1Text);
		//建立mask_2节点，存储mask的第二通道
		TiXmlElement *mask_2 = new TiXmlElement("mask_2");
		subImg->LinkEndChild(mask_2);
		string mask2Str;
		TensorToString(result.mask, &mask2Str, 1);
		TiXmlText *mask2Text = new TiXmlText(mask2Str.c_str());
		mask_2->LinkEndChild(mask2Text);

		//建立score2节点，存储model2的结果
		TiXmlElement *score2 = new TiXmlElement("score2");
		subImg->LinkEndChild(score2);
		string score2Str = "";
		for (int j = 0; j < result.score2.size(); j++) {
			score2Str = score2Str + " " + to_string(result.score2[j]);
		}
		TiXmlText *score2Text = new TiXmlText(score2Str.c_str());
		score2->LinkEndChild(score2Text);
	}
	writeDoc->SaveFile(savePath.c_str());
	delete writeDoc;
}

void MyExport::initializeConfigMember()
{
	vector<string> opsOutput;
	opsOutput.push_back("dense_2/Sigmoid:0");
	opsOutput.push_back("conv2d_1/truediv:0");
	tfConfig(
		&m_tfConfig1, 512, 512, 3, 1, "input_1:0", opsOutput,
		m_model1Path
	);

	vector<string> opsOutput2;
	opsOutput2.push_back("dense_2/Sigmoid:0");
	tfConfig(
		&m_tfConfig2, 256, 256, 3, 1, "input_1:0", opsOutput2,
		m_model2Path
	);
}


