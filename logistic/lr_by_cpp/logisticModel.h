/***************************************************************************
 * 
 * Copyright (c) 2014 Baidu.com, Inc. All Rights Reserved
 * 
 **************************************************************************/
 
 
 
/**
 * @file logisticModel.h
 * @author tianzhiliang(com@baidu.com)
 * @date 2014/11/14 20:07:12
 * @brief 
 *  
 **/




#ifndef  __LOGISTICMODEL_H_
#define  __LOGISTICMODEL_H_

#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <stdint.h>
#include <string>
#include <cstring>
#include <ext/hash_map>
#include <sys/time.h>
#include <cmath>
#include <map>
#include <time.h>

using namespace std;
using namespace __gnu_cxx;

static const double g_MAX_EXP = 30;
static const char g_DELIM_FIRST = '\t';
static const char g_DELIM_SECOND = ':';
static const char g_DELIM_CONFFILE = ' ';
static const int g_MAX_CELL_NUM = 10;
//static const double g_PREDICT_THREAHOLD = 0.3;//算作正确的阈值,不影响训练
static const double g_MIN_DOUBLE = 1e-10;
static const int g_MAX_CH_LINE = 2048;

class Stat;
class Logistic;

struct Conf{
    //文件
    string trainFile;
    string predictFile;
    string modelFileTrainOutput;
    string modelFileToLoad;//只有reload继续训才会用,不能和modelFile重名
    //string modelFilePost;//只需在测试/预测阶段指定

    //训练参数
    int iterNum;
    int workerNum;
    int featureSize;
    float alpha;
    float L2_regular;
    float randRangeW;
    float randRangeB;
    int randDivisor;
    
    //次要参数
    float predictRightThres;

    //模式
    int debug;
    int loadMode;//0:训练产物二进制 1:训练产物明文
    int txtMode;//0:训练/测试id化文件 1:训练/测试原始文件(string) 2:训练/测试二进制文件
    int predictMode;//0:有标签,训练 1:有标签,测试 2:没标签,预测
    int isReLoadModel;//0:不使用现有模型 1:读入已有模型,继续训,使用已有模型的参数 2:读入已有模型,继续训,使用conf文件的参数

};

struct Model{
    /*int featureSize;
    double* weight;
    double bias;*/
    Logistic* lr;
};

struct Sample{
    string* IdStr;
    int* Id;//标号从0开始
    double* value;
    int label;
    double output;
    int result;
    int featureNum;
    double error;
    bool isRight;
};

Model* loadModel(string confPath,string confFile);

void freeModel(Model* model);

int trainModel(string confPath,string confFile);

int predict(Model* model,Sample* sample);

int predictOneFile(Model* model);

//以下是内部接口与结构,用户无需关心

class Timer{
    public:
    int printNowTime();
    int printNowTimeLight();

    private:
    struct tm* m_nowTime;
};


class Stat{
    public:

    long long m_totalCount;
    long long m_posResCount;
    long long m_negResCount;
    long long m_posLabelCount;
    long long m_negLabelCount;
    long long m_p2p;
    long long m_p2n;
    long long m_n2n;
    long long m_n2p;
    
    int m_nowIter;
    string m_mark;//统计的类型
    
    //函数
    int initStat(string mark);//??重名
    int addOneSample(Logistic* lr,Sample* sample);
    long long getRightCount();
    long long getWrongCount();
    double getAcc();
    int collectStatAndPrint(Logistic* lr);
};


class Logistic{
    public:
    //文件
    string m_trainFile;
    string m_predictFile;
    string m_modelFileTrainOutput;
    //string m_modelFilePost;
    string m_modelFileToLoad;

    //训练参数
    int m_iterNum;
    int m_workerNum;
    int m_featureSize;
    int m_randDivisor;
    double m_alpha;
    double m_L2_regular;
    double m_randRangeW;
    double m_randRangeB;

    //次要参数
    double m_predictRightThres;//算作正确的阈值,不影响训练

    //模式
    int m_debug;
    int m_loadMode;//0:训练产物二进制 1:训练产物明文
    int m_txtMode;//0:训练/测试id化文件 1:训练/测试原始文件(string) 2:训练/测试二进制文件
    int m_predictMode;//0:有标签,训练 1:有标签,测试 2:没标签,预测
    //int m_labelMode;//0:有标签,测试 1:没标签,预测
    int m_isReLoadModel;//0:不使用现有模型 1:读入已有模型,继续训,使用已有模型的参数 2:读入已有模型,继续训,使用conf文件的参数
        

    //模型相关
    double* m_weight;
    double m_bias;
    double* m_feature;//一般只用int
    double m_WX_b;
    double m_biasError;
    double* m_weightError;
    //double m_LROutput;

    //样本级别
    Sample* m_sample;//临时存样本
    long long m_lineCount;
    
    //统计相关
    Stat* m_stat_iter;
    Stat* m_stat_global;

    //临时变量
    string* m_lineBuffer;
    //int m_saveModelIterByhand;
    bool m_isSaveModel;
    //bool m_isTrain;

    //函数

    int saveModel();
    int initNetPara();
    int mallocSample(Logistic* lr,Sample* sample);
    void freeSample(Sample* sample);
    int copyConfValue(Conf* conf);
    int logistic(double in,double& res);
    int openOneFile(string file,ifstream& input);
    int closeOneFile(ifstream& input);
    int readOneSample(ifstream& input);
    int feedForward();
    int transSampleToInput();
    int predict();
    int getGradient();
    int backPropagate();
    int trainOneSample(ifstream& input);
    void setNowIter(int iter);
    int getNowIter();
    int reLoadModel(Conf* conf);
    int readModel(string file);
    double getL2Norm();
};

//工具
bool doubleEqual(double a,double b);
int mySplit(string str_dest,char demli,string* str_res);
int setInitDouble(double* array,int size,double value);
int setInitInt(int* array,int size,int value);
int randInit(double* array,int size,double min,double max,int factorAsDivisor);
int vectorXvector(double* a,double* b,int dim,double& res);

//公共
Logistic* initLogistic(string confPath,string confFile,int predictMode);
void freeLogistic(Logistic* lr);
int readConf(string confPath,string confFile,Conf* conf);
int loadConfFile(map<string,string> &argMap,string confName);

#endif  //__LOGISTICMODEL_H_

/* vim: set expandtab ts=4 sw=4 sts=4 tw=100: */
