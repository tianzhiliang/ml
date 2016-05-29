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
//static const double g_PREDICT_THREAHOLD = 0.3;//������ȷ����ֵ,��Ӱ��ѵ��
static const double g_MIN_DOUBLE = 1e-10;
static const int g_MAX_CH_LINE = 2048;

class Stat;
class Logistic;

struct Conf{
    //�ļ�
    string trainFile;
    string predictFile;
    string modelFileTrainOutput;
    string modelFileToLoad;//ֻ��reload����ѵ�Ż���,���ܺ�modelFile����
    //string modelFilePost;//ֻ���ڲ���/Ԥ��׶�ָ��

    //ѵ������
    int iterNum;
    int workerNum;
    int featureSize;
    float alpha;
    float L2_regular;
    float randRangeW;
    float randRangeB;
    int randDivisor;
    
    //��Ҫ����
    float predictRightThres;

    //ģʽ
    int debug;
    int loadMode;//0:ѵ����������� 1:ѵ����������
    int txtMode;//0:ѵ��/����id���ļ� 1:ѵ��/����ԭʼ�ļ�(string) 2:ѵ��/���Զ������ļ�
    int predictMode;//0:�б�ǩ,ѵ�� 1:�б�ǩ,���� 2:û��ǩ,Ԥ��
    int isReLoadModel;//0:��ʹ������ģ�� 1:��������ģ��,����ѵ,ʹ������ģ�͵Ĳ��� 2:��������ģ��,����ѵ,ʹ��conf�ļ��Ĳ���

};

struct Model{
    /*int featureSize;
    double* weight;
    double bias;*/
    Logistic* lr;
};

struct Sample{
    string* IdStr;
    int* Id;//��Ŵ�0��ʼ
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

//�������ڲ��ӿ���ṹ,�û��������

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
    string m_mark;//ͳ�Ƶ�����
    
    //����
    int initStat(string mark);//??����
    int addOneSample(Logistic* lr,Sample* sample);
    long long getRightCount();
    long long getWrongCount();
    double getAcc();
    int collectStatAndPrint(Logistic* lr);
};


class Logistic{
    public:
    //�ļ�
    string m_trainFile;
    string m_predictFile;
    string m_modelFileTrainOutput;
    //string m_modelFilePost;
    string m_modelFileToLoad;

    //ѵ������
    int m_iterNum;
    int m_workerNum;
    int m_featureSize;
    int m_randDivisor;
    double m_alpha;
    double m_L2_regular;
    double m_randRangeW;
    double m_randRangeB;

    //��Ҫ����
    double m_predictRightThres;//������ȷ����ֵ,��Ӱ��ѵ��

    //ģʽ
    int m_debug;
    int m_loadMode;//0:ѵ����������� 1:ѵ����������
    int m_txtMode;//0:ѵ��/����id���ļ� 1:ѵ��/����ԭʼ�ļ�(string) 2:ѵ��/���Զ������ļ�
    int m_predictMode;//0:�б�ǩ,ѵ�� 1:�б�ǩ,���� 2:û��ǩ,Ԥ��
    //int m_labelMode;//0:�б�ǩ,���� 1:û��ǩ,Ԥ��
    int m_isReLoadModel;//0:��ʹ������ģ�� 1:��������ģ��,����ѵ,ʹ������ģ�͵Ĳ��� 2:��������ģ��,����ѵ,ʹ��conf�ļ��Ĳ���
        

    //ģ�����
    double* m_weight;
    double m_bias;
    double* m_feature;//һ��ֻ��int
    double m_WX_b;
    double m_biasError;
    double* m_weightError;
    //double m_LROutput;

    //��������
    Sample* m_sample;//��ʱ������
    long long m_lineCount;
    
    //ͳ�����
    Stat* m_stat_iter;
    Stat* m_stat_global;

    //��ʱ����
    string* m_lineBuffer;
    //int m_saveModelIterByhand;
    bool m_isSaveModel;
    //bool m_isTrain;

    //����

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

//����
bool doubleEqual(double a,double b);
int mySplit(string str_dest,char demli,string* str_res);
int setInitDouble(double* array,int size,double value);
int setInitInt(int* array,int size,int value);
int randInit(double* array,int size,double min,double max,int factorAsDivisor);
int vectorXvector(double* a,double* b,int dim,double& res);

//����
Logistic* initLogistic(string confPath,string confFile,int predictMode);
void freeLogistic(Logistic* lr);
int readConf(string confPath,string confFile,Conf* conf);
int loadConfFile(map<string,string> &argMap,string confName);

#endif  //__LOGISTICMODEL_H_

/* vim: set expandtab ts=4 sw=4 sts=4 tw=100: */
