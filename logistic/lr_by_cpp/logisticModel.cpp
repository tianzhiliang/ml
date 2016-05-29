/***************************************************************************
 * 
 * Copyright (c) 2014 Baidu.com, Inc. All Rights Reserved
 * 
 **************************************************************************/



/**
 * @file logisticModel.cpp
 * @author tianzhiliang(com@baidu.com)
 * @date 2014/11/14 19:59:23
 * @brief 
 *  
 **/

#include <logisticModel.h>

bool doubleEqual(double a,double b){
    if(abs(a-b)<g_MIN_DOUBLE){
        return true;
    }
    return false;
}

int mySplit(string str_dest,char demli,string* str_res){
    int len = str_dest.length();
    if(0 == len){
        return 0;
    }
    int i = 0;
    int last = 0;
    int index = 0;
    while(i < len){
        if(str_dest[i] == demli){
            str_res[index] = str_dest.substr(last,i-last);
            index ++;
            last = i + 1;//len of demli is 1
        }
        i ++;
    }

    if(0==i-last){//如果最后一个间隔符后面没有值,返回
        return index;
    }

    //如果最后一个间隔符后面有值,存
    str_res[index] = str_dest.substr(last,i-last);
    return index+1;//return size
}

int setInitDouble(double* array,int size,double value){
    for(int i=0;i<size;i++){
        array[i]=value;
    }
    return 0;
}

int setInitInt(int* array,int size,int value){
    for(int i=0;i<size;i++){
        array[i]=value;
    }
    return 0;
}

int printDoubleArray(double* array,int size){
    for(int i=0;i<size;i++){
        cerr<<i<<":"<<array[i]<<"\t";
    }
    cerr<<endl;
    return 0;
}

int randInit(double* array,int size,double min,double max,int factorAsDivisor){
    if(max<min){
        double tmp = max;
        max = min;
        min = tmp;
    }

    for(int i=0;i<size;i++){
        array[i] = (((double)rand()/RAND_MAX)*(max-min) + min) / factorAsDivisor;
    }
    return 0;
}

int vectorXvector(double* a,double* b,int dim,double& res){
    res=0;
    for(int i=0;i<dim;i++){
        res+=a[i]*b[i];
    }
    return 0;
}

int Timer::printNowTime(){
    time_t now = time(0);
    m_nowTime = localtime(&now);
    fprintf(stderr,"%d/%d_%d:%d:%d\n",m_nowTime->tm_mon+1,m_nowTime->tm_mday,\
            m_nowTime->tm_hour,m_nowTime->tm_min,m_nowTime->tm_sec);
    return 0;
}

int Timer::printNowTimeLight(){
    time_t now = time(0);
    m_nowTime = localtime(&now);
    fprintf(stderr,"%d:%d:%d\n",m_nowTime->tm_hour,m_nowTime->tm_min,m_nowTime->tm_sec);
    return 0;
}


int Logistic::copyConfValue(Conf* conf){
    if(0!=m_predictMode){
        m_predictFile =  conf->predictFile;
        m_modelFileToLoad = conf->modelFileToLoad;
        m_predictMode = conf->predictMode;
    }
    else{
        m_trainFile = conf->trainFile;
        m_iterNum = conf->iterNum;
        m_workerNum = conf->workerNum;
        m_featureSize = conf->featureSize;
        m_alpha = conf->alpha;
        m_L2_regular = conf->L2_regular;
        m_randRangeW = conf->randRangeW;
        m_randRangeB = conf->randRangeB;
        m_randDivisor = conf->randDivisor;

        m_loadMode = conf->loadMode;
        m_txtMode = conf->txtMode;
        m_isReLoadModel = conf->isReLoadModel;
        m_modelFileTrainOutput = conf->modelFileTrainOutput;
    }

    m_predictRightThres = conf->predictRightThres;
    m_debug = conf->debug;

    return 0;
}

int Logistic::logistic(double in,double& res){
    if(in>g_MAX_EXP){
        res=1;
        return 0;
    }

    if(in<0-g_MAX_EXP){
        res=0;
        return 0;
    }

    res=1/(1+exp(0-in));
    return 0;
}

int loadConfFile(map<string,string> &argMap,string confName){
    FILE *fin=fopen(confName.c_str(),"r");
    if(NULL==fin){
        cerr<<"ERROR:fopen failed,file:"<<confName<<endl;
        return -1;
    }

    char line[g_MAX_CH_LINE];

    string wordlist[g_MAX_CELL_NUM];
    int size=0,count=0;

    while(fgets(line,g_MAX_CH_LINE,fin)){
        size=mySplit(string(line),g_DELIM_CONFFILE,wordlist);
        if(2>size){
            continue;
        }
        else if (2<size){
            cerr<<"ERROR:2<size in loadConfFile,line:"<<count<<"\tsize:"<<size<<"\tline:"<<line<<endl;
            return -1;
        }
        count++;

        if('\n'==wordlist[1].at(wordlist[1].size()-1)){
            wordlist[1] = wordlist[1].substr(0,wordlist[1].size()-1);
        }
        argMap[wordlist[0]]=wordlist[1];
    }

    fclose(fin);
    fin=NULL;

    return 0;
}

int readConf(string confPath,string confFile,Conf* conf){
    map<string,string> confMap;
    string confName = confPath + "/" + confFile;
    cerr<<"will readConf,file:"<<confName<<endl;

    if(-1==loadConfFile(confMap,confName)){
        return -1;
    }

    //文件
    if(confMap.find("trainFile")==confMap.end()){
        cerr<<"WARNING: trainFile not in conf"<<endl;
    }
    else{
        conf->trainFile=confMap["trainFile"];
        cerr<<"trainFile:"<<conf->trainFile<<endl;
    }

    if(confMap.find("predictFile")==confMap.end()){
        cerr<<"WARNING: predictFile not in conf"<<endl;
    }
    else{
        conf->predictFile=confMap["predictFile"];
        cerr<<"predictFile:"<<conf->predictFile<<endl;
    }

    if(confMap.find("modelFileTrainOutput")==confMap.end()){
        cerr<<"WARNING: modelFileTrainOutput not in conf"<<endl;
    }
    else{
        conf->modelFileTrainOutput=confMap["modelFileTrainOutput"];
        cerr<<"modelFileTrainOutput:"<<conf->modelFileTrainOutput<<endl;
    }

    if(confMap.find("modelFileToLoad")==confMap.end()){
        cerr<<"WARNING: modelFileToLoad not in conf"<<endl;
    }
    else{
        conf->modelFileToLoad=confMap["modelFileToLoad"];
        cerr<<"modelFileToLoad:"<<conf->modelFileToLoad<<endl;
    }

    //if(confMap.find("modelFilePost")==confMap.end()){
    //    cerr<<"WARNING: modelFilePost not in conf"<<endl;
    //}
    //else{
    //    conf->modelFilePost=confMap["modelFilePost"];
    //    cerr<<"modelFilePost:"<<conf->modelFilePost<<endl;
    //}


    //训练参数
    if(confMap.find("iterNum")==confMap.end()){
        cerr<<"WARNING: iterNum not in conf"<<endl;
    }
    else{
        conf->iterNum=atoi(confMap["iterNum"].c_str());
        cerr<<"iterNum:"<<conf->iterNum<<endl;
    }

    if(confMap.find("workerNum")==confMap.end()){
        cerr<<"WARNING: workerNum not in conf"<<endl;
    }
    else{
        conf->workerNum=atoi(confMap["workerNum"].c_str());
        cerr<<"workerNum:"<<conf->workerNum<<endl;
    }

    if(confMap.find("featureSize")==confMap.end()){
        cerr<<"WARNING: featureSize not in conf"<<endl;
    }
    else{
        conf->featureSize=atoi(confMap["featureSize"].c_str());
        cerr<<"featureSize:"<<conf->featureSize<<endl;
    }

    if(confMap.find("alpha")==confMap.end()){
        cerr<<"WARNING: alpha not in conf"<<endl;
    }
    else{
        conf->alpha=atof(confMap["alpha"].c_str());
        cerr<<"alpha:"<<conf->alpha<<endl;
    }

    if(confMap.find("L2_regular")==confMap.end()){
        cerr<<"WARNING: L2_regular not in conf"<<endl;
    }
    else{
        conf->L2_regular=atof(confMap["L2_regular"].c_str());
        cerr<<"L2_regular:"<<conf->L2_regular<<endl;
    }

    if(confMap.find("randRangeW")==confMap.end()){
        cerr<<"WARNING: randRangeW not in conf"<<endl;
    }
    else{
        conf->randRangeW=atof(confMap["randRangeW"].c_str());
        cerr<<"randRangeW:"<<conf->randRangeW<<endl;
    }

    if(confMap.find("randRangeB")==confMap.end()){
        cerr<<"WARNING: randRangeB not in conf"<<endl;
    }
    else{
        conf->randRangeB=atof(confMap["randRangeB"].c_str());
        cerr<<"randRangeB:"<<conf->randRangeB<<endl;
    }

    if(confMap.find("predictRightThres")==confMap.end()){
        cerr<<"WARNING: predictRightThres not in conf"<<endl;
    }
    else{
        conf->predictRightThres=atof(confMap["predictRightThres"].c_str());
        cerr<<"predictRightThres:"<<conf->predictRightThres<<endl;
    }

    if(confMap.find("randDivisor")==confMap.end()){
        cerr<<"WARNING: randDivisor not in conf"<<endl;
    }
    else{
        conf->randDivisor=atoi(confMap["randDivisor"].c_str());
        cerr<<"randDivisor:"<<conf->randDivisor<<endl;
    }



    //模式
    if(confMap.find("debug")==confMap.end()){
        cerr<<"WARNING: debug not in conf"<<endl;
    }
    else{
        conf->debug=atoi(confMap["debug"].c_str());
        cerr<<"debug:"<<conf->debug<<endl;
    }

    if(confMap.find("loadMode")==confMap.end()){
        cerr<<"WARNING: loadMode not in conf"<<endl;
    }
    else{
        conf->loadMode=atoi(confMap["loadMode"].c_str());
        cerr<<"loadMode:"<<conf->loadMode<<endl;
    }

    if(confMap.find("txtMode")==confMap.end()){
        cerr<<"WARNING: txtMode not in conf"<<endl;
    }
    else{
        conf->txtMode=atoi(confMap["txtMode"].c_str());
        cerr<<"txtMode:"<<conf->txtMode<<endl;
    }

    if(confMap.find("predictMode")==confMap.end()){
        cerr<<"WARNING: predictMode not in conf"<<endl;
    }
    else{
        conf->predictMode=atoi(confMap["predictMode"].c_str());
        cerr<<"predictMode:"<<conf->predictMode<<endl;
    }

    if(confMap.find("isReLoadModel")==confMap.end()){
        cerr<<"WARNING: isReLoadModel not in conf"<<endl;
    }
    else{
        conf->isReLoadModel=atoi(confMap["isReLoadModel"].c_str());
        cerr<<"isReLoadModel:"<<conf->isReLoadModel<<endl;
    }

    cerr<<"readConf done"<<endl;
    return 0;
}

int Logistic::openOneFile(string file,ifstream& input){
    input.open(file.c_str());

    if(input.fail()){
        cerr<<"ERROR:input.fail() in openTrainFile,file:"<<file<<endl;
        return -1;
    }

    m_lineCount=0;
    return 0;
}

int Logistic::closeOneFile(ifstream& input){
    input.close();
    return 0;
}

int Logistic::readOneSample(ifstream& input){
    string lineStr;
    string cell[g_MAX_CELL_NUM];
    int size=0,index=0,cellsize=0;

    if(0==m_txtMode){
        if(getline(input,lineStr).eof()){
            return -1;
        }

        size = mySplit(lineStr,g_DELIM_FIRST,m_lineBuffer);

        if((size<1 && 1>=m_predictMode) || size<0){
            cerr<<"ERROR:(size<1 && ) || size<0 in readOneSample,line:"<<m_lineCount<<endl;
            return -2;
        }
        if(0==size && 2==m_predictMode){
            cerr<<"WARING:0==size && 1==mode in readOneSample,line:"<<m_lineCount<<endl;
        }

        if(1>=m_predictMode){
            m_sample->label = atoi(m_lineBuffer[index].c_str());
            if(1!=m_sample->label && 0!=m_sample->label){
                cerr<<"ERROR:labelInput label MUST be 0/1. 1!=m_sample-0>label && 0!=m_sample,line:"\
                    <<m_lineCount<<endl;
                return -2;
            }
            index++;
        }

        m_sample->featureNum = size-index;
        if(m_sample->featureNum>m_featureSize){
            cerr<<"ERROR,sample.featureNum>m_featureSize,featureNum:"<<\
                m_sample->featureNum<<"\tm_featureSize:"<<m_featureSize<<endl;
            return -2;
        }

        for(int i=0;i<m_sample->featureNum;i++){
            cellsize=mySplit(m_lineBuffer[index],g_DELIM_SECOND,cell);
            if(2!=cellsize){
                cerr<<"ERROR:2!=cellsize in readOneSample,line:"<<m_lineCount<<"\tindex:"<<index<<endl;
                return -2;
            }
            index++;

            m_sample->Id[i]=atoi(cell[0].c_str());
            m_sample->value[i]=atof(cell[1].c_str());

            //cerr<<"set id:"<<m_sample->Id[i]<<"\tvalue:"<<m_sample->value[i]<<endl;
        }
    }

    m_lineCount++;
    return 0;
}

int Logistic::feedForward(){
    double WX=0;

    transSampleToInput();

    vectorXvector(m_feature,m_weight,m_featureSize,WX);
    m_WX_b = WX + m_bias;

    logistic(m_WX_b,m_sample->output);

    return 0;
}

int Logistic::transSampleToInput(){//不会校验特征是否重复,重复的话直接使用后面的值
    setInitDouble(m_feature,m_featureSize,0);

    for(int i=0;i<m_sample->featureNum;i++){
        //cerr<<"id:"<<m_sample->Id[i]<<"\tvalue:"<<m_sample->value[i]<<endl;
        m_feature[m_sample->Id[i]] = m_sample->value[i];
    }

    return 0;
}

double Logistic::getL2Norm(){
    double res = 0;
    for(int i=0;i<m_featureSize;i++){
        res += m_weight[i]*m_weight[i];
    }
    res += m_bias* m_bias;//???
    //cerr<<"L2Norm:sum:"<<res;
    //res = sqrt(res);
    //cerr<<"\tres:"<<res<<endl;
    return res;
}

int Logistic::predict(){
    double L2norm = getL2Norm();
    m_sample->error = m_sample->label - m_sample->output;
    if(0>m_sample->error){
        L2norm=0-L2norm;
    }
    m_sample->error += m_L2_regular * L2norm;

    if(0==m_debug%3){
    cerr<<"labelError:"<<m_sample->label - m_sample->output<<"+L2Norm:"<<L2norm<<"*m_L2_regular:"<<m_L2_regular<<"=finalError:"<<m_sample->error<<endl;
    }

    if(1==m_sample->label){
        if(m_sample->output > 1-m_predictRightThres){
            m_sample->result = 1;
            m_sample->isRight = true;
        }
        else{
            m_sample->result = 0;
            m_sample->isRight = false;
        }
    }
    else if(0==m_sample->label){
        if(m_sample->output < m_predictRightThres){
            m_sample->result = 0;
            m_sample->isRight = true;
        }
        else{
            m_sample->result = 1;
            m_sample->isRight = false;
        }
    }

    if(0==m_debug%2 || 0==m_debug%3){
        if(m_sample->isRight){
            cerr<<"m_WX_b:"<<m_WX_b<<"\toutput:"<<m_sample->output<<"\tresult:"<<m_sample->result<<"\tlabel:"<<m_sample->label<<"\terror:"<<m_sample->error<<"\tRIGHT"<<endl;
        }
        else{
            cerr<<"m_WX_b:"<<m_WX_b<<"\toutput:"<<m_sample->output<<"\tresult:"<<m_sample->result<<"\tlabel:"<<m_sample->label<<"\terror:"<<m_sample->error<<"\tWRONG"<<endl;
        }
    }

    m_stat_iter->addOneSample(this,m_sample);
    m_stat_global->addOneSample(this,m_sample);

    if(0==m_debug%4){
    cerr<<"DEBUG,line:"<<m_lineCount<<"\t";
    printDoubleArray(m_feature,m_featureSize);
    }

    return 0;
}

int Logistic::getGradient(){
    //setInitDouble(m_weightError,m_featureSize,0);

    for(int i=0;i<m_featureSize;i++){
        m_weightError[i] = m_sample->error * m_feature[i];
    }

    m_biasError = m_sample->error;
    return 0;
}

int Logistic::backPropagate(){
    getGradient();

    for(int i=0;i<m_featureSize;i++){
        m_weight[i] += m_alpha * m_weightError[i];
    }

    m_bias += m_alpha * m_biasError;

    return 0;
}

int Stat::initStat(string mark){
    m_totalCount = 0;
    m_posResCount = 0;
    m_negResCount = 0;
    m_posLabelCount = 0;
    m_negLabelCount = 0;
    m_p2p = 0;
    m_p2n = 0;
    m_n2n = 0;
    m_n2p = 0;

    m_mark = mark;
    return 0;
}

int Stat::addOneSample(Logistic* lr,Sample* sample){
    m_totalCount++;

    if(0==sample->result){
        m_posResCount++;
    }
    if(1==sample->result){
        m_negResCount++;
    }
    if(2==lr->m_predictMode){
        return 0;
    }

    if(0==sample->label){
        m_negLabelCount++;
        if(0==sample->result){
            m_n2n++;
        }
        if(1==sample->result){
            m_n2p++;
        }
    }
    if(1==sample->label){
        m_posLabelCount++;
        if(0==sample->result){
            m_p2n++;
        }
        if(1==sample->result){
            m_p2p++;
        }
    }

    return 0;
}

long long Stat::getRightCount(){
    return m_p2p + m_n2n;
}

long long Stat::getWrongCount(){
    return m_p2n + m_n2p;
}

double Stat::getAcc(){
    long long right = getRightCount();
    long long wrong = getWrongCount();
    if(m_totalCount != right+wrong){
        cerr<<"ERROR:m_totalCount != right+wrong,m_totalCount:"<<m_totalCount<<"\tright:"<<right<<"\twrong"<<wrong<<endl;
        return -1;
    }
    return (double)right/m_totalCount;
}

int Logistic::getNowIter(){
    return m_stat_global->m_nowIter;
}

void Logistic::setNowIter(int iter){
    m_stat_global->m_nowIter = iter;
}

int Stat::collectStatAndPrint(Logistic* lr){
    Timer time;

    if(1>=lr->m_predictMode){
        if(0==lr->m_debug%3){
            fprintf(stderr,"%s Count: %ld Acc: %f Right: %ld posRes: %ld posLabel: %ld p2p: %ld p2n: %ld n2n: %ld n2p: %ld line: %ld\t",\
                    m_mark.c_str(),m_totalCount,getAcc(),getRightCount(),m_p2p+m_n2p,m_posLabelCount,m_p2p,\
                    m_p2n,m_n2n,m_n2p,lr->m_lineCount);
            time.printNowTime();
        }
        else if(0==lr->m_debug%2){
            fprintf(stderr,"%s Count: %ld Acc: %f p2p: %ld p2n: %ld n2n: %ld n2p: %ld\t",\
                    m_mark.c_str(),m_totalCount,getAcc(),m_p2p,m_p2n,m_n2n,m_n2p);
            time.printNowTimeLight();
        }
    }
    else if(2==lr->m_predictMode){
        fprintf(stderr,"%s Count: %ld posRes: %ld negRes: %ld\t",m_mark.c_str(),m_totalCount,m_posResCount,m_negResCount);
        time.printNowTime();
    }

    return 0;
}

int Logistic::trainOneSample(ifstream& input){
    int ret = 0;
    ret=readOneSample(input);

    if(0!=ret){
        return ret;
    }

    feedForward();

    predict();

    backPropagate();

    return 0;
}

Logistic* initLogistic(string confPath,string confFile,int predictMode){
    Logistic* lr;
    try{
        lr = new Logistic;

        lr->m_sample = new Sample;
        lr->m_stat_global = new Stat;
        lr->m_stat_iter = new Stat;
    }
    catch(bad_alloc &info){
        cerr<<info.what()<<endl;
        cerr<<"FATAL:initLogistic failed"<<endl;
        return NULL;
    }

    lr->m_predictMode=predictMode;

    Conf conf;
    if(-1==readConf(confPath,confFile,&conf)){
        return NULL;
    }

    lr->copyConfValue(&conf);


    lr->reLoadModel(&conf);

    if(-1==lr->mallocSample(lr,lr->m_sample)){
        return NULL;
    }

    if(0==lr->m_isReLoadModel && 0==lr->m_predictMode){
        lr->m_weight=(double*)calloc(lr->m_featureSize,sizeof(double));
    }
    lr->m_weightError=(double*)calloc(lr->m_featureSize,sizeof(double));
    lr->m_feature=(double*)calloc(lr->m_featureSize,sizeof(double));

    if(NULL==lr->m_weight || NULL==lr->m_weightError || NULL==lr->m_feature){
        cerr<<"FATAL:initLogistic failed"<<endl;
        return NULL;
    }

    try{
        lr->m_lineBuffer = new string[lr->m_featureSize+10];
    }
    catch(bad_alloc &info){
        cerr<<info.what()<<endl;
        cerr<<"FATAL:initLogistic failed"<<endl;
        return NULL;
    }

    if(0==lr->m_isReLoadModel && 0==lr->m_predictMode){
        lr->initNetPara();
    }

    lr->m_stat_global->initStat("GLOBAL");

    return lr;
}

void freeLogistic(Logistic* lr){
    free(lr->m_weight);
    free(lr->m_weightError);
    free(lr->m_feature);

    delete []lr->m_lineBuffer;

    delete lr->m_stat_global;
    delete lr->m_stat_iter;

    lr->freeSample(lr->m_sample);

    delete lr;
}

int Logistic::mallocSample(Logistic* lr,Sample* sample){
    try{
        sample->IdStr = new string[lr->m_featureSize];
    }
    catch(bad_alloc &info){
        cerr<<info.what()<<endl;
        cerr<<"FATAL:mallocSample failed"<<endl;
        return -1;
    }

    sample->Id = (int*)calloc(lr->m_featureSize,sizeof(int));
    sample->value = (double*)calloc(lr->m_featureSize,sizeof(double));

    if(NULL==sample->Id || NULL==sample->value){
        cerr<<"FATAL:NULL==sample->Id || NULL==sample->value,mallocSample failed"<<endl;
        return -1;
    }
    return 0;
}

void Logistic::freeSample(Sample* sample){
    delete []sample->IdStr;
    free(sample->Id);
    free(sample->value);
}

int Logistic::initNetPara(){
    randInit(m_weight,m_featureSize,0-m_randRangeW,m_randRangeW,m_randDivisor);

    randInit(&m_bias,1,0-m_randRangeB,m_randRangeB,m_randDivisor);
    return 0;
}

int Logistic::saveModel(){
    char file[g_MAX_CH_LINE];

    if(true==m_isSaveModel){
        static int saveModelIterByhand = 0;
        sprintf(file,"%s.hand%d",m_modelFileTrainOutput.c_str(),saveModelIterByhand);
        saveModelIterByhand++;
    }
    else{
        sprintf(file,"%s.iter%d",m_modelFileTrainOutput.c_str(),getNowIter());
    }

    ofstream output(file,ios::binary|ios::out);

    output.write((char*)&m_iterNum,sizeof(int));
    output.write((char*)&m_workerNum,sizeof(int));
    output.write((char*)&m_featureSize,sizeof(int));
    output.write((char*)&m_randDivisor,sizeof(int));

    output.write((char*)&m_alpha,sizeof(double));
    output.write((char*)&m_L2_regular,sizeof(double));
    output.write((char*)&m_randRangeW,sizeof(double));
    output.write((char*)&m_randRangeB,sizeof(double));

    output.write((char*)m_weight,sizeof(double)*m_featureSize);
    output.write((char*)&m_bias,sizeof(double));

    output.close();

    return 0;
}

int Logistic::readModel(string file){
    ifstream input(file.c_str(),ios::binary|ios::in);

    input.read((char*)&m_iterNum,sizeof(int));
    input.read((char*)&m_workerNum,sizeof(int));
    input.read((char*)&m_featureSize,sizeof(int));
    input.read((char*)&m_randDivisor,sizeof(int));

    input.read((char*)&m_alpha,sizeof(double));
    input.read((char*)&m_L2_regular,sizeof(double));
    input.read((char*)&m_randRangeW,sizeof(double));
    input.read((char*)&m_randRangeB,sizeof(double));

    m_weight=(double*)calloc(m_featureSize,sizeof(double));

    input.read((char*)m_weight,sizeof(double)*m_featureSize);
    input.read((char*)&m_bias,sizeof(double));

    //todo:print

    input.close();
    return 0;
}

int Logistic::reLoadModel(Conf* conf){
    if(0==m_isReLoadModel && 0==m_predictMode){
        return -1;
    }

    readModel(m_modelFileToLoad);

    if(2==m_isReLoadModel && 0==m_predictMode){
        m_iterNum = conf->iterNum;
        m_workerNum = conf->workerNum;
        //m_featureSize = conf->featureSize;
        m_alpha = conf->alpha;
        m_L2_regular = conf->L2_regular;
        m_randRangeW = conf->randRangeW;
        m_randRangeB = conf->randRangeB;
        m_randDivisor = conf->randDivisor;
    }

    //todo:print

    return 0;
}

int trainModel(string confPath,string confFile){
    Timer time;
    int ret = 0;
    cerr<<"Logistic Regression Train Started.Have a good trip!"<<endl;
    time.printNowTime();

    Logistic* lr = initLogistic(confPath,confFile,0);
    if(NULL==lr){
        return -1;
    }

    cerr<<"initLogistic done"<<endl;
    time.printNowTime();
    for(int i=0;i<lr->m_iterNum;i++){
        ifstream input;
        lr->openOneFile(lr->m_trainFile,input);
        lr->setNowIter(i);

        lr->m_stat_iter->initStat("Iter");

        cerr<<"Train Begin Iter: "<<lr->getNowIter()<<endl;

        while(0==(ret=lr->trainOneSample(input))){
        }

        lr->closeOneFile(input);
        if(-2==ret){
            cerr<<"Training Error"<<endl;
            return -1;
        }

        cerr<<"Train Done Iter: "<<lr->getNowIter()<<endl;

        lr->m_stat_global->collectStatAndPrint(lr);
        lr->m_stat_iter->collectStatAndPrint(lr);
        lr->saveModel();
    }


    time.printNowTime();
    cerr<<"Logistic Regression Train Totally Done!"<<endl;

    return 0;
}

Model* loadModel(string confPath,string confFile){
    Model* model = new Model;

    model->lr = initLogistic(confPath,confFile,1);

    return model;
}

void freeModel(Model* model){
    freeLogistic(model->lr);
    delete model;
}

int predict(Model* model,Sample* sample){
    model->lr->m_sample = sample;//???

    model->lr->feedForward();
    model->lr->predict();

    return 0;
}

int predictOneFile(Model* model){
    Timer time;
    int ret = 0;
    cerr<<"Logistic Regression Predict Started.Good Luck!"<<endl;
    time.printNowTime();

    ifstream input;
    model->lr->openOneFile(model->lr->m_predictFile,input);

    while(true){
        ret=model->lr->readOneSample(input);

        if(-1==ret){
            break;
        }
        if(-2==ret){
            model->lr->closeOneFile(input);
            return -1;
        }

        model->lr->feedForward();

        model->lr->predict();
    }

    model->lr->closeOneFile(input);
    model->lr->m_stat_global->collectStatAndPrint(model->lr);

    time.printNowTime();
    cerr<<"Logistic Regression Predict Totally Done!"<<endl;

    return 0;
}





/* vim: set expandtab ts=4 sw=4 sts=4 tw=100: */
