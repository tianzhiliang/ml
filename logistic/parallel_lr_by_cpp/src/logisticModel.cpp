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

    if(0==i-last){
        return index;
    }

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
        fprintf(stderr,"%d:%f\t",i,array[i]);
    }
    fprintf(stderr,"\n");
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

int Logistic::vectorXvector(double* a,double* b,int dim,double& res){
    res=0;
    int index = 0;
    if(1==m_isSparseFeature){
        for(int i=0;i<m_sample->featureNum;i++){
            index = m_sample->IdSorted[i];
            res += a[index]*b[index];
        }
    }
    else{
        for(int i=0;i<m_featureSize;i++){
            res+=a[i]*b[i];
        }
    }


    return 0;
}

int Timer::printNowClock(){
    time_t now = time(0);
    m_nowTime = localtime(&now);
    fprintf(stderr,"%d/%d_%d:%d:%d\n",m_nowTime->tm_mon+1,m_nowTime->tm_mday,\
            m_nowTime->tm_hour,m_nowTime->tm_min,m_nowTime->tm_sec);
    return 0;
}

int Timer::printNowClockLight(){
    time_t now = time(0);
    m_nowTime = localtime(&now);
    fprintf(stderr,"%d:%d:%d\n",m_nowTime->tm_hour,m_nowTime->tm_min,m_nowTime->tm_sec);
    return 0;
}

Timer::Timer(){
    gettimeofday(&m_start,NULL);
}

void Timer::getTimeReset(){
    gettimeofday(&m_start,NULL);
}

Timer& Timer::Obj(){
    static Timer time;
    return time;
} 

double Timer::getTime(){
    struct timeval now;
    double time=0;
    gettimeofday(&now,NULL);

    time=(now.tv_sec-m_start.tv_sec)*1000 + ((double)(now.tv_usec-m_start.tv_usec))/1000;

    getTimeReset();

    return time;
}

MPIC& MPIC::Obj(){
    static MPIC obj;
    return obj;
}

int MPIC::initMPI(int argc,char** argv){
    MPI_Init(&argc,&argv);
    return 0;
}

int MPIC::getMachineNum(){
    int size;
    MPI_Comm_size(MPI_COMM_WORLD,&size);
    return size;
}

int MPIC::getMachineId(){
    int id;
    MPI_Comm_rank(MPI_COMM_WORLD,&id);
    return id;
}

void MPIC::freeMPI(){
    MPI_Finalize();
}

int MPIC::mergeNetPara(double* sendBuf,double* recvBuf,int size){
    MPI_Allreduce(sendBuf,recvBuf,size,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
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
        //m_workerNum = conf->workerNum;
        m_featureSize = conf->featureSize;
        m_alpha = conf->alpha;
        m_paraBatch = conf->paraBatch;
        m_randRangeW = conf->randRangeW;
        m_randRangeB = conf->randRangeB;
        m_randDivisor = conf->randDivisor;

        m_lossMode = conf->lossMode;
        m_loadMode = conf->loadMode;
        m_txtMode = conf->txtMode;
        m_isReLoadModel = conf->isReLoadModel;
        m_modelFileTrainOutput = conf->modelFileTrainOutput;
    }

    m_predictRightThres = conf->predictRightThres;
    m_debug = conf->debug;
    m_odictPath = conf->odictPath;
    m_odictName = conf->odictName;
    m_isSparseFeature = conf->isSparseFeature;
    m_isUseOdictAutoAdjust = conf->isUseOdictAutoAdjust;

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
        fprintf(stderr,"ERROR:fopen failed,file:%s\n",confName.c_str());
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
            fprintf(stderr,"ERROR:2<size in loadConfFile,count:%d\tsize:%d\tline:%s\n",count,size,line);
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
    fprintf(stderr,"will readConf,file:%s\t",confName.c_str());

    if(-1==loadConfFile(confMap,confName)){
        return -1;
    }

    //文件
    if(confMap.find("trainFile")==confMap.end()){
        fprintf(stderr,"WARNING: trainFile not in conf\n");
    }
    else{
        conf->trainFile=confMap["trainFile"];
        fprintf(stderr,"trainFile:%s\n",conf->trainFile.c_str());
    }

    if(confMap.find("predictFile")==confMap.end()){
        fprintf(stderr,"WARNING: predictFile not in conf\n");
    }
    else{
        conf->predictFile=confMap["predictFile"];
        fprintf(stderr,"predictFile:%s\n",conf->predictFile.c_str());
    }

    if(confMap.find("modelFileTrainOutput")==confMap.end()){
        fprintf(stderr,"WARNING: modelFileTrainOutput not in conf\n");
    }
    else{
        conf->modelFileTrainOutput=confMap["modelFileTrainOutput"];
        fprintf(stderr,"modelFileTrainOutput:%s\n",conf->modelFileTrainOutput.c_str());
    }

    if(confMap.find("modelFileToLoad")==confMap.end()){
        fprintf(stderr,"WARNING: modelFileToLoad not in conf\n");
    }
    else{
        conf->modelFileToLoad=confMap["modelFileToLoad"];
        fprintf(stderr,"modelFileToLoad:%s\n",conf->modelFileToLoad.c_str());
    }

    if(confMap.find("odictPath")==confMap.end()){
        fprintf(stderr,"WARNING: odictPath not in conf\n");
    }
    else{
        conf->odictPath=confMap["odictPath"];
        fprintf(stderr,"odictPath:%s\n",conf->odictPath.c_str());
    }

    if(confMap.find("odictName")==confMap.end()){
        fprintf(stderr,"WARNING: odictName not in conf\n");
    }
    else{
        conf->odictName=confMap["odictName"];
        fprintf(stderr,"odictName:%s\n",conf->odictName.c_str());
    }


    //训练参数
    if(confMap.find("iterNum")==confMap.end()){
        fprintf(stderr,"WARNING: iterNum not in conf\n");
    }
    else{
        conf->iterNum=atoi(confMap["iterNum"].c_str());
        fprintf(stderr,"iterNum:%d\n",conf->iterNum);
    }

    if(confMap.find("featureSize")==confMap.end()){
        fprintf(stderr,"WARNING: featureSize not in conf\n");
    }
    else{
        conf->featureSize=atoi(confMap["featureSize"].c_str());
        fprintf(stderr,"featureSize:%d\n",conf->featureSize);
    }

    if(confMap.find("alpha")==confMap.end()){
        fprintf(stderr,"WARNING: alpha not in conf\n");
    }
    else{
        conf->alpha=atof(confMap["alpha"].c_str());
        fprintf(stderr,"alpha:%f\n",conf->alpha);
    }

    if(confMap.find("L2_regular")==confMap.end()){
        fprintf(stderr,"WARNING: L2_regular not in conf\n");
    }
    else{
        conf->L2_regular=atof(confMap["L2_regular"].c_str());
        fprintf(stderr,"L2_regular:%f\n",conf->L2_regular);
    }

    if(confMap.find("paraBatch")==confMap.end()){
        fprintf(stderr,"WARNING: paraBatch not in conf\n");
    }
    else{
        conf->paraBatch=atoi(confMap["paraBatch"].c_str());
        fprintf(stderr,"paraBatch:%d\n",conf->paraBatch);
    }

    if(confMap.find("randRangeW")==confMap.end()){
        fprintf(stderr,"WARNING: randRangeW not in conf\n");
    }
    else{
        conf->randRangeW=atof(confMap["randRangeW"].c_str());
        fprintf(stderr,"randRangeW:%f\n",conf->randRangeW);
    }

    if(confMap.find("randRangeB")==confMap.end()){
        fprintf(stderr,"WARNING: randRangeB not in conf\n");
    }
    else{
        conf->randRangeB=atof(confMap["randRangeB"].c_str());
        fprintf(stderr,"randRangeB:%f\n",conf->randRangeB);
    }

    if(confMap.find("predictRightThres")==confMap.end()){
        fprintf(stderr,"WARNING: predictRightThres not in conf\n");
    }
    else{
        conf->predictRightThres=atof(confMap["predictRightThres"].c_str());
        fprintf(stderr,"predictRightThres:%f\n",conf->predictRightThres);
    }

    if(confMap.find("randDivisor")==confMap.end()){
        fprintf(stderr,"WARNING: randDivisor not in conf\n");
    }
    else{
        conf->randDivisor=atoi(confMap["randDivisor"].c_str());
        fprintf(stderr,"randDivisor:%d\n",conf->randDivisor);
    }



    //模式
    if(confMap.find("debug")==confMap.end()){
        fprintf(stderr,"WARNING: debug not in conf\n");
    }
    else{
        conf->debug=atoi(confMap["debug"].c_str());
        fprintf(stderr,"debug:%d\n",conf->debug);
    }

    if(confMap.find("lossMode")==confMap.end()){
        fprintf(stderr,"WARNING: lossMode not in conf\n");
    }
    else{
        conf->lossMode=atoi(confMap["lossMode"].c_str());
        fprintf(stderr,"lossMode:%d\n",conf->lossMode);
    }

    if(confMap.find("loadMode")==confMap.end()){
        fprintf(stderr,"WARNING: loadMode not in conf\n");
    }
    else{
        conf->loadMode=atoi(confMap["loadMode"].c_str());
        fprintf(stderr,"loadMode:%d\n",conf->loadMode);
    }

    if(confMap.find("txtMode")==confMap.end()){
        fprintf(stderr,"WARNING: txtMode not in conf\n");
    }
    else{
        conf->txtMode=atoi(confMap["txtMode"].c_str());
        fprintf(stderr,"txtMode:%d\n",conf->txtMode);
    }

    if(confMap.find("isSparseFeature")==confMap.end()){
        fprintf(stderr,"WARNING: isSparseFeature not in conf\n");
    }
    else{
        conf->isSparseFeature=atoi(confMap["isSparseFeature"].c_str());
        fprintf(stderr,"isSparseFeature:%d\n",conf->isSparseFeature);
    }

    if(confMap.find("isUseOdictAutoAdjust")==confMap.end()){
        fprintf(stderr,"WARNING: isUseOdictAutoAdjust not in conf,default is 1.means use\n");
    }
    else{
        conf->isUseOdictAutoAdjust=atoi(confMap["isUseOdictAutoAdjust"].c_str());
        fprintf(stderr,"isUseOdictAutoAdjust:%d\n",conf->isUseOdictAutoAdjust);
        conf->isUseOdictAutoAdjust = 1;
    }

    if(confMap.find("predictMode")==confMap.end()){
        fprintf(stderr,"WARNING: predictMode not in conf,default is 1,means sparse\n");
    }
    else{
        conf->predictMode=atoi(confMap["predictMode"].c_str());
        fprintf(stderr,"predictMode:%d\n",conf->predictMode);
        conf->isSparseFeature = 1;
    }

    if(confMap.find("isReLoadModel")==confMap.end()){
        fprintf(stderr,"WARNING: isReLoadModel not in conf\n");
    }
    else{
        conf->isReLoadModel=atoi(confMap["isReLoadModel"].c_str());
        fprintf(stderr,"isReLoadModel:%d\n",conf->isReLoadModel);
    }

    fprintf(stderr,"Worker: %d readConf done\n",MPIC::Obj().getMachineId());
    return 0;
}

int Logistic::openOneFile(string file,ifstream& input){
    if(0==m_predictMode && 1<m_workerNum){//多worker的训练模式  
        char fileCh[g_MAX_CH_LINE];
        sprintf(fileCh,"%s_%d",file.c_str(),m_workerId);
        file = string(fileCh);
    }

    input.open(file.c_str());

    if(input.fail()){
        fprintf(stderr,"ERROR:input.fail() in openTrainFile,file:%s\n",file.c_str());
        return -1;
    }

    m_lineCount=0;

    if(0==m_debug%2){
        fprintf(stderr,"openOneFile:%s\n",file.c_str());
    }

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
    stringstream numSstr;
    sodict_snode_t* snode = new sodict_snode_t;

    if(getline(input,lineStr).eof()){
        return -1;
    }

    size = mySplit(lineStr,g_DELIM_FIRST,m_lineBuffer);

    if((size<1 && 1>=m_predictMode) || size<0){
        fprintf(stderr,"ERROR:(size<1 && ) || size<0 in readOneSample,line:%lld\n",m_lineCount);
        return -2;
    }
    if(0==size && 2==m_predictMode){
        fprintf(stderr,"WARING:0==size && 1==mode in readOneSample,line:%lld\n",m_lineCount);
    }

    if(1>=m_predictMode){
        m_sample->label = atoi(m_lineBuffer[index].c_str());
        if(1!=m_sample->label && 0!=m_sample->label){
            fprintf(stderr,"ERROR:labelInput label MUST be 0/1. 1!=m_sample-0>label && 0!=m_sample,line:%lld\n"\
                    ,m_lineCount);
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

        if(0==m_txtMode){
            m_sample->Id[i]=atoi(cell[0].c_str());
        }
        else if(1==m_txtMode){
            numSstr.clear();
            numSstr<<cell[0];
            numSstr>>m_sample->IdOri[i];

            snode->sign1=m_sample->IdOri[i] & g_MAX_UINT;
            snode->sign2=m_sample->IdOri[i] >> 32;

            if(ODB_SEEK_OK!=odb_seek_search(m_featureMap,snode)){
                fprintf(stderr,"WARNING:feature not find in odb_seek,line:%lld\tindex:%d\n",m_lineCount,index);
                continue;
            }           
            m_sample->Id[i]=snode->cuint1;
            snode->cuint1=0;
        }
        else{
            fprintf(stderr,"not support,txtMode:%d\n",m_txtMode);
        }
        m_sample->value[i]=atof(cell[1].c_str());

    }

    m_lineCount++;

    delete snode;
    return 0;
}

int Logistic::idCopyAndSort(){
    memcpy(m_sample->IdSorted,m_sample->Id,m_sample->featureNum*sizeof(int));
    sort(m_sample->IdSorted,m_sample->IdSorted+m_sample->featureNum);
}

int Logistic::feedForward(){
    double WX=0;

    Timer time1;
    transSampleToInput();

    if(0==m_debug%11){
        fprintf(stderr,"NOTICE:transSampleToInput in feedForward Time:%f ms\n",time1.getTime());
    }

    if(1==m_isSparseFeature){
        idCopyAndSort();
        if(0==m_debug%11){
            fprintf(stderr,"NOTICE:idCopyAndSort in feedForward Time:%f ms\n",time1.getTime());
        }
    }

    vectorXvector(m_feature,m_weight,m_featureSize,WX);
    m_WX_b = WX + m_bias;

    if(0==m_debug%11){
        fprintf(stderr,"NOTICE:vectorXvector in feedForward Time:%f ms\n",time1.getTime());
    }

    logistic(m_WX_b,m_sample->output);
    if(0==m_debug%11){
        fprintf(stderr,"NOTICE:logistic in feedForward Time:%f ms\n",time1.getTime());
    }

    return 0;
}


int Logistic::transSampleToInput(){
    if(0==m_isSparseFeature){
    memset(m_feature,0,sizeof(double)*m_featureSize);
    }

    for(int i=0;i<m_sample->featureNum;i++){
        m_feature[m_sample->Id[i]] = m_sample->value[i];
    }

    return 0;
}

double Logistic::getL2Norm(){
    double res = 0;
    for(int i=0;i<m_featureSize;i++){
        res += m_weight[i]*m_weight[i];
    }
    res += m_bias* m_bias;
    return res;
}

int Logistic::predict(){
    m_sample->error = m_sample->label - m_sample->output;

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

    if(0==m_debug%3){
        if(m_sample->isRight){
            fprintf(stderr,"Worker: %d m_WX_b:%f\toutput:%f\tresult:%d\tlabel:%d\terror:%f\tRIGHT\n",m_workerId,m_WX_b,m_sample->output,m_sample->result,m_sample->label,m_sample->error);
        }
        else{
            fprintf(stderr,"Worker: %d m_WX_b:%f\toutput:%f\tresult:%d\tlabel:%d\terror:%f\tWRONG\n",m_workerId,m_WX_b,m_sample->output,m_sample->result,m_sample->label,m_sample->error);
        }
    }

    m_stat_pbatch->addOneSample(this,m_sample);
    m_stat_iter->addOneSample(this,m_sample);
    m_stat_global->addOneSample(this,m_sample);

    if(0==m_debug%4){
        cerr<<"Worker: "<<m_workerId<<" DEBUG,line:"<<m_lineCount<<"\t";
        printDoubleArray(m_feature,m_featureSize);
    }

    return 0;
}

int Logistic::batchComm(){

    if(0==m_stat_global->m_totalCount % m_paraBatch && 0!=m_stat_global->m_totalCount){
        m_stat_pbatch->m_networkTimer->getTimeReset();
        Timer time,time1;

        memcpy(m_sendBuf,m_weight,m_featureSize*sizeof(double));
        m_sendBuf[m_featureSize]=m_bias;
    
        if(0==m_debug%11){
        fprintf(stderr,"memcpy to sendBuf in network Time: %f ms\n",time1.getTime());
        }

        MPIC::Obj().mergeNetPara(m_sendBuf,m_recvBuf,m_featureSize+1);
        
        if(0==m_debug%11){
        fprintf(stderr,"mergeNetPara in network Time: %f ms\n",time1.getTime());
        }

        for(int i=0;i<m_featureSize+1;i++){
            m_recvBuf[i] /= m_workerNum;
        }

        memcpy(m_weight,m_recvBuf,m_featureSize*sizeof(double));//用memcpy代替for循环中+=,意味着本机器本次训的error不要了
        m_bias=m_recvBuf[m_featureSize];

        if(0==m_debug%11){
        fprintf(stderr,"m_recvBuf memcpy in network Time: %f ms\n",time1.getTime());
        }

        if(0==m_debug%2){
            fprintf(stderr,"Worker: %d Do Allreduce: at Count: %lld Time: %.2f ms\n",\
                    m_workerId,m_stat_global->m_totalCount,time.getTime());
        }
        if(0==m_debug%3){
            fprintf(stderr,"Worker: %d Do Allreduce Detail: Size: %d weight_0: %f bias: %f\n",\
                    m_workerId,m_featureSize*sizeof(double),m_weight[0],m_bias);
        }

        m_stat_pbatch->m_networkTime += m_stat_pbatch->m_networkTimer->getTime();
        m_stat_pbatch->m_totalTime += m_stat_pbatch->m_totalTimer->getTime();        

        m_stat_pbatch->collectStatAndPrint(this);        
        m_stat_iter->addTimeStat(m_stat_pbatch);
        m_stat_pbatch->timeAllPrint(this);

        setInitDouble(m_sendBuf,m_featureSize+1,0);
        m_stat_pbatch->initStat("P_batch");

        if(0==m_workerId){
            if(0==access("saveModel",0)){
                m_isSaveModel = true;
                saveModel();
            }
        }

    }

    return 0;
}

int Logistic::backPropagate(){
    int index = 0;
    if(1==m_isSparseFeature){
        for(int i=0;i<m_sample->featureNum;i++){
            index = m_sample->IdSorted[i];
            m_weight[index] += m_alpha * (m_sample->error * m_feature[index] + m_L2_regular * m_weight[index]);
        }
    }
    else{
        for(int i=0;i<m_featureSize;i++){
            if(!doubleEqual(m_feature[i],0)){
                m_weight[i] += m_alpha * (m_sample->error * m_feature[i] + m_L2_regular * m_weight[i]);
                index = i;
            }
        }
    }


    m_bias += m_alpha * m_sample->error;

    if(0==m_debug%3){
        fprintf(stderr,"Worker: %d lastIndex: %d alpha:%f*(labelError:%f*feature:%f+L2norm:%f)=finalError:%f\n"\
                ,m_workerId,index,m_alpha,m_sample->label-m_sample->output,m_feature[index],\
                m_L2_regular*m_weight[index], m_alpha * (m_sample->error * m_feature[index] + m_L2_regular * m_weight[index]));
    }

    return 0;
}

Stat::Stat(){
    m_trainTimer = new Timer;
    m_networkTimer = new Timer;
    m_readTimer = new Timer;
    m_totalTimer = new Timer;
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

    timeAllReset();
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

void Stat::timeAllReset(){
    m_trainTimer->getTimeReset();
    m_networkTimer->getTimeReset();
    m_readTimer->getTimeReset();
    m_totalTimer->getTimeReset();
    m_trainTime=0;
    m_networkTime=0;
    m_readTime=0;
    m_totalTime=0;
}

int Stat::timeAllPrint(Logistic* lr){
    if(m_mark!="Global" && m_mark!="Iter" && 0!=lr->m_debug%2 && 0!=lr->m_debug%3){
        return -1;
    }

    fprintf(stderr,"Worker: %d %s readTime: %f trainTime: %f ms\tnetworkTime: %f ms\ttotalTime: %f ms\n",\
            lr->m_workerId,m_mark.c_str(),m_readTime,m_trainTime,m_networkTime,m_totalTime);

    timeAllReset();
    return 0;
}

int Stat::addTimeStat(Stat* stat){
    m_trainTime+=stat->m_trainTime;
    m_networkTime+=stat->m_networkTime;
    m_readTime += stat->m_readTime;
    m_totalTime+=stat->m_totalTime;

    return 0;
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
            fprintf(stderr,"Worker: %d %s Count: %lld Acc: %f Right: %lld posRes: %lld posLabel: %lld p2p: %lld p2n: %lld n2n: %lld n2p: %lld line: %lld\t",\
                    lr->m_workerId,m_mark.c_str(),m_totalCount,getAcc(),getRightCount(),m_p2p+m_n2p,m_posLabelCount,m_p2p,\
                    m_p2n,m_n2n,m_n2p,lr->m_lineCount);
            time.printNowClock();
        }
        else if(0==lr->m_debug%2){
            fprintf(stderr,"Worker: %d %s Count: %lld Acc: %f p2p: %lld p2n: %lld n2n: %lld n2p: %lld\t",\
                    lr->m_workerId,m_mark.c_str(),m_totalCount,getAcc(),m_p2p,m_p2n,m_n2n,m_n2p);
            time.printNowClockLight();
        }
    }
    else if(2==lr->m_predictMode){
        fprintf(stderr,"Worker: %d %s Count: %lld posRes: %lld negRes: %lld\t",\
                lr->m_workerId,m_mark.c_str(),m_totalCount,m_posResCount,m_negResCount);
        time.printNowClock();
    }

    return 0;
}

int Logistic::trainOneSample(ifstream& input){
    int ret = 0;

    m_stat_pbatch->m_totalTimer->getTimeReset();
    m_stat_pbatch->m_readTimer->getTimeReset();

    ret=readOneSample(input);

    m_stat_pbatch->m_readTime += m_stat_pbatch->m_readTimer->getTime();

    if(0!=ret){
        return ret;
    }

    m_stat_pbatch->m_trainTimer->getTimeReset();

    Timer time1;


    feedForward();
    if(0==m_debug%11){
        fprintf(stderr,"NOTICE:feedForward Time:%f ms\n",time1.getTime());
    }

    predict();

    if(0==m_debug%11){
        fprintf(stderr,"NOTICE:predict Time:%f ms\n",time1.getTime());
    }

    backPropagate();

    if(0==m_debug%11){
        fprintf(stderr,"NOTICE:backPropagate Time:%f ms\n",time1.getTime());
    }

    m_stat_pbatch->m_trainTime += m_stat_pbatch->m_trainTimer->getTime();
    double tmp = m_stat_pbatch->m_totalTimer->getTime();
    m_stat_pbatch->m_totalTime += tmp;

    batchComm();

    return 0;
}

sodict_build_t* initOdict(){
    sodict_build_t* odict=odb_creat(g_ODICT_SIZE);
    if(NULL==odict){
        return NULL;
    }
    return odict;
}

int freeOdict(sodict_build_t* odict){
    if (ODB_DESTROY_OK != odb_destroy(odict)){
        return -1;
    }
    return 0;
}

int freeReadOnlyOdict(sodict_search_t* odict){
    if(ODB_DESTROY_OK!=odb_destroy_search(odict)){
        return -1;
    }
    return 0;
}

Logistic* initLogistic(string confPath,string confFile,int predictMode){
    Logistic* lr;
    try{
        lr = new Logistic;

        lr->m_sample = new Sample;
        lr->m_stat_global = new Stat;
        lr->m_stat_iter = new Stat;
        lr->m_stat_pbatch = new Stat;
        lr->m_lineBuffer = new string[g_MAX_FEATURE_SIZE+1];

    }
    catch(bad_alloc &info){
        cerr<<info.what()<<endl;
        cerr<<"FATAL:initLogistic failed"<<endl;
        return NULL;
    }

    lr->m_predictMode=predictMode;
    lr->m_workerId = MPIC::Obj().getMachineId();
    lr->m_workerNum = MPIC::Obj().getMachineNum();
    lr->m_isSaveModel = false;

    Conf conf;
    if(-1==readConf(confPath,confFile,&conf)){
        return NULL;
    }

    lr->copyConfValue(&conf);

    if(1==lr->m_txtMode){
        if(-1==(lr->m_featureSize=lr->loadFeatureMap())){
            fprintf(stderr,"FATAL:initLogistic failed\n");
            return NULL;
        }    
    }

    lr->reLoadModel(&conf);


    if(-1==lr->mallocSample(lr,lr->m_sample)){
        return NULL;
    }

    if(0==lr->m_isReLoadModel && 0==lr->m_predictMode){
        lr->m_weight=(double*)calloc(lr->m_featureSize,sizeof(double));
    }
    lr->m_feature=(double*)calloc(lr->m_featureSize,sizeof(double));
    lr->m_sendBuf=(double*)calloc(lr->m_featureSize+10,sizeof(double));
    lr->m_recvBuf=(double*)calloc(lr->m_featureSize+10,sizeof(double));
    setInitDouble(lr->m_sendBuf,lr->m_featureSize+1,0);

    if(NULL==lr->m_weight||NULL==lr->m_weight||NULL==lr->m_feature||NULL==lr->m_sendBuf||NULL==lr->m_recvBuf){
        cerr<<"FATAL:initLogistic failed"<<endl;
        return NULL;
    }


    if(0==lr->m_isReLoadModel && 0==lr->m_predictMode){
        lr->initNetPara();
    }

    lr->m_stat_pbatch->initStat("P_batch");
    lr->m_stat_global->initStat("Global");

    return lr;
}

void freeLogistic(Logistic* lr){
    free(lr->m_weight);
    free(lr->m_sendBuf);
    free(lr->m_recvBuf);
    free(lr->m_feature);

    freeReadOnlyOdict(lr->m_featureMap);

    delete []lr->m_lineBuffer;

    delete lr->m_stat_global;
    delete lr->m_stat_iter;
    delete lr->m_stat_pbatch;

    lr->freeSample(lr->m_sample);

    delete lr;
}

int Logistic::mallocSample(Logistic* lr,Sample* sample){

    sample->IdOri = (uint64_t*)calloc(lr->m_featureSize,sizeof(uint64_t));
    sample->Id = (int*)calloc(lr->m_featureSize,sizeof(int));
    sample->IdSorted = (int*)calloc(lr->m_featureSize,sizeof(int));
    sample->value = (double*)calloc(lr->m_featureSize,sizeof(double));

    if(NULL==sample->Id || NULL==sample->value || NULL==sample->IdOri || NULL==sample->IdSorted){
        cerr<<"FATAL:NULL==sample->Id || NULL==sample->value,mallocSample failed"<<endl;
        return -1;
    }
    return 0;
}

void Logistic::freeSample(Sample* sample){
    free(sample->IdOri);
    free(sample->Id);
    free(sample->IdSorted);
    free(sample->value);
}

int Logistic::initNetPara(){
    randInit(m_weight,m_featureSize,0-m_randRangeW,m_randRangeW,m_randDivisor);

    randInit(&m_bias,1,0-m_randRangeB,m_randRangeB,m_randDivisor);
    return 0;
}

int Logistic::saveModel(){
    if(1<m_workerNum && 0!=m_workerId && 1!=m_workerId){//只有0,1两个进程存模型
        return -1;
    }

    char file[g_MAX_CH_LINE];

    if(true==m_isSaveModel){
        static int saveModelIterByhand = 0;
        sprintf(file,"%s.hand%d",m_modelFileTrainOutput.c_str(),saveModelIterByhand);
        saveModelIterByhand++;
        m_isSaveModel = false;  
        system("rm -rf saveModel");
    }
    else{
        sprintf(file,"%s.iter%d.worker%d",m_modelFileTrainOutput.c_str(),getNowIter(),m_workerId);
    }

    ofstream output(file,ios::binary|ios::out);

    output.write((char*)&m_iterNum,sizeof(int));
    output.write((char*)&m_featureSize,sizeof(int));
    output.write((char*)&m_randDivisor,sizeof(int));
    output.write((char*)&m_paraBatch,sizeof(int));

    output.write((char*)&m_alpha,sizeof(double));
    output.write((char*)&m_L2_regular,sizeof(double));
    output.write((char*)&m_randRangeW,sizeof(double));
    output.write((char*)&m_randRangeB,sizeof(double));

    output.write((char*)m_weight,sizeof(double)*m_featureSize);
    output.write((char*)&m_bias,sizeof(double));

    output.close();

    fprintf(stderr,"Worker: %d saveModel done,file:%s time:",m_workerId,file);
    Timer::Obj().printNowClock();
    return 0;
}

int Logistic::readModel(string file){
    ifstream input(file.c_str(),ios::binary|ios::in);

    input.read((char*)&m_iterNum,sizeof(int));
    input.read((char*)&m_featureSize,sizeof(int));
    input.read((char*)&m_randDivisor,sizeof(int));
    input.read((char*)&m_paraBatch,sizeof(int));

    input.read((char*)&m_alpha,sizeof(double));
    input.read((char*)&m_L2_regular,sizeof(double));
    input.read((char*)&m_randRangeW,sizeof(double));
    input.read((char*)&m_randRangeB,sizeof(double));

    m_weight=(double*)calloc(m_featureSize,sizeof(double));

    input.read((char*)m_weight,sizeof(double)*m_featureSize);
    input.read((char*)&m_bias,sizeof(double));

    input.close();

    if(0==m_debug%2){
        cerr<<"Worker: "<<m_workerId<<" readModel done,m_iterNum:"<<m_iterNum<<" m_featureSize:"<<m_featureSize<<" m_randDivisor:"<<m_randDivisor;
        cerr<<"m_paraBatch:"<<m_paraBatch<<" m_alpha:"<<m_alpha<<" m_L2_regular:"<<m_L2_regular<<" m_randRangeW:"<<m_randRangeW;
        cerr<<"m_randRangeB:"<<m_randRangeB<<" m_weight_0:"<<m_weight[0]<<" m_bias:"<<m_bias<<endl;
    }

    return 0;
}

int Logistic::reLoadModel(Conf* conf){
    if(0==m_isReLoadModel && 0==m_predictMode){
        return -1;
    }

    readModel(m_modelFileToLoad);

    if(2==m_isReLoadModel && 0==m_predictMode){
        m_iterNum = conf->iterNum;
        m_alpha = conf->alpha;
        m_L2_regular = conf->L2_regular;
        m_randRangeW = conf->randRangeW;
        m_randRangeB = conf->randRangeB;
        m_paraBatch = conf->paraBatch;
        m_randDivisor = conf->randDivisor;
    }

    //todo:print

    return 0;
}

int trainModel(string confPath,string confFile){
    Timer time;
    int ret = 0;
    int workerId = MPIC::Obj().getMachineId();
    int workerNum = MPIC::Obj().getMachineNum();
    cerr<<"Logistic Regression Train Started.Worker:"<<workerId<<" of "<<workerNum<<" Started\t";
    time.printNowClock();

    Logistic* lr = initLogistic(confPath,confFile,0);
    if(NULL==lr){
        return -1;
    }

    cerr<<"Worker: "<<lr->m_workerId<<" initLogistic done"<<endl;
    time.printNowClock();
    for(int i=0;i<lr->m_iterNum;i++){
        ifstream input;
        lr->openOneFile(lr->m_trainFile,input);
        lr->setNowIter(i);

        lr->m_stat_iter->initStat("Iter");

        cerr<<"Worker: "<<lr->m_workerId<<" Train Begin Iter: "<<lr->getNowIter()<<endl;

        while(0==(ret=lr->trainOneSample(input))){
        }

        lr->closeOneFile(input);
        if(-2==ret){
            cerr<<"Training Error"<<endl;
            return -1;
        }

        cerr<<"Worker: "<<lr->m_workerId<<" Train Done Iter: "<<lr->getNowIter()<<endl;

        lr->m_stat_global->addTimeStat(lr->m_stat_iter);
        lr->m_stat_iter->timeAllPrint(lr);

        lr->m_stat_global->collectStatAndPrint(lr);
        lr->m_stat_iter->collectStatAndPrint(lr);
        lr->saveModel();
    }


    cerr<<"Worker: "<<lr->m_workerId<<" Logistic Regression Train Totally Done!\t";
    time.printNowClock();

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

int Logistic::loadFeatureMap(){
    int featureSize = 0;    

    m_featureMap = odb_load_search((char*)m_odictPath.c_str(),(char*)m_odictName.c_str());
    if(NULL==m_featureMap){
        fprintf(stderr,"ERROR,odb_load_search failed in loadFeatureMap,path:%s\tname:%s\n",\
                m_odictPath.c_str(),m_odictName.c_str());
        return -1;
    }

    featureSize = odb_search_get_nodenum(m_featureMap);

    fprintf(stderr,"TRACE:odict nodeNum:%d\todict hashNum:%d\n",featureSize,odb_search_get_hashnum(m_featureMap));
    fprintf(stderr,"TRACE,loadFeatureMap done,path:%s\tname:%s\n",m_odictPath.c_str(),m_odictName.c_str());

    return featureSize;
}

int getFeatureOdict(string confPath,string confFile){
    Conf* conf = new Conf;
    if(-1==readConf(confPath,confFile,conf)){
        return -1;
    }

    sodict_build_t* featureMap = initOdict();
    if(NULL==featureMap){
        return -1;
    }

    if(-1==buildFeatureMap(conf->trainFile,featureMap,conf->debug)){
        return -1;
    }

    if(ODB_SAVE_OK!=odb_save(featureMap,(char*)conf->odictPath.c_str(),(char*)conf->odictName.c_str())){
        fprintf(stderr,"ERROR,odb_save failed in getFeatureOdict\n");
        return -1;
    }

    freeOdict(featureMap);
    delete conf;

    fprintf(stderr,"TRACE,getFeatureOdict done\n");
    return 0;
}

int buildFeatureMap(string file,sodict_build_t* featureMap,int debug){
    ifstream input(file.c_str());
    if(input.fail()){
        fprintf(stderr,"open file failed in buildFeatureMap,file:%s\n",file.c_str());
        return -1;
    }

    fprintf(stderr,"TRACE,will buildFeatureMap,file:%s\n",file.c_str());

    string lineStr, cell[g_MAX_CELL_NUM];
    int size=0,index=0,cellsize=0,ret=0,featureSize=0,label=0;
    long long lineCount = 0;
    uint64_t featureOri = 0;
    sodict_snode_t* snode = new sodict_snode_t;
    string* lineBuffer = new string[g_MAX_FEATURE_SIZE];
    Sample* sample = new Sample;
    stringstream numSstr;

    while(!getline(input,lineStr).eof()){
        size = mySplit(lineStr,g_DELIM_FIRST,lineBuffer);

        if(size<1){
            fprintf(stderr,"ERROR:size<1 in buildFeatureMap,line:%lld\n",lineCount);
            return -1;
        }

        index = 0;
        sample->label = atoi(lineBuffer[index].c_str());
        if(1!=sample->label && 0!=sample->label){
            fprintf(stderr,"ERROR:labelInput label MUST be 0/1. 1!=m_sample-0>label && 0!=m_sample,line:%lld\n"\
                    ,lineCount);
            return -1;
        }
        index++;

        sample->featureNum = size-index;
        for(int i=0;i<sample->featureNum;i++){
            cellsize=mySplit(lineBuffer[index],g_DELIM_SECOND,cell);
            if(2!=cellsize){
                cerr<<"ERROR:2!=cellsize in readOneSample,line:"<<lineCount<<"\tindex:"<<index<<endl;
                return -2;
            }
            index++;

            numSstr.clear();
            numSstr<<cell[0];   
            numSstr>>featureOri;

            snode->sign1 = featureOri&g_MAX_UINT;
            snode->sign2 = featureOri>>32;

            if(0==debug%4){
                fprintf(stderr,"TRACE,buildFeatureMap 4,cell %s|%s featureOri %llu sign1 %u sign2 %u\n",cell[0].c_str(),cell[1].c_str(),featureOri,snode->sign1,snode->sign2);
            }

            if(ODB_SEEK_OK==odb_seek(featureMap,snode)){
                continue;
            }   
            snode->cuint1=featureSize;
            snode->cuint2=0;
            ret = odb_add(featureMap,snode,0);//0:如果数据节点存在,不做插入操作
            if(ODB_ADD_OK!=ret){
                fprintf(stderr,"ERROR:odb_add failed in buildFeatureMap,error info:%d\tline:%lld\tindex:%d\n"\
                        ,ret,lineCount,index);
            }

            if(0==debug%4){
                fprintf(stderr,"TRACE,buildFeatureMap 5,add ok featureOri %llu sign1 %u sign2 %u m_featureSize %d\n",featureOri,snode->sign1,snode->sign2,featureSize);
            }
            featureSize++;          
        }
        lineCount++;
    }
    input.close();

    fprintf(stderr,"buildFeatureMap done,featureSize:%d\tlineCount:%lld\n",featureSize,lineCount);

    if(1==m_isUseOdictAutoAdjust){
    odb_adjust(featureMap);//调整hash单元
    }

    fprintf(stderr,"TRACE:odict nodeNum:%d\todict hashNum:%d\n",odb_get_nodenum(featureMap),odb_get_hashnum(featureMap));

    delete snode;
    delete sample;
    delete []lineBuffer;    

    return 0;
}

int predict(Model* model,Sample* sample){
    model->lr->m_sample = sample;

    model->lr->feedForward();
    model->lr->predict();

    return 0;
}

int predictOneFile(Model* model){
    Timer time;
    int ret = 0;
    cerr<<"Logistic Regression Predict Started.Good Luck!"<<endl;
    time.printNowClock();

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
    model->lr->m_stat_global->timeAllPrint(model->lr);

    time.printNowClock();
    cerr<<"Logistic Regression Predict Totally Done!"<<endl;

    return 0;
}





/* vim: set expandtab ts=4 sw=4 sts=4 tw=100: */
