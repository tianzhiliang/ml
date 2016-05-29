/***************************************************************************
 * 
 * Copyright (c) 2014 Baidu.com, Inc. All Rights Reserved
 * 
 **************************************************************************/
 
 
 
/**
 * @file testLRTrain.cpp
 * @author tianzhiliang(com@baidu.com)
 * @date 2014/11/15 14:29:46
 * @brief 
 *  
 **/

#include <logisticModel.h>

int main(int argc,char** argv){
    if(3!=argc){
        cerr<<argv[0]<<" confPath confFile"<<endl;
        return -1;
    }

    string confPath = string(argv[1]);
    string confFile = string(argv[2]);

    Model* model = loadModel(confPath,confFile);

    predictOneFile(model);

    return 0;
}




















/* vim: set expandtab ts=4 sw=4 sts=4 tw=100: */
