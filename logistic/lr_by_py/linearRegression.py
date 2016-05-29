import sys
import random
import math

g_EXP_MAX_NUM=300

def sign(x):   #ignore x == 0
    if x > 0:
        return 1
    elif x < 0:
        return -1

def logsitic(WX_B):
    if WX_B < 0-g_EXP_MAX_NUM:
        return 0
    elif WX_B > g_EXP_MAX_NUM:
        return 1
    tmp=math.exp(-WX_B)
    return 1/(1+tmp)

#def Linear(WX_B):

def trainOneSample(W,b,alpha,label,X,threshold): #sgd
    label=(label+1)/2 #1,-1->1,0 take care
    WX = 0
    lenX=len(X) #lenW == lenX
    for i in range(0,lenX):
        WX = WX + float(X[i]) * W[i]
    #Y = sign(WX+b)
    #Y = logsitic(WX+b)
    Y = WX+b
    flog.write("label: "+str(label)+" Y: "+str(Y)+" WX+b "+str(WX+b)+" WX: "+str(WX)+" b: "+str(b)+"\n")
    if not abs(label - Y) < threshold: #test
        for i in range(lenX):#lenW == lenX
            W[i]=W[i] + alpha*float(label - Y)*float(X[i])
        b = b + alpha*float(label - Y)
        return False,W,b
    else:
        return True,W,b

if __name__ == '__main__': 
    if not len(sys.argv)== 10:
        print "./"+sys.argv[0]+" trainFile modelFile logFile featureSize learning-rate W-init-range b-init-range threshold iterNum"
        print "eg: ./"+sys.argv[0]+" ./data/data_train model.txt log2_train.txt 200 0.01 5 5 0.0001 100 > log_train.txt"
        exit(1)
    fin=open(sys.argv[1],"r") #trainFile
    #ftest=open(sys.argv[2],"r") #testFile
    fmodel=open(sys.argv[2],"w") #modelFile
    #fout=open(sys.argv[4],"w") #predictResultFile
    flog=open(sys.argv[3],"w") 
    featureSize=int(sys.argv[4]) #featureSize
    alphaI = float(sys.argv[5]) #0.1
    WrandomI = int(sys.argv[6]) #1,5
    brandomI = int(sys.argv[7]) #1,5
    threshold = float(sys.argv[8]) #0.0001
    iterNum = int(sys.argv[9])#100
    W=[]
    for i in range(0,featureSize):
        W.append(random.random()*1000%WrandomI) #-1~1
    b=random.random()*1000%brandomI #-1~1
    alpha = 0.1

    count,right_count,this_count,this_right_count = 0,0,0,0
    trainData=[]
    for l in fin:
        l=l.strip().split()
        label=int(l[0])  #1,2->-1,1
        feature=l[1:]
        trainData.append([label,feature])
    lenTrainData = len(trainData)
    index=0
    iterNumCount=0
    while True:
        if index == lenTrainData-1:
            sys.stdout.write("right: "+str(right_count)+" count: "+str(count)+" acc: "+str(float(right_count)/count)+" this_right: "+str(this_right_count)+" this_count: "+str(this_count)+" this acc: "+str(float(this_right_count)/this_count)+"\n")
            iterNumCount=iterNumCount+1
            if this_right_count == this_count:
                print "Train Done ACC 100%"
                break
            if iterNum == iterNumCount:
                print "Train Done, iterNum is enough"
                break
            index,this_count,this_right_count = 0,0,0

        right,W,b=trainOneSample(W,b,alpha,trainData[index][0],trainData[index][1],threshold)
        index=index+1
        if right:
            flog.write("count: "+str(count)+" right\n")
            right_count  = right_count + 1
            this_right_count  = this_right_count + 1
        else:
            flog.write("count: "+str(count)+" wrong\n")
        count=count+1
        this_count=this_count+1
    for w in W:
        fmodel.write(str(w)+" ")
    fmodel.write("\n")
    fmodel.write(str(b)+"\n")
