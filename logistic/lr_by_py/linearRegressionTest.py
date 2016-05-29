import sys
import math

g_MAX_EXP_NUM=300
def sign(x):   #ignore x == 0
    if x > 0:
        return 1
    elif x < 0:
        return -1

def logistic(WX_B):
    if WX_B < 0-g_MAX_EXP_NUM:
        return 0
    elif WX_B > g_MAX_EXP_NUM:
        return 1
    else:
        tmp=math.exp(0-WX_B)
        return 1/(1-tmp)

def testOneSample(W,b,label,X,flog,fres,threshold):
    label=(int(label)+1)/2
    lenX=len(X)
    WX=0
    for i in range(0,lenX):
        WX=WX+float(X[i]) * float(W[i])
    #Y=sign(WX+float(b))
    #Y=logistic(WX+float(b))
    Y = WX+float(b)
    flog.write("WX: "+str(WX)+" WX+b: "+str(WX+float(b))+" Y: "+str(Y)+" label: "+str(label))
    fres.write(str(Y)+"\n")
    if abs(label-Y) < threshold:
        flog.write(" right\n")
        return True
    else:
        flog.write(" wrong\n")
        return False

if __name__ == '__main__':
    fin=open(sys.argv[1],"r")
    fmodel=open(sys.argv[2],"r")
    fres=open(sys.argv[3],"w")
    flog=open(sys.argv[4],"w")
    threshold=float(sys.argv[5])#0.0001

    #for l in fmodel:
    W=fmodel.readline().strip().split()
    b=fmodel.readline().strip().split()[0]

    count,right_count=0,0
    for l in fin:
        l=l.strip().split()
        label=l[0]
        feature=l[1:]
        if testOneSample(W,b,label,feature,flog,fres,threshold):
            right_count=right_count+1
        count=count+1
    sys.stdout.write("right: "+str(right_count)+" count: "+str(count)+" acc: "+str(float(right_count)/count)+"\n")
