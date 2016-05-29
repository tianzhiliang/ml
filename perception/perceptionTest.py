import sys

def sign(x):   #ignore x == 0
    if x > 0:
        return 1
    elif x < 0:
        return -1

def testOneSample(W,b,label,X,flog,fres):
    lenX=len(X)
    WX=0
    for i in range(0,lenX):
        WX=WX+float(X[i]) * float(W[i])
    Y=sign(WX+float(b))
    flog.write("WX: "+str(WX)+" WX+b: "+str(WX+float(b))+" Y: "+str(Y)+" label: "+str(label))
    fres.write(str(Y)+"\n")
    if str(Y) == label:
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

    #for l in fmodel:
    W=fmodel.readline().strip().split()
    b=fmodel.readline().strip().split()[0]

    count,right_count=0,0
    for l in fin:
        l=l.strip().split()
        label=l[0]
        feature=l[1:]
        if testOneSample(W,b,label,feature,flog,fres):
            right_count=right_count+1
        count=count+1
    sys.stdout.write("right: "+str(right_count)+" count: "+str(count)+" acc: "+str(float(right_count)/count)+"\n")
