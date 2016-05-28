#input=v1_train_data_h10
#input=hand
#input=v1_train_data_h100
#input=data/dealed/train_0
input=data/train_data_dealed/train_0_sampled
outlog=outlog
errlog=errlog

if [ $# -eq 1 ] ; then
    mark=$1
    errlogtmp=$errlog
    errlog=$errlogtmp_$mark
fi

nohup python svm.py $input > $outlog 2> $errlog &
pid=$(ps -ef | grep "python svm.py" | grep -v grep | awk '{print $2}')
echo "kill me command: kill" $pid 
