input=data/hand
#input=data/v1_train_data_h100
outlog=outlog
errlog=errlog_$input

if [ $# -eq 1 ] ; then
    mark=$1
    outlog=outlog_$mark
    errlog=errlog_$mark
fi

nohup python svm.py $input > $outlog 2> $errlog &
pid=$(ps -ef | grep "python svm.py" | grep -v grep | awk '{print $2}')
echo "kill me command: kill" $pid 
