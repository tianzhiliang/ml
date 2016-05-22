input=data/test_data_dealed/test_0_sampled
output=data/test_result/test_0_sampled
errlog=errlog
modelfile=model.iter8

mkdir -p data/test_result

if [ $# -eq 1 ] ; then
    mark=$1
    errlog=$mark
fi


cat $input | python predict.py $modelfile > $output 2> $errlog
tail $errlog
