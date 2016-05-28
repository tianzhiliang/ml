dataset=(test_0 test_1 test_2 test_3 test_4 test_5 test_6 test_7 test_8 test_9)

for file in ${dataset[*]}; 
do
#    cat $file | python normalized.py > train_data_dealed/$file &
    cat test/$file | python normalized.py > test_data_dealed/$file &
done
