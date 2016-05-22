datasets=(train_1 train_2 train_3 train_4 train_5 train_6 train_7 train_8 train_9)

for file in ${datasets[*]}; 
do
    input=dealed/$file
    output=dealed/$file"_sampled"
    awk -F "\t" '{if($1=="-1"){if(rand() < 0.1111){print $0}} else{print $0}}' $input > $output &
done
