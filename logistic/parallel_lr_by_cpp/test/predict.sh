#./LRTestTool . model.txt.iter1 ../data/liblinearA1a/ala.t.shuffle.test.deal 1

cp LRTestTool bin
./bin/LRTestTool . lr.conf
scp LRPredTool hcibase@cp01-sys-hic-gpu-14.cp01.baidu.com:/home/hcibase/tianzhiliang/parallel/logistic/test/bin
scp lr.conf hcibase@cp01-sys-hic-gpu-14.cp01.baidu.com:/home/hcibase/tianzhiliang/parallel/logistic/test/

