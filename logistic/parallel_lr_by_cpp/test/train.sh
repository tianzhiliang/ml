nodeconf="node.conf"
p_num=`awk '{c+=1}END{print c}' $nodeconf`
pernode=1

cp LRTrainTool bin
scp LRTrainTool hcibase@cp01-sys-hic-gpu-14.cp01.baidu.com:/home/hcibase/tianzhiliang/parallel/logistic/test/bin
scp lr.conf hcibase@cp01-sys-hic-gpu-14.cp01.baidu.com:/home/hcibase/tianzhiliang/parallel/logistic/test/
mpirun -npernode $pernode -machinefile $nodeconf ./bin/LRTrainTool . lr.conf 
