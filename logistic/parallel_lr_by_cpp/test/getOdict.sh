nodeconf="node.conf"
p_num=`awk '{c+=1}END{print c}' $nodeconf`

cp getFeatureOdictTool ./bin/

scp getFeatureOdictTool hcibase@cp01-sys-hic-gpu-14.cp01.baidu.com:/home/hcibase/tianzhiliang/parallel/logistic/test/bin
scp lr.conf hcibase@cp01-sys-hic-gpu-14.cp01.baidu.com:/home/hcibase/tianzhiliang/parallel/logistic/test/
mpirun -npernode 1 -machinefile $nodeconf ./bin/getFeatureOdictTool . lr.conf
