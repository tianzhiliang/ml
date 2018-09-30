mark=emb_1w
input=data/model.txt_with_dict_10w_h1w

K=50
similarity_type="cos"
total_iter=30

auto_mark=${mark}_${K}_${similarity_type}_${total_iter}
result=result/${auto_mark}
errlog=errlog/${auto_mark}

mkdir -p errlog
mkdir -p result

python Kmeans.py $input $K $similarity_type $total_iter > $result 2> $errlog
