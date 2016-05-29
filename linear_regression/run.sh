python linearRegression.py ./data/data_train model.txt log2_train.txt 200 0.01 10 10 0.5 200 > log_train.txt

python linearRegressionTest.py ./data/data_test model.txt result.txt log2_test.txt 0.5 #> log_test.txt
