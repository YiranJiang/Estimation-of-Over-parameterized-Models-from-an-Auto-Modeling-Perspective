AM:

python am_imputation.py -o 0
python am_estimation.py -o 0

Options:
-o 0: FC 400-400, L1
-o 1: FC 400-400, L2
-o 2: FC 800-800, L1
-o 3: FC 800-800, L2
-o 4: FC 1600-1600, L1
-o 5: FC 1600-1600, L2
-o 6: CNN, L1
-o 7: CNN, L2


Standard Regularization:


python penalty_estimation.py -o 0

Options:
-o 0: FC 400-400, L1
-o 1: FC 400-400, L2
-o 2: FC 800-800, L1
-o 3: FC 800-800, L2
-o 4: FC 1600-1600, L1
-o 5: FC 1600-1600, L2
-o 6: CNN, L1
-o 7: CNN, L2



Dropout:

python penalty_estimation.py -o 0

Options:
-o 0: FC 400-400
-o 1: FC 800-800
-o 2: FC 1600-1600
-o 3: CNN


Early-Stopping:

python early_stopping_estimation.py



