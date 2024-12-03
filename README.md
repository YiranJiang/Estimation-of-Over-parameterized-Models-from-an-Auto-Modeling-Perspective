
Estimation of Over-parameterized Models from an Auto-Modeling (AM) Perspective
===============================================================
From a model-building perspective, we propose a paradigm shift for fitting over-parameterized models. Philosophically, the mindset is to fit models to future observations rather than to the observed sample. Technically, given an imputation method to generate future observations, we fit over-parameterized models to these future observations by optimizing an approximation of the desired expected loss function based on its sample counterpart and an adaptive *duality function*. The required imputation method is also developed using the same estimation technique with an adaptive $m\text{-out-of-}n$ bootstrap approach.

## Related Publication

Yiran Jiang and Chuanhai Liu, [Estimation of Over-parameterized Models from an Auto-Modeling Perspective.](https://arxiv.org/pdf/2206.01824) *Major revision at Journal of the American Statistical Association*.

<br>

### Reproduce Main Experimental Results in the Paper:
 
 The paper contains three application studies:

1. Estimation of many-normal-means (R script)
2. $n < p$ Linear regression (R script)
3. MNIST image classification with neural networks (Python script)

<br>

#### Estimation of Many-Normal-Means:

Required package:  

deconvolveR (>=1.2-1), quadprog (>=1.5-8), dirichletprocess (>= 0.4.2)

Run Simulation Experiments:

```{R}
Rscript mnm-simulation.R $a $b
```

 - a: Index of the experiment (1-3)
 - b: Index of the repetitions (1-50), each containing 10 datasets

Alternative -- SLURM Job Script (modify the file if required):
```{sh}
sbatch myjob-array.sh
```

Summary:
```{R}
Rscript summary.R
```

<br>


#### $n < p$ Linear Regression:

Required package:  

glmnet (>=4.1-8), lars (>=1.3), natural (>=0.9.0), LaplacesDemon (>=16.1.6), MASS (>=7.3-60)

Generate Data:
```{R}
cd linear-regression
Rscript data_simulation.R
```

Run Simulation Experiments:
```{R}
Rscript linear_regression.R $alpha $k $tau 
```

 - alpha: Sparsity-related parameter $`\alpha \in \{0.3,0.6,0.9\}`$
 - k: Index of the dataset (1-100)
 - tau: SNR-related parameter $`\tau \in \{0.3,1,3\}`$ 

Alternative -- SLURM Job Script (modify the file if required):
```{sh}
sbatch myjob-array.sh
```

<br>


#### MNIST Image Classification with Neural Networks

Required Packages:

torch (>=1.12.1), torchvision (>=0.13.1), numpy (>=1.21.5), scipy (>=1.9.1), argparse, random, pickle, math

```{python}
cd neural-network
```
AM:

```{python}
python am_imputation.py -o 0
python am_estimation.py -o 0
```

 - -o 0: FC 400-400, L1
 - -o 1: FC 400-400, L2
 - -o 2: FC 800-800, L1
 - -o 3: FC 800-800, L2
 - -o 4: FC 1600-1600, L1
 - -o 5: FC 1600-1600, L2
 - -o 6: CNN, L1
 - -o 7: CNN, L2


Standard Regularization:

```{python}
python penalty_estimation.py -o 0
```

 - -o 0: FC 400-400, L1
 - -o 1: FC 400-400, L2
 - -o 2: FC 800-800, L1
 - -o 3: FC 800-800, L2
 - -o 4: FC 1600-1600, L1
 - -o 5: FC 1600-1600, L2
 - -o 6: CNN, L1
 - -o 7: CNN, L2

Dropout:
```{python}
python dropout.py -o 0
```
 - -o 0: FC 400-400
 - -o 1: FC 800-800
 - -o 2: FC 1600-1600
 - -o 3: CNN



Early-Stopping:
```{python}
python early_stopping_estimation.py
```



### Reproduce Additional Experiments in the Paper



## References

Friedman, Jerome, Hastie, Trevor, and Tibshirani, Rob. "Regularization Paths for Generalized Linear Models via Coordinate Descent." *Journal of Statistical Software*, 33.1 (2010): 1-22.

Miller, Brandon J., and Walker, Stephen G. "DirichletProcess: An R Package for Bayesian Nonparametric Modeling." *CRAN Vignettes*. URL: [https://cran.r-project.org/web/packages/dirichletprocess/vignettes/dirichletprocess.pdf](https://cran.r-project.org/web/packages/dirichletprocess/vignettes/dirichletprocess.pdf).

Narasimhan, Balasubramanian, and Efron, Bradley. "deconvolveR: A $g-$Modeling Program for Deconvolution and Empirical Bayes Estimation." *Journal of Statistical Software*, 94.11 (2020).

Paszke, Adam, Gross, Sam, Massa, Francisco, Lerer, Adam, Bradbury, James, Chanan, Gregory, Killeen, Trevor, Lin, Zeming, Gimelshein, Natalia, Antiga, Luca, Desmaison, Alban, Köpf, Andreas, Yang, Edward, DeVito, Zach, Raison, Martin, Tejani, Alykhan, Chilamkurthy, Sasank, Steiner, Benoit, Lu, Fang, Bai, Junjie, and Chintala, Soumith. "PyTorch: An Imperative Style, High-Performance Deep Learning Library." *Cornell University Library, arXiv.org* (2019).

LeCun, Yann, Bottou, Léon, Bengio, Yoshua, and Haffner, Patrick. "Gradient-Based Learning Applied to Document Recognition." *Proceedings of the IEEE*, 86.11 (1998): 2278-2324.

