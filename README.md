
Estimation of Over-parameterized Models from an Auto-Modeling (AM) Perspective
===============================================================
From a model-building perspective, we propose a paradigm shift for fitting over-parameterized models. Philosophically, the mindset is to fit models to future observations rather than to the observed sample. Technically, given an imputation method to generate future observations, we fit over-parameterized models to these future observations by optimizing an approximation of the desired expected loss function based on its sample counterpart and an adaptive *duality function*. The required imputation method is also developed using the same estimation technique with an adaptive $m\text{-out-of-}n$ bootstrap approach.

## Related Publication

Yiran Jiang and Chuanhai Liu, [Estimation of Over-parameterized Models from an Auto-Modeling Perspective.](https://arxiv.org/pdf/2206.01824) *Major Revision at JASA*.

<br>

### Reproduce Experimental Results in the Paper:
 
 The paper contains three application studies:

1. Estimation of many-normal-means (R script)
2. $n < p$ Linear regression (R script)
3. MNIST image classification with neural networks (Python script)

<br>

#### Estimation of Many-Normal-Means:

Required package:  

deconvolveR (>=1.2-1), quadprog (>=1.5-8)

Run Simulation Experiments:
```{R}
cd many-normal-means
Rscript mnm-simulations.R 1  ## Simulation Setting 1
Rscript mnm-simulations.R 2  ## Simulation Setting 2
Rscript mnm-simulations.R 3  ## Simulation Setting 3
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
python penalty_estimation.py -o 0
```
 - -o 0: FC 400-400
 - -o 1: FC 800-800
 - -o 2: FC 1600-1600
 - -o 3: CNN



Early-Stopping:
```{python}
python early_stopping_estimation.py
```
