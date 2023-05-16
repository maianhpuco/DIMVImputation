
# Conditional expectation with regularization for missing data imputation (DIMV) 

This repo contains the codebase for the following paper: "Conditional expectation with regularization for missing data imputation (DIMV) ". This paper is under review at the conference NeurIPS 2023.


## Contents:
The codes are structured as follows:  

``` 
.
├── README.md
├── example.ipynb
├── requirements.txt
├── setup.py
└── src
    ├── DIMVImputation.py
    ├── __init__.py
    ├── conditional_expectation.py
    ├── dpers.py
    └── utils.py 
 ``` 
The contents of this repo can be described as follows. ```/src``` folders:
- ```DIMVImputation.py``` implements DIMV imputation algorithm. 
- ``` dpers.py``` implements DPER algorithm implementation for a normalizeed dataset, which computing the covariance matrix used in the DIMV algorithm. 
- ```conditional_expectation.py``` containing implemtation for the computation for the regularized conditional expectation for a sliced position in the dataset given the covariance matrix. 


## Usage: 

- Step 1: Create a virtual environment named "env" and activate the environment 
- Step 2: Install the libraries from the "requirements.txt" file. 

``` 
pip install -r requirements.txt 
```

- Step3 : Fit on train set to compute the covariance matrix using DPER algorithm 
```
from DIMVImputation import DIMVImputation

#For example we have a missing dataset to impute   
data = np.random.randint(0, 100, size=(100, 30)).astype('float64')
missing_rate = 0.5
missing_data = create_randomly_missing(data, missing_rate)


#Create train test split
test_size = .2
split_index = int(len(missing_data) * (1 - test_size))
X_train_ori, X_test_ori = data[:split_index, :], data[split_index:, :]
X_train_miss = missing_data[:split_index, :]
X_test_miss = missing_data[split_index:, :]  

```  


``` 
from DIMVImputation.DIMVImputation import DIMVImputation 
imputer = DIMVImputation()
imputer.fit(X_train_miss, initializing=False, n_jobs=1) 
```

Then use cross validation to search for optimal value for reguralization value $\alpha$ and finally tranform the missing data X_test_miss 

```
imputer.cross_validate(train_percent=1, alphas = [0.0, 0.01, 0.1, 1.0, 10.0, 100.0] ) 
# default value for alpha = [0.0, 0.01, 0.1, 1.0, 10.0, 100.0] 

X_test_imp = imputer.transform(X_test_miss) 
```
 
