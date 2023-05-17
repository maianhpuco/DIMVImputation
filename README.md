
# Conditional expectation with regularization for missing data imputation (DIMV) 

The code repository associated with the paper: "Conditional expectation with regularization for missing data imputation". This paper is under evaluation for presentation at the NeurIPS 2023 conference.


## Contents:
The codes are structured as follows:  

``` 
.
├── README.md
├── example.ipynb
├── requirements.txt
└── src
    ├── DIMVImputation.py
    ├── __init__.py
    ├── conditional_expectation.py
    ├── dpers.py
    └── utils.py 
 ``` 
 
 

In ```/src``` folders:

- ```DIMVImputation.py``` implements DIMV imputation algorithm for imputing for missing data. 
- ``` dpers.py``` that implements the DPER algorithm for computing the covariance matrix used in the DIMV (Conditional expectation with regularization for missing data imputation) algorithm. (input is a normalized input matrix). 
- ```conditional_expectation.py``` contains the computation for the regularized conditional expectation for a sliced position in the dataset, given the covariance matrix. 
    
```example.ipynb``` is a Jupyter Notebook file that contains examples demonstrating how to use the functionalities and methods. 


## Usage: 
### Installation: Install from source: 

- Step 1: Clone the repository 

```git clone <repository-url>``` 
Then, create a virtual environment and activate the environment. 

- Step 2: Install the libraries from the "requirements.txt" file.  

```
pip install -r requirements.txt 
```

### Model fitting: 

The ```.fit()``` function applied on train set to compute the covariance matrix. The convariance matrix is computed from the train set. 

Create a sample dataset: 
```
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

Fit the model on train set: 
``` 
from DIMVImputation.DIMVImputation import DIMVImputation 
imputer = DIMVImputation()
imputer.fit(X_train_miss, initializing=False) 
```

Then use ```.cross_validate()``` to grid search for optimal value for reguralization value $\alpha$ and finally tranform the missing data ```X_test_miss``` 

```
imputer.cross_validate(train_percent=1, alphas = [0.0, 0.01, 0.1, 1.0, 10.0, 100.0] ) 
# default value for alpha = [0.0, 0.01, 0.1, 1.0, 10.0, 100.0] 

X_test_imp = imputer.transform(X_test_miss) 
```
 
