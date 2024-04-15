
# Conditional expectation with regularization for missing data imputation (DIMV) 

This is an **imputation package for missing data**, which can be easily installed with pip. 

The code repository associated with the paper: "Conditional expectation with regularization for missing data imputation." This paper is under evaluation for the journal you can find it at https://arxiv.org/abs/2302.00911 

# Introduction
Conditional Distribution-based Imputation of Missing Values with Regularization (DIMV): An algorithm for imputing missing data with low RMSE, scalability, and explainability. Ideal for critical domains like medicine and finance, DIMV offers reliable analysis, approximated confidence regions, and robustness to assumptions, making it a versatile choice for data imputation. DIMV is under the assumption that it relies on the normally distributed assumption as part of its theoretical foundation. The assumption of normality is often used in statistical methods and imputation techniques because it simplifies data modeling.  

# Comparision 

In this comparison, we evaluate DIMV's performance on both small datasets with randomly missing data patterns and medium datasets (MNIST and FashionMNIST) with monotone missing data patterns (cutting a piece of the image on the top right). 
 
## Randomly Missing Pattern
For small datasets with random missing data: 
<img src="https://github.com/maianhpuco/DIMVImputation/assets/34562568/8eec91bf-37af-4344-be15-d57c4e58bb64" alt="image4" width="500">    

## Monotonic missing pattern 
For medium datasets (MNIST and FashionMNIST):

<img src="https://github.com/maianhpuco/DIMVImputation/assets/34562568/7a08d514-9805-4f83-88b0-7e413294c53a" alt="image5" width="500"> 

Here's an illustration of DIMV's imputation for MNIST and FashionMNIST:

<img src="https://github.com/maianhpuco/DIMVImputation/assets/34562568/394f60cf-d886-4071-9873-1ebc56aa12f7" alt="image1" width="200"> 
<img src="https://github.com/maianhpuco/DIMVImputation/assets/34562568/442c630a-804c-48d7-89f3-39ca1e09be9e" alt="image2" width="200"> 

DIMV has shown promising performance in terms of computational efficiency and robustness across small to medium datasets, accommodating a variety of missing data patterns. However, like many imputation methods, DIMV may face challenges with computational time when dealing with large datasets or high-dimensional data. For instance, popular imputation methods like k-nearest Neighbors Imputation (KNNI) can sometimes encounter performance issues in these scenarios. 

  
# Contents
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

- ```DIMVImputation.py``` implements the DIMV imputation algorithm for imputing missing data. 
- ``` dpers.py``` that implements the DPER algorithm for computing the covariance matrix used in the DIMV (Conditional expectation with regularization for missing data imputation) algorithm. (input is a normalized input matrix). 
- ```conditional_expectation.py``` contains the computation for the regularized conditional expectation for a sliced position in the dataset, given the covariance matrix. 
    
```example.ipynb``` is a Jupyter Notebook file that contains examples demonstrating how to use the functionalities and methods.  


#  Installation
### Option 1: Install with pip 
Install the package with: 
```
!pip install git+https://github.com/maianhpuco/DIMVImputation.git 
```

### Option 2: Install from source: 

- Step 1: Clone the repository 

```git clone <repository-url>``` 
Then, create a virtual environment and activate the environment. 

- Step 2: Install the libraries from the "requirements.txt" file.  

```
pip install -r requirements.txt 
```

# Usages: 

## If you install with option 1 (with pip):
For example, suppose you have a dataset with missing values named X, and you wish to apply the DIMV imputation method to fill in these missing values. 

```python 
from DIMVImputation import DIMVImputation

# Create an instance of the DIMVImputation class
imputer = DIMVImputation()

# Fit the imputer to the data X (possibly X_train if splitting is necessary), without initializing it
imputer.fit(X, initializing=False)

# Apply imputation to the data X (possibly X_test if splitting is necessary)
# By default, the algorithm implements cross-validation to find the best value for the regularization parameter (alpha)
# Default regularization parameter values: alphas = [0.0, 0.01, 0.1, 1.0, 10.0, 100.0]
# Default percentage of data used for training in cross-validation: train_percent=100
X_imputed = imputer.transform(X)

``` 


## If you install with option 2(clone the repo) 
The `.fit()` function is applied to the training set to compute the covariance matrix, which is then calculated based on the training set. 

Create a sample dataset as a numpy array ```missing_data``` 
```python 
#Create train test split
test_size = .2
split_index = int(len(missing_data) * (1 - test_size))

X_train_ori, X_test_ori = data[:split_index, :], data[split_index:, :]

X_train_miss = missing_data[:split_index, :]
X_test_miss = missing_data[split_index:, :]  
```  

Fit the model on the train set: 
```python 
from DIMVImputation.DIMVImputation import DIMVImputation 
imputer = DIMVImputation()
imputer.fit(X_train_miss, initializing=False)

```

Then use ```.cross_validate()``` to grid search for optimal value for reguralization value $\alpha$ and finally tranform the missing data ```X_test_miss``` 

```python

# To input your alpha grid and data percentage for cross-validation, use the following two lines of code
imputer.cross_validate(train_percent=80, alphas=[0.0, 0.01, 0.1, 1.0])
X_test_imp = imputer.transform(X_test_miss, cross_validation=False)

```







 
