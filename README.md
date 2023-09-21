
# Conditional expectation with regularization for missing data imputation (DIMV) 

This is an **imputation package for missing data**, which can be easily install with pip. 

The code repository associated with the paper: "Conditional expectation with regularization for missing data imputation." This paper is under evaluation for the journal you can find it at (![DIMV](https://arxiv.org/abs/2302.00911))

# Introduction:
Conditional Distribution-based Imputation of Missing Values with Regularization (![DIMV](https://arxiv.org/abs/2302.00911)): An algorithm for imputing missing data with low RMSE, scalability, and explainability. Ideal for critical domains like medicine and finance, DIMV offers reliable analysis, approximated confidence regions, and robustness to assumptions, making it a versatile choice for data imputation. DIMV is under the assumption that it relies on the normally distributed assumption as part of its theoretical foundation. The assumption of normality is often used in statistical methods and imputation techniques because it simplifies data modeling.  


#  Installation: 
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
For example you have missing data array named X and you want to impute following the DIMV imputation 

```python 
from DIMVImputation import DIMVImputation

imputer = DIMVImputation()
imputer.fit(X, initializing=self.initializing) # fit on X_train if splitting is neccessary 

if self.cross_validation: # regularization 
imputer.cross_validate()


X_imputed = imputer.transform(X)  # transform on X_test if splitting is neccessary
``` 


## If you install with option 2(clone the repo) 
The ```.fit()``` function applied on train set to compute the covariance matrix. The convariance matrix is computed from the train set. 

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
imputer.cross_validate(train_percent=1, alphas = [0.0, 0.01, 0.1, 1.0, 10.0, 100.0] ) 
# default value for alpha = [0.0, 0.01, 0.1, 1.0, 10.0, 100.0] 

X_test_imp = imputer.transform(X_test_miss) 
```


# Comparision: 


Here's an illustration of DIMV's imputation for MNIST and FashionMNIST:
 
![image](https://github.com/maianhpuco/DIMVImputation/assets/34562568/9fe8efb4-4085-41fa-993f-c61335c33751)

![image](https://github.com/maianhpuco/DIMVImputation/assets/34562568/6e8f9732-6bcf-4a84-aceb-bd2218ab4f7e)

In this comparison, we evaluate DIMV's performance on both small datasets with randomly missing data patterns and medium datasets (MNIST and FashionMNIST) with monotone missing data patterns (cutting a piece of the image on the top right).

For small datasets with random missing data: 

![image](https://github.com/maianhpuco/DIMVImputation/assets/34562568/8eec91bf-37af-4344-be15-d57c4e58bb64)


For medium datasets (MNIST and FashionMNIST):
 ![image](https://github.com/maianhpuco/DIMVImputation/assets/34562568/7a08d514-9805-4f83-88b0-7e413294c53a) 

 
DIMV has demonstrated strong performance in terms of computational efficiency and robustness, spanning from small to medium datasets and accommodating various types of missing data patterns. 
Indeed, some popular imputation methods, like k-nearest Neighbors Imputation (KNNI), can encounter performance issues regarding computational time, especially when dealing with large datasets or high-dimensional data. 
 
# Contents:
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

 
