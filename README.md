

# Install packages:
create env and install libs

``` 
python3 -m venv env
pip install -r requirements.txt 
```

# Example: 
1. Compute the covariance matrix using fit function 

```
from DIMVImputation import DIMVImputation

#create a sample data
data = np.random.randint(0, 100, size=(100, 30)).astype('float64')
missing_rate = 0.5
missing_data = create_randomly_missing(data, missing_rate)

#train test split
test_size = .2
split_index = int(len(missing_data) * (1 - test_size))
X_train_ori, X_test_ori = data[:split_index, :], data[split_index:, :]
X_train_miss, X_test_miss = missing_data[:split_index, :], missing_data[
    split_index:, :]
hyperparam_1 = 
impuation_with_no_initialization(X_train_miss, X_test_miss, X_test_ori) 


imputer1 = DIMVImputation()
start = time.time()

imputer1.fit(X_train_miss, initializing=False, n_jobs=1)

imputer1.cross_validate(train_percent=1)
X_test_imp1 = imputer.transform(X_test_miss) 
```
2. Imputation:
 ```
 from DIMVImputation.DIMVImputation import DIMVImputation

 ```
