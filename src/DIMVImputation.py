import numpy as np 
from tqdm import tqdm 
from typing import * 
from dpers import DPERS 
from conditional_expectation import RegularizedConditionalExpectation 
import numpy as np
from utils import normalize, find_largest_elements, rmse_loss, rescale

class DIMVImputation:
    """ Imputation class that use conditional expectation to fill the missing data position
        The covariance matrix would be computed by the DPER algorithm  
    """
    def __init__(self):
        self.initializing = False
        self.Xtrain = None
        self.X_train_norm = None
        self.train_mean = None
        self.train_std = None
        self.cov = None
        self.no_0_var_mask = None
        self.cov_no_zeros = None
        self.estimator = None
        self.best_alpha = None
        self.cv_score = None
        
    def fit(self, X: np.ndarray, initializing: bool= False, n_jobs = None) -> None:
        """
        Imputation class that use conditional expectation to fill the missing data position
        The covariance matrix would be computed by the DPER algorithm 
        When to use initializing : Initialize = True should be set if there is many missing pattern in the dataset for example randomly missing
        Args:
            X (np.ndarray): Xtrain, dataset used to computer the covariance matrix 
            initializing (bool, optional): Defaults to False; when  initializing is set as True, the missing position would be all initialize by 0 (also the mean of the scaled)
            n_jobs : umber of parallel jobs to run the covariance computation 
        """
        self.initializing = initializing 
        self.Xtrain = X
        self.X_train_norm, self.train_mean, self.train_std = normalize(X, X)

        self.cov = DPERS().fit(self.X_train_norm, j_jobs = n_jobs)
        self.no_0_var_mask = np.diag(self.cov)!=0

        self.cov_no_zeros = self.cov[self.no_0_var_mask, :][:, self.no_0_var_mask]

        self.estimator = RegularizedConditionalExpectation(
                cov = self.cov_no_zeros,
                initializing = self.initializing) 
        self.best_alpha = None 
        self.cv_score = None 
        
    def fit(self, X: np.ndarray, initializing: bool = False) -> None:
        """
        Fit the imputation model on the given training data.

        Args:
            X (np.ndarray): Training data.
            initializing (bool, optional): Whether to initialize missing values to zero. Defaults to False.
        """
        self.initializing = initializing
        self.Xtrain = X
        self.X_train_norm, self.train_mean, self.train_std = normalize(X, X)
        self.cov = DPERS().fit(self.X_train_norm)
        self.no_0_var_mask = np.diag(self.cov) != 0
        self.cov_no_zeros = self.cov[self.no_0_var_mask, :][:, self.no_0_var_mask]
        self.estimator = RegularizedConditionalExpectation(cov=self.cov_no_zeros, initializing=self.initializing)
        self.best_alpha = None
        self.cv_score = None
        self.cv_mode = False

    def cross_validate(
        self,
        alphas: List[float],
        train_percent: float,
        features_corr_threshold: float = 0,
        mlargest_features: int = 1
    ) -> Dict[str, Union[float, List[float]]]:
        """
        Perform cross-validation on the model.

        Args:
            alphas (List[float], optional): Alpha values to use for cross-validation. Defaults is [0.0, 0.01, 0.1, 1.0, 10.0, 100.0].
            train_percent (float, optional): Percentage of the training data to use for cross-validation. Defaults to 1. (100%)
            features_corr_threshold (float, optional): Correlation threshold to use when selecting features. Defaults to 1.
            mlargest_features (int, optional): Number of top features to use when selecting features. Defaults to 1.

        Returns:
            Dict[str, Union[float, List[float]]]: {"best_alpha" :best_alpha, "cv_scores": scores}  Results of cross-validation.
        """
        #set cv_mode = True
        self.cv_mode = True 

        if train_percent is None:
            train_percent = 1 

        if  alphas is None: 
             alphas = [0.0, 0.01, 0.1, 1.0, 10.0, 100.0]
        
        print("Start Cross Validation with alphas = {} and {} % of training set".format(alphas, train_percent*100))
        assert (train_percent <= 1 and train_percent > 0.1), " train_percent must be in range(0.1, 1] "

        best_score = np.inf
        best_alpha = None

        Xtrain = self.Xtrain
        
        scores = {}
        n = self.Xtrain.shape[0]
        
        shuffle_idxes = np.arange(n)
        np.random.shuffle(shuffle_idxes)
        sample_idxes  = shuffle_idxes[: int(n * train_percent)]
        
        X_cv = self.Xtrain[sample_idxes, :]
        
        observed_mask = ~np.isnan(X_cv)

        for alpha in alphas:
            print("Running Cross Validation, alpha = ".format(alpha)) 
            try:
                X_cv_imputed = self.transform(
                    X_cv, 
                    alpha=alpha,
                    features_corr_threshold=features_corr_threshold, 
                    mlargest_features=mlargest_features)
                    
                # print(X_cv_imputed.shape)

                score = rmse_loss(X_cv, X_cv_imputed) 
                #X_cv contain nan value, only computer score at not NaN position (not using initialize as 0 position when initialized=True)
                #print("RMSE at alpha {} is {}".format(alpha, score))

                scores.update({alpha: score})

                if score < best_score:
                    best_score = score
                    best_alpha = alpha 

            except Exception as err:
                print(f"Unexpected {err=}, {type(err)=}")
                raise
        
        self.best_alpha = best_alpha
        self.cv_score = scores  
        results = {"best_alpha" :best_alpha, "cv_scores": scores} 
        #turn off the cross-validation mode: cv_mode 
        self.cv_mode = False 
        return best_alpha 

    def filter_features(
        self, 
        s_missing_fts: np.array,
        s_avai_fts: np.array, 
        idx: int,  
        th: float = 0,
        mlargest: int = None,
        ):
        """
        Choosing a set of feature that correlation with the target feature f (imputed features) with criteria:
            1. Have correlation equal to or greater than a threshold features_corr_threshold
            2. If no feature meet the condition, choose top m features that have highest correlation with feature f
        
        Args: 
            s_missing_fts (numpy.array): a boolean array indicating which features are missing for the current instance
            s_avai_fts (numpy.array): a boolean array indicating which features are available for the current instance
            idx (int): the index of the target feature to be imputed 
            th (float, optional): the threshold for selecting features with correlation to the target feature. Default value is 0. (mean all the feature in `s_avai_fts`)
            mlargest (int, optional): the maximum number of features to select if no feature meets the correlation threshold. Default value is None, which indicates that all available features should be considered.
        
        Returns:
            new_s_avai_fts (numpy.array): a boolean array indicating which features have been selected for imputation, based on the specified criteria. 
        """

        mlargest =  mlargest + 1
        S = self.cov_no_zeros 

        threshold_filter = abs(self.cov_no_zeros[idx, :]) >= th 
        mlargest_filter = find_largest_elements(
            s_missing_fts, 
            s_avai_fts, 
            self.cov_no_zeros[idx, :].copy(), 
            mlargest) 
        
        new_s_avai_fts = np.zeros_like(self.cov_no_zeros[idx, :], dtype=bool) 

        if np.sum(threshold_filter) > 1:             
            new_s_missing_fts  =  s_missing_fts   
            new_s_avai_fts     =  s_avai_fts * threshold_filter
        
        if np.sum(new_s_avai_fts) == 0: 
            new_s_avai_fts     =  s_avai_fts * mlargest_filter
            n = self.cov_no_zeros.shape[0]
            tmp = np.arange(n)

        return new_s_avai_fts 

    def transform(
            self, 
            X_input: np.ndarray, 
            alpha: np.ndarray = 0,
            #run_cross_validation: bool = True,
            #train_percent_cv: float = 1.0,
            #alphas_cv: List = None,
            features_corr_threshold = None, 
            mlargest_features = None 
            ) -> np.ndarray:  

        """ 
        Imputes the estimated value for missing position in the input array X_input using the covariance matrix calculated in the fit step 
        Args:
            X_input (np.ndarray): Input array of shape (n_samples, n_features) containing missing values
            alpha (np.ndarray, optional): Array of shape (n_features,) containing the regularization parameter for each feature
            cv_mode (bool, optional): If True, performs cross-validation for selecting the best value of alpha. Default is False
            train_percent (float, optional): Float value between 0 and 1 representing the percentage of data used for training in cross-validation. Default is 1.0
            features_corr_threshold (float, optional): Correlation threshold value to select features. Default is None
            mlargest_features (int, optional): The maximum number of top features to select based on correlation with target feature. Default is None

        Returns:
            np.ndarray: Imputed array of shape (n_samples, n_features)
        """
        X_input = X_input.astype(np.float64)
        
        if mlargest_features is not None:
            assert mlargest_features <= self.cov_no_zeros.shape[0], \
                "Value of features_corr_threshold should be smaller than number of features that have " 
            assert features_corr_threshold is not None, \
                "mlargest_features is the m features to filter if the condition features_corr_threshold is not meet, please config the features_corr_threshold's value" 
        
        if features_corr_threshold is not None: 
            assert 0 <= features_corr_threshold <= 1, \
                "Value must be between 0 and 1"  
            if mlargest_features is None:
                mlargest_features = 1

        #scaling by mean and std of train set 
        X_test_norm, _, _ = normalize(X_input, mean=self.train_mean, std=self.train_std)
        
        X            = X_test_norm[:, self.no_0_var_mask]
        missing_data = X_test_norm[:, self.no_0_var_mask]

        if self.initializing == True:
            X[np.isnan(X)] = 0  

        _, m  = X.shape
        
        X_imp_normed = np.zeros_like(X, dtype='float') 
    
        #check cross_validation only exist if initialize = True 

        if self.initializing==True:
            X[np.isnan(X)] == 0

        S = self.cov_no_zeros

        for idx in tqdm(range(m)):
            
            if self.initializing == True or self.cv_mode == True:
                all_values = np.arange(X[:, idx].shape[0])
                missing_values = all_values
            else: 
                missing_values = np.where(np.isnan(X[:, idx]) != 0)[0] 
             
            while len(missing_values) > 0 : 
                s = missing_values[0]
                #reduce feature set

                s_missing_fts = np.isnan(X[s,:]) 
                s_avai_fts    = ~np.isnan(X[s,:])
                # import pdb; pdb.set_trace()
                if features_corr_threshold is not None: 
                    s_avai_fts = self.filter_features(
                        s_missing_fts, 
                        s_avai_fts,
                        idx,
                        th = features_corr_threshold,
                        mlargest = mlargest_features
                        ) 
                    # print("Ture") 
                    
                # print("s_avai_fts", np.sum(s_avai_fts))
                
                same_missing_fts_mask = np.sum(~np.isnan(X[:, s_missing_fts]), axis=1) == 0 
                same_avai_fts_mask    = np.sum(np.isnan(X[:, s_avai_fts]), axis=1) == 0
                same_missing_pattern  = same_missing_fts_mask * same_avai_fts_mask

                s_missing_fts[idx] = False 
                s_avai_fts[idx] = False  

                pred = self.estimator.transform(
                    feature_idxes = s_avai_fts, 
                    label_idx = idx, 
                    rows = same_missing_pattern, 
                    missing_data = missing_data, 
                    alpha = alpha
                    )
                
                X_imp_normed[same_missing_pattern, idx] = pred

                #pop from the queue 
                pop_idxs = np.where(same_missing_pattern)[0]
                missing_values = missing_values[~np.in1d(missing_values, pop_idxs)]
        
        #rescaled the imputed result to the original scale
        Xoutput = X_test_norm.copy()
        Xoutput[:, self.no_0_var_mask] = X_imp_normed
        Xoutput[np.isnan(Xoutput)] = 0 

        #rescaled by mean and std of train set 
        X_imp_rescaled = rescale(Xoutput, self.train_mean, self.train_std) 

        Ximp = X_input.copy()
        Ximp[np.isnan(X_input)] = X_imp_rescaled[np.isnan(X_input)]
        
        if self.cv_mode == True:
            Ximp_cv_mode = X_imp_rescaled
            return Ximp_cv_mode

        return Ximp
