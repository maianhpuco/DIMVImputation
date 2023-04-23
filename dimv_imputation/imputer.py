import numpy as np 
from typing import * 
from dimv_imputation.utils import *
#from .cross_validation import CrossValidation 
from dimv_imputation.dpers import DPERS 

class DIMVImputation:
    def __init__(
            self, 
            alpha: float = 1.0, 
            mean: np.ndarray = None, 
            covariance_matrix: np.ndarray = None, 
            zero_initialization: bool = False, 
            normalize = True,
            ): 

        self.alpha = alpha
        #self.cv = CrossValidation()
        self.X_norm = None 
    
    

    def fit(self, X: np.ndarray) -> None:
        """
        fit : compute the covariance matrix for the normalized dataset 
        
        """
        self.Xnorm = normalize(X)
        
        #self.missing_mask = np.isnan(X)

        self.cov = DPERS().fit(X)

    def transform(
            self, 
            X_input: np.ndarray, 
            alpha: float = 0, 
            initialize: int = True) -> None: 

        """ 
        Imputing the missing array X, using the covariance matrix calculated in the fit step:
        - with initialization = 0 
        - without the initialization 
        Recommendation for parameter setting: with data that have a large number of missing pattern -> using the setting initialization (performance is much different)

        """
        X, mean, std = normalize(X_input)

        S = self.cov #S represent for the covariance matrix of the train set self.cov 
        
        missing_mask = np.isnan(X)

        n, m  = X.shape
        
        X_imp_normed = np.zeros_like(X, dtype='float') 
            
        if initialization==True:
            X[np.isnan(X)] == 0

        for idx in tqdm(range(m)):
            if initialization==True: 
                all_values = np.arange(X[:, idx].shape[0])
                missing_values = all_values
            else: 
                missing_values = np.where(np.isnan(X[:, idx]) != 0)[0] 
            
            
            while len(missing_values) > 0 : 
    
                s_missing_fts     =  np.isnan(X[s,:])
                s_avai_fts = ~np.isnan(X[s,:])
    
                same_missing_fts_mask = np.sum(~np.isnan(X[:, s_missing_fts]), axis=1) == 0 
    
                same_avai_fts_mask = np.sum(np.isnan(X[:, s_avai_fts]), axis=1) == 0
                same_missing_pattern = same_missing_fts_mask * same_avai_fts_mask

                s_missing_fts[idx] = False 
                s_avai_fts[idx] = False  
    
                Sf = S[idx, :][s_avai_fts][:, np.newaxis]
                
                So = S[s_avai_fts,: ][:, s_avai_fts]
    
                I = np.identity(So.shape[0]) 
                S_inv = np.linalg.inv(So + alpha * I) 
    
            
                first_term = S_inv @ Sf 
                Zf = X[same_missing_pattern, :][:, s_avai_fts] 
                result = first_term.T @ Zf.T 
    
    
                X_imp_normed[same_missing_pattern, idx] = result
                pop_idxs = np.where(same_missing_pattern)[0]
                missing_values = missing_values[~np.in1d(missing_values, pop_idxs)]
        
        #rescaled the imputed result to the original scale

        X_imp = rescaled(X_imp_norm, mean, std) 

        return X_imp_normed
    
    
    
