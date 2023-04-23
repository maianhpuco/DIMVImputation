import numpy as np
from typing import *
from tqdm import tqdm

class DPERS:
    def fit(self, X:np.ndarray)->np.ndarray:
        """
        DPER implementations in pythons
        """
        assert isinstance(X, np.ndarray) and (np.ndim(X) == 2),\
                ValueError("Expected 2D numpy array");
    
        n, p = X.shape;
    
        # Covariance matrix to be estimated
        S = np.zeros((p, p));
        
        # The diagonal line of S is the sample variance
        for i in range(p):
            x_i = X[:, i];
            x_i = x_i[~np.isnan(x_i)]
            S[i, i] = np.var(x_i) 
    
        # Upper triangle indices
        upper_idx = list(zip(*np.triu_indices(p, 1)));
    
    
        # Calculate the upper triangle matrix of S
        for (i, j) in tqdm(upper_idx):
            X_ij = X[:, [i, j]];
            
            # remove entry with all missing value
            missing_idx = np.isnan(X_ij).all(1)
            X_ij = X_ij[~missing_idx];
    
            S_ii, S_jj = S[i, i], S[j, j];
            if (S_ii != 0) and (S_jj !=0 ):
                S[i, j] = self.find_cov_ij(X_ij, S_ii, S_jj);
            else:
                S[i, j] = np.nan;
    
        S = S + S.T;
    
        # Halving the diagonal line;
        for i in range(p):
            S[i,i] = S[i,i] * .5;
    
        return S;
        
    
    def find_cov_ij(self, X_ij:np.ndarray, S_ii:float, S_jj:float)->float:
        """
        Given matrix (N x 2), each columns is an observation with some missing value;
        """
     
    
        # Number of entries without any missing value
        idx = ~np.isnan(X_ij).any(-1);
    
        # X without any missing observations
        complt_X = X_ij[idx, :]
        
        # Number of observations without any missing value
        m = np.sum(idx);
    
        s11 = np.sum(complt_X[:, 0]**2);
        s22 = np.sum(complt_X[:, 1]**2);
        s12 = np.sum(complt_X[:, 0] * complt_X[:, 1]);
    
    
        # Coef of polynomial
        coef = np.array([
            s12 * S_ii * S_jj,
            m * S_ii * S_jj - s22 * S_ii - s11 * S_jj,
            s12,
            -m
            ])[::-1]
    
        roots = np.roots(coef);
        roots = np.real(roots);
    
        scond = S_jj - roots ** 2/ S_ii;
    
        etas = -m * np.log(
                scond, 
                out=np.ones_like(scond)*np.NINF, 
                where=(scond>0)
                ) - (S_jj - 2 * roots / S_ii * s12 + roots**2 / S_ii**2 * s11)/scond
        return roots[np.argmax(etas)];
    
    
