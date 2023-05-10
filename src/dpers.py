import numpy as np
from typing import *
from tqdm import tqdm
import multiprocessing as mp 
from multiprocessing import Pool 

class DPERS:
    def __init__(self):
        self.Xtrain = None
        self.cov = None 
    
    """
    Efficient Parameter Estimation for Randomly Missing Data.

    This implementation is a modification of DPERS used only with mean-scaled dataset.

    Attributes:
        None
    """    
    def fit(self, X:np.ndarray, n_jobs=None)->np.ndarray:
        """
        Estimates the covariance matrix of a given dataset with missing values.

        Args:
            X (np.ndarray): A 2D numpy array containing the dataset.

        Returns:
            np.ndarray: A 2D numpy array representing the estimated covariance matrix.
        """
        X = X.astype(np.float64)
        # remove entry with all missing value
        missing_rows = np.isnan(X).all(axis=1) 
        X = X[~missing_rows] 


        assert isinstance(X, np.ndarray) and np.ndim(X) == 2, \
                ValueError("Expected 2D numpy array")
        self.Xtrain = X  
        n, p = X.shape 
    
        # Covariance matrix to be estimated
        S = np.zeros((p, p));
        
        # The diagonal line of S is the sample variance
        for i in range(p):
            x_i = X[:, i];
            x_i = x_i[~np.isnan(x_i)]
            S[i, i] = np.var(x_i) 
        
        # Upper triangle indices
        upper_idx = list(zip(*np.triu_indices(p, 1)));

        if n_jobs is None:
            n_jobs = 1 

        if n_jobs == -1:
            n_jobs = mp.cpu_count()

        if n_jobs==1:

            upper_triag = []
            for i, j in upper_idx:
                upper_triag.append(self.find_cov_ij(i, j, S[i, i], S[j, j], X))
        else:
            with Pool(processes=n_jobs) as pool:   
                upper_triag = []
                with tqdm(total=len(upper_idx)) as pbar:
                    for result in pool.starmap(
                            self.find_cov_ij, 
                            [(i, j, S[i, i], S[j, j], X) for (i, j) in upper_idx]):
                        upper_triag.append(result)
                        pbar.update(1) 
        for idx, (i, j) in enumerate(upper_idx):
            S[i, j] = upper_triag[idx]

        S = S + S.T
        # Halving the diagonal line;
        for i in range(p):
            S[i,i] = S[i,i] * .5
            
        self.cov = S
        return S 
       
    @staticmethod
    def find_cov_ij(i: int, j: int, S_ii: int, S_jj: int, X: np.ndarray) -> float:
        """Estimates the covariance between two features with missing values.

        Args:
            i, j (int): index of the upper triangle covariance matrix  
            S_ii, S_jj (int): diagonal of covariance matrix at position i and respectively 
            X (np.ndarray): The input matrix 

        Returns:
            float: The estimated covariance between the two features.
        """

        X_ij = X[:, [i, j]]
        X_ij = X[:, [i, j]]
        #if (S_ii != 0) and (S_jj !=0 ):
            #return self.find_cov_ij(X_ij, S_ii, S_jj);
        if (S_ii == 0) or (S_jj == 0 ):
            return np.nan 
    
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

