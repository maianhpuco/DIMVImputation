import numpy as np
from typing import * 

class RegularizedConditionalExpectation: 
    """
    Class for computing regularized conditional expectation for a sliced data. 

    Args:
        cov (ndarray): Covariance matrix.
        initializing (bool): If True, initialize missing values as 0. Default is False.

    Attributes:
        cov (ndarray): Covariance matrix.
        initializing (bool): If True, initialize missing values as 0.

    Methods:
        transform: Computes regularized conditional expectation.

    Raises:
        ValueError: If the variance of the covariance matrix is zero.

    """
    def __init__(self, 
                 cov: np.ndarray = None, 
                 initializing: bool = False):
        if np.count_nonzero(np.diag(cov)) == 0:
            raise ValueError("Variance cannot be zero.")
        self.cov = cov
        self.initializing = initializing


    def transform(self, 
                  feature_idxes: List[int] = None, 
                  label_idx: int = None, 
                  rows: List[int] = None,
                  missing_data: np.ndarray = None, 
                  alpha: float = None) -> np.ndarray:
        """
        Computes regularized conditional expectation for a slice position in the dataset
            - missing_data[rows, label_idx]: is the target variable that we want to compute 
            - missing_data[rows, feature_idxes]: is the features that used to compute the conditional expectation. 

        Args:
            feature_idxes (list): List of feature indices. Default is None.
            label_idx (int): Index of label column. Default is None.
            rows (list): List of row indices. Default is None.
            missing_data (ndarray): Missing data matrix. Default is None.
            alpha (float): Regularization parameter. Default is None.

        Returns:
            ndarray: Predicted labels.

        Raises:
            ValueError: If an error occurs during computation.

        """
        if self.initializing:
            data = missing_data.copy()
            data[np.isnan(data)] = 0
        else:
            data = missing_data

        cov_x_y = self.cov[label_idx, :][feature_idxes][:, np.newaxis]
        S_observe_fts = self.cov[feature_idxes, :][:, feature_idxes]

        I = np.identity(S_observe_fts.shape[0])
        try:
            cov_inv = np.linalg.inv(S_observe_fts + alpha * I)

            Z = data[rows, :][:, feature_idxes]
            first_term = cov_inv @ cov_x_y
            y_pred = first_term.T @ Z.T
            return y_pred
        except Exception as inst:
            raise ValueError("Error during computation: ", inst)