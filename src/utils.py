import math
from typing import *

import numpy as np


def normalize(
    X: np.ndarray,
    Xtrain: Optional[np.ndarray] = None,
    mean: Optional[np.ndarray] = None,
    std: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Normalize input data by subtracting the mean and dividing by the standard deviation.

    Args:
        X (ndarray): Input data array to be normalized.
        Xtrain (ndarray, optional): Training data array used to compute mean and std. Defaults to None.
        mean (ndarray, optional): Mean value to be subtracted from the data. Defaults to None.
        std (ndarray, optional): Standard deviation value to divide the data by. Defaults to None.

    Returns:
        tuple: A tuple containing the normalized input data, the mean value used for normalization, and the 
               standard deviation value used for normalization.

    Raises:
        ValueError: If the Xtrain array is not provided and the mean and std values are not provided.

    Notes:
        - If the Xtrain array is provided, the mean and std values are computed from it.
        - If the mean and std values are provided, they are used directly.
        - If the std value is zero, the corresponding column is set to zero in the output data.
        - If the input data has any NaN or infinite values, they are replaced with zeros in the output data.
        - If the input data originally had NaN values, they are restored in the output data.
    """

    na_mask = np.isnan(X)

    if mean is None and std is None:
        mean = np.nanmean(Xtrain, axis=0).reshape(1, -1)
        std = np.nanstd(Xtrain, axis=0).reshape(1, -1)

    sd_equal_zero_mask = np.where(std == 0)[0]
    centered_ms = X - mean
    xnorm = np.divide(centered_ms, std, out=np.zeros_like(X), where=std != 0)

    xnorm[np.isnan(xnorm)] = 0
    xnorm[~np.isfinite(xnorm)] = 0

    xnorm[na_mask] = np.nan
    return xnorm, mean, std


def rescale(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """
    Rescale the input array X using the mean and standard deviation (std) parameters.

    Args:
        X (numpy.ndarray): The input array to be rescaled.
        mean (numpy.ndarray): The mean of the data (usually taken from the training set).
        std (numpy.ndarray): The standard deviation of the data (usually taken from the training set).

    Returns:
        numpy.ndarray: The rescaled version of the input array X.
    """
    return (X * std) + mean


def create_image_monotone_missing(
        data: np.ndarray, perc_del: float, perc_width: float,
        perc_height: float, im_width: int,
        im_height: int) -> Tuple[np.ndarray, np.ndarray]:
    """    
    Creates a monotone missing pattern in an image dataset by removing a section of the image from the bottom right corner.

    Args:
        data: A numpy ndarray of shape (n, p) containing the image dataset.
        perc_del: A float representing the percentage of rows to remove from the dataset.
        perc_width: A float representing the percentage of the width of the image to remove.
        perc_height: A float representing the percentage of the height of the image to remove.
        im_width: An integer representing the width of the image.
        im_height: An integer representing the height of the image.

    Returns:
        A tuple containing:
        - A numpy ndarray of shape (n, p) representing the dataset with missing values.
        - A numpy ndarray of shape (m,) containing the indices of the rows with missing values.
    """

    n = data.shape[0]
    m = im_width
    p = im_width * im_height

    # position (from width and height) of pixel would be deleted
    from_width = math.ceil((1 - perc_width) * im_width) - 1
    from_height = math.ceil((1 - perc_width) * im_width) - 1

    nan_rows = np.unique(np.sort(np.random.randint(0, n, int(n * perc_del))))
    nan_rows = nan_rows[:, np.newaxis]

    col_idxs = np.arange(p).reshape(-1, m)

    filter_height = np.arange(from_height, im_height)
    filter_width = np.arange(from_width, im_width)

    col_idxs = col_idxs[:, filter_width][filter_height, :].reshape(-1)

    # flatten row removed
    missing_data = data.copy().astype('float')
    missing_data[nan_rows, col_idxs] = np.nan

    return (missing_data.reshape(n, p), nan_rows.ravel())


def create_randomly_missing(data: np.ndarray, perc_del: float) -> np.ndarray:
    """
    Creates a randomly missing mask for the input data.

    Args:
        data (np.ndarray): The input data.
        perc_del (float): The percentage of missing values to create.

    Returns:
        np.ndarray: An array with the same shape as `data` where missing values are marked as NaN.
    """
    n = data.shape[0]
    # Flatten data into 1 row
    flatten_data = data.reshape(1, -1)
    # Uniform missing mask
    missing_mask = np.random.uniform(0, 1,
                                     flatten_data.shape[1]).reshape(1, -1)
    # Mark as missing if value in mask  < perc_del
    missing_data = flatten_data.copy().astype('float')
    missing_data[missing_mask <= perc_del] = np.nan

    return missing_data.reshape(n, -1)


def rmse_loss(a: np.ndarray, b: np.ndarray) -> float:
    """
    Calculate the root mean squared error (RMSE) between two arrays.

    Args:
        a (np.ndarray): The first array.
        b (np.ndarray): The second array.

    Returns:
        float: The RMSE between the two arrays.
    """
    subtracted = a - b
    nan_mask = np.isnan(subtracted)
    subtracted[nan_mask] = 0
    numerator = np.sum(subtracted**2)

    denominator_m = np.ones_like(a)
    denominator_m[nan_mask] = 0
    denominator = np.sum(denominator_m)

    rmse = np.sqrt(numerator / float(denominator))
    return rmse


def find_largest_elements(s_missing_fts, s_avai_fts, arr, m):
    """
    Returns a boolean array indicating whether each element in arr is one of the m largest elements.

    Args:
        s_missing_fts (np.ndarray): A boolean array indicating which features are missing.
        s_avai_fts (np.ndarray): A boolean array indicating which features are available.
        arr (List[float]): A list of floats representing feature values.
        m (int): The number of largest elements to include in the boolean array.

    Returns:
        np.ndarray: A boolean array indicating whether each element in arr is one of the m largest elements.    
    """

    # import pdb; pdb.set_trace()
    arr[s_missing_fts] = -np.inf
    arr[~s_avai_fts] = -np.inf

    # Get the indices of the m largest elements
    indices_sorted_descending = np.argsort(arr)[::-1]
    largest_element_indices = indices_sorted_descending[:m]

    # Create a boolean array indicating whether each element in arr is one of the m largest elements
    is_largest = np.zeros_like(arr, dtype=bool)
    is_largest[largest_element_indices] = True
    is_largest[~s_avai_fts] = False
    return is_largest
