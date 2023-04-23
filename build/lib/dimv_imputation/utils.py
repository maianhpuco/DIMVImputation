import numpy as np 

def normalize(X):
    na_mask = np.isnan(X)
    mean = np.nanmean(X, axis=0).reshape(1, -1)
    sd = np.nanstd(X, axis=0).reshape(1, -1)

    sd_equal_zero_mask = np.where(sd==0)[0]
    centered_ms = X - mean
    xnorm = np.divide(centered_ms, sd, out=np.zeros_like(X), where=sd!=0)


    xnorm[np.isnan(xnorm)]     = 0 
    xnorm[~np.isfinite(xnorm)] = 0 

    xnorm[na_mask] = np.nan
    return xnorm, mean, sd


def rescale(X, mean, std):
    return (X * std) + mean   
