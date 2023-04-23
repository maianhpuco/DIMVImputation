from dpers import DPERS
import numpy as np 

mean = [0, 0]
cov = [[1, 0.3], [0.3, 1]] 

X = np.random.multivariate_normal(mean, cov, size = 20)


if __name__ == "__main__":
    print(X)
