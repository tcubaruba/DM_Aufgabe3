from turtledemo.chaos import plot

import numpy as np
import matplotlib.pyplot as plt
x = np.array([[-3,-2], [-2,-1], [-1,0], [0,1], [1,2], [2,3], [-2,-2], [-1,-1], [0 ,0] , [1 ,1] , [2 ,2], [-2,-3],
              [-1,-2], [0,-1], [1 ,0], [2 ,1],[3 ,2]]).T

print(x)

# calculate covariance

M = np.cov(x)
print("Covariance Matrix: \n", M)

# calculate eigenvalues and eigenvectors of M

w, v = np.linalg.eig(M)
print("Eigenvalues: \n ", w)
print("Eigenvectors: \n", v)

# Determine the smallest eigenvalue and remove its corresponding eigenvector

pos = np.argmin(w)
v_new = np.delete(v, pos, axis=0)

print("Remaining eigenvector: \n", v_new)

# Transform all vectors in X in this new sub-space by expressing all vectors in X in this new basis.

x_pca = v_new.dot(x)

print("X reduced: \n", x_pca)