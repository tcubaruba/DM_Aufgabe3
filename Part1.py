import numpy as np

# AUFGABE 3-1
print("*"*15, "AUFGABE 3-1", "*"*15, "\n")
x = np.array([[-3, -2], [-2, -1], [-1, 0], [0, 1], [1, 2], [2, 3], [-2, -2], [-1, -1], [0, 0], [1, 1], [2, 2], [-2, -3],
              [-1, -2], [0, -1], [1, 0], [2, 1], [3, 2]]).T

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


# AUFGABE 3-2
print("\n", "*"*15, "AUFGABE 3-2", "*"*15, "\n")

# Conduct a principal axis transformation on the following data set:
x = np.array([(1, 0, 3), (0, 0, 3), (1, 0, 1), (0, 0, 1)]).T

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