import numpy as np


class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None

    def fit(self, X):
        # mean centering
        self.mean = np.mean(X, axis=0)
        X = X - self.mean

        # covariance (needs samples as columns)
        cov = np.cov(X.T)

        # eigenvalues, eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(cov)

        # -> eigenvector v = [:,i]
        # transpose for easier calculations
        eigenvectors = eigenvectors.T

        # sort eigenvectors

        idxs = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idxs]
        eigenvectors = eigenvectors[idxs]

        # store first n eigenvectors
        self.components = eigenvectors[0 : self.n_components]

    def transform(self, X):
        # project data
        X = X - self.mean
        return np.dot(X, self.components.T)


# This is the sample code for matrix multiplication
def matrixMultiplication(A, B):
    result = []
    for i in range(len(A)):
        row = []
        for j in range(len(B[0])):
            row.append(0)
        result.append(row)
    for i in range(len(A)):
        for j in range(len(B[0])):
            for k in range(len(B)):
                result[i][j] += A[i][k] * B[k][j]
    print(result)
    return result


# Sample code for Dot product of two matrices
def dotProduct(A, B):
    C = [0] * len(B[0])
    for i in range(len(A)):
        for j in range(len(A[0])):
            C[i] += A[i][j] * B[i][j]
    return C


# Sample code for Addition of Two Matrices
def additionMatrix(a, b):
    result = [[0 for x in range(len(a))] for y in range(len(b[0]))]
    for i in range(len(a)):
        for j in range(len(a[0])):
            result[i][j] = a[i][j] + b[i][j]
    print(result)
    return result


# Project the data onto the 1 primary principal components
X = np.random.rand(28, 28)
pca = PCA(1)
pca.fit(X)
X_projected = pca.transform(X)
print(X_projected)
