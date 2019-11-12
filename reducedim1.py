from sklearn.decomposition import PCA as sklearnPCA
import sys
import pandas as pd
import numpy as np

if len(sys.argv) != 5:
    print('usage : ', sys.argv[0], 'data_file labels_file output_vector_file output_reduced_data_file')
    sys.exit()

def pca2(X):
    # Data matrix X, assumes 0-centered
    n, m = X.shape
    X -= X.mean(axis=0)
    # Compute covariance matrix
    C = np.dot(X.T, X) / (n - 1)
    # Eigen decomposition
    eigen_vals, eigen_vecs = np.linalg.eig(C)
    # need to select top 2 eigen values and eigen vectors
    idx = np.argsort(eigen_vals)[::-1]  # sort in reverse order
    eigen_vals = eigen_vals[idx]
    eigen_vecs = eigen_vecs[:, idx]
    eigen_vals = eigen_vals[0:2]
    eigen_vecs = eigen_vecs[:, 0:2]
    np.savetxt(sys.argv[3], np.transpose(eigen_vecs), delimiter=',')
    # Project X onto PC space
    X_pca = np.dot(X, eigen_vecs)
    return X_pca

X = pd.DataFrame(np.loadtxt(sys.argv[1], delimiter=","))
np.savetxt(sys.argv[4], pca2(X), delimiter=',')