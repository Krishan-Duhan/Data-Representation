import sys
import numpy as np

if len(sys.argv) != 5:
    print('usage : ', sys.argv[0], 'data_file labels_file output_vector_file output_reduced_data_file')
    sys.exit()

data_file = sys.argv[1]
label_file = sys.argv[2]

Xt = np.loadtxt(data_file, delimiter=",")
# print(Xt.shape)   # 150X4 for iris data
X = np.transpose(Xt)
# print(X.shape)
R = np.matmul(X,Xt)
# print(R)
eigenval, eigenvect = np.linalg.eig(R)
# print("eignval unsorted: ", eigenval)
# print("eigenvect unsorted: ", eigenvect)

# Sort the eigenvalues
idx = eigenval.argsort()[::-1]
eigenval = eigenval[idx]
eigenvect = eigenvect[:, idx]
# print("eignval sorted: ", eigenval)
# print("eigenvect sorted: ", eigenvect)

# Pick 1st two max eigen values and corresponding vectors
TopEigenVect = eigenvect[:, :2]
# print(TopEigenVect)

ReducedData = np.matmul(Xt, TopEigenVect)
# print(ReducedData)

np.savetxt(sys.argv[3], np.transpose(TopEigenVect), delimiter=',')
np.savetxt(sys.argv[4], ReducedData, delimiter=',')