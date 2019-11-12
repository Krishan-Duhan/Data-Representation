import numpy as np
import sys
import pandas as pd
np.set_printoptions(precision=4)

if len(sys.argv) != 5:
    print('usage : ', sys.argv[0], 'data_file labels_file output_vector_file output_reduced_data_file')
    sys.exit()

def scatter1(X, y):
    n_dim, m_dim = X.shape
    df = X.join(pd.Series(y, name='class'))
    class_feature_means = pd.DataFrame(X.columns)
    for c, rows in df.groupby('class'):
        class_feature_means[c] = rows.mean()
    within_class_scatter_matrix = np.zeros((m_dim, m_dim))
    for c, rows in df.groupby('class'):
        rows = rows.drop(['class'], axis=1)
        s = np.zeros((m_dim, m_dim))
        for index, row in rows.iterrows():
            x, mc = row.values.reshape(m_dim, 1), class_feature_means[c].values.reshape(m_dim, 1)
            s += (x - mc).dot((x - mc).T)
            within_class_scatter_matrix += s
    eigen_values, eigen_vectors = np.linalg.eig((within_class_scatter_matrix).dot(within_class_scatter_matrix))
    pairs = [(np.abs(eigen_values[i]), eigen_vectors[:, i]) for i in range(len(eigen_values))]
    pairs = sorted(pairs, key=lambda x: x[0])
    w_matrix = np.hstack((pairs[0][1].reshape(m_dim, 1), pairs[1][1].reshape(m_dim, 1))).real
    np.savetxt(sys.argv[3], w_matrix.T, delimiter=',')
    X_reduced = np.array(X.dot(w_matrix))
    return X_reduced

X = pd.DataFrame(np.loadtxt(sys.argv[1], delimiter=","))
y = np.loadtxt(sys.argv[2], delimiter=",")
np.savetxt(sys.argv[4], scatter1(X, y), delimiter=',')