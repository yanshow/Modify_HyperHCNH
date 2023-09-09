import numpy as np

def Eu_dis(x):
    """
    Calculate the distance among each raw of x
    :param x: N X D
           N: the object number
           D: Dimension of the feature
    :return: N X N distance matrix
    """
    x = np.mat(x)
    aa = np.sum(np.multiply(x, x), 1)
    ab = x * x.T
    dist_mat = aa + aa.T - 2 * ab
    dist_mat[dist_mat < 0] = 0
    dist_mat = np.sqrt(dist_mat)
    dist_mat = np.maximum(dist_mat, dist_mat.T)
    return dist_mat

def graph_construct_kNN(X, k_neig=10, is_probH=False, m_prob=1.0):
    """
    param:
        X: N_object x feature_number
        k_neig: the number of neighbor expansion
    return:
        A: N_object x N_object
    """

    dis_mat = Eu_dis(X)
    n_obj = dis_mat.shape[0]
    n_edge = n_obj
    H = np.zeros((n_obj, n_edge))

    for center_idx in range(n_obj):
        dis_mat[center_idx, center_idx] = 0
        dis_vec = dis_mat[center_idx]
        nearest_idx = np.array(np.argsort(dis_vec)).squeeze()
        avg_dis = np.average(dis_vec)
        if not np.any(nearest_idx[:k_neig] == center_idx):
            nearest_idx[k_neig - 1] = center_idx

        for node_idx in nearest_idx[:k_neig]:
            H[node_idx, center_idx] = 1.0

        for node_idx in nearest_idx[:k_neig]:
            if is_probH:
                H[node_idx, center_idx] = np.exp(-dis_vec[0, node_idx] ** 2 / (m_prob * avg_dis) ** 2)
            else:
                H[node_idx, center_idx] = 1.0

    # print("KNN: H的top-k元素数量：", np.count_nonzero(H))
    return H