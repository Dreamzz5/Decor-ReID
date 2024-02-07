from sklearn.cluster import KMeans, SpectralClustering
import numpy as np
import torch
import time
def spectral_clustering(similarity_matrix,n_cluster=8):
    W = similarity_matrix
    
    sz = W.shape[0]
    end = time.time()
    # np.random.seed(21)
    sp = SpectralClustering(n_clusters=n_cluster,affinity='precomputed',eigen_solver='amg',random_state=21)
    # print(time.time()-end)
    y_pred = torch.tensor(sp.fit_predict(W))
    del W
    ground_true_matrix = torch.zeros((sz,sz))
    loss_mask_num = []
    loss_mask_den = []
    for i in range(n_cluster):
        idx = torch.where(y_pred==i)[0].unsqueeze(1)
        cur_mask_num = torch.zeros(sz,sz)
        cur_mask_den = torch.zeros(sz,sz)
        idx_i = idx.repeat((len(idx),1))
        idx_j = idx.repeat((1,len(idx))).reshape(-1)
        ground_true_matrix[idx_j, idx_i] = 1
        cur_mask_num[idx_j, idx_i] = 1
        cur_mask_den[idx] = 1
        # for j in idx:
        #     ground_true_matrix[j][idx] = 1
        #     cur_mask_num[j][idx] = 1
        #     cur_mask_den[j][:] = 1
        loss_mask_num.append(cur_mask_num)
        loss_mask_den.append(cur_mask_den)
    loss_mask_num = torch.stack(loss_mask_num)
    loss_mask_den = torch.stack(loss_mask_den)
    return y_pred, ground_true_matrix.float().cuda(), loss_mask_num.float().cuda(), loss_mask_den.float().cuda()


def kmeans_clustering(x, n_cluster=8):

    sz = x.shape[0]
    end = time.time()
    # np.random.seed(21)
    sp = KMeans(n_clusters=n_cluster, random_state=21)
    # print(time.time()-end)
    y_pred = torch.tensor(sp.fit_predict(x))
    # print(torch.unique(y_pred))
    return y_pred



if __name__ == '__main__':
    import torch

    similarity_matrix = torch.rand(512, 512)
    spectral_clustering(similarity_matrix)