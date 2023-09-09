import torch
def hypergraph_propagation(X, H):
    n_edge = H.shape[1]
    # the weight of the hyperedge
    W = torch.ones(n_edge, device='cuda:0')
    X=X.float()
    W=W.float()
    H=H.float()
    # the degree of the node
    DV = torch.sum(torch.matmul(H, W.unsqueeze(1)), dim=1)
    # the degree of the hyperedge
    DE = torch.sum(H, dim=0)

    DE2 = torch.diag(torch.pow(DE, -0.5))  # DE^(-1/2)
    DV2 = torch.diag(torch.pow(DV, -0.5))  # DV^(-1/2)
    W = torch.diag(W)
    HT = H.t()

    # HT = DE2 * HT * DV2
    HT = torch.matmul(DE2, torch.matmul(HT, DV2))
    # H = DV2 * H * W * DE2
    H = torch.matmul(DV2, torch.matmul(H, torch.matmul(W, DE2)))

    E = HT.matmul(X)
    # X = H.matmul(E)
    return E