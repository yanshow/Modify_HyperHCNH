import torch
import torch.nn.functional as F



# L2范数正则化（权重衰减）
def l2_loss(w):
    return torch.sum(w**2)/2
    # return torch.norm(w, p=2)


def l2_normalize(X, dim=1):
    norm_th = torch.nn.functional.normalize(X, p=2, dim=dim, eps=1e-12)
    return norm_th


# 交叉熵损失
def cross_entropy_loss(preds, labels, label_index):
    label_index = label_index.type(torch.int64)
    cross_entropy_loss = torch.nn.CrossEntropyLoss().cuda(0)
    loss = cross_entropy_loss(preds[label_index], labels[label_index].type(torch.int64))
    return loss


# 一致性损失
def consis_loss(logps, temp=0.5):
    ps = [torch.exp(p) for p in logps]
    sum_p = 0.
    for p in ps:
        sum_p = sum_p + p
    avg_p = sum_p / len(ps)
    # p2 = torch.exp(logp2)

    sharp_p = (torch.pow(avg_p, 1. / temp) / torch.sum(torch.pow(avg_p, 1. / temp), dim=1, keepdim=True)).detach()
    loss = 0.
    for p in ps:
        loss += torch.mean((p - sharp_p).pow(2).sum(1))
    loss = loss / len(ps)
    return loss


# 一致性损失
def consis_loss3(H_list, H):
    loss = 0.
    for H_i in H_list:
        loss += torch.sum(torch.pow(torch.sub(H_i, H), 2))
    return loss


# 一致性损失
def consis_loss2(H_list):
    H1 = H_list[0]
    H2 = H_list[1]
    loss = torch.sum(torch.pow(torch.sub(H1, H2), 2))
    # loss = torch.pow(torch.norm(torch.sub(H1, H2), p=2), 2)
    return loss



def accuracy(preds, labels, label_index):
    label_index = label_index.type(torch.int64)
    _, label_preds = torch.max(preds[label_index], dim=1)

    correct_prediction = torch.sum(label_preds.type(torch.int32) == labels[label_index].type(torch.int32))
    accuracy_all = correct_prediction.float() / len(label_preds)
    return accuracy_all


