import torch
from torch.nn.functional import cosine_similarity


def compute_cluster_loss(q, seed_nodes):
    pos_centroid = q[seed_nodes].mean(dim=0)
    pos_loss = 2 - 2*cosine_similarity(q[seed_nodes], pos_centroid, dim=-1).mean()
    return pos_loss


def find_k_nodes_eval(q1, q2, tofind_k, seed_nodes, cmty_remain_members_set):
    pos_centroid1 = q1[seed_nodes].mean(dim=0)
    pos_centroid2 = q2[seed_nodes].mean(dim=0)
    similarities1 = cosine_similarity(q1, pos_centroid1, dim=-1)
    similarities2 = cosine_similarity(q2, pos_centroid2, dim=-1)
    similarities = (similarities1*similarities2)  # /data.node_degrees_without_id

    similarities[seed_nodes] = -float('inf')
    rcls = []
    for i in range(1, 11):
        tk = torch.topk(similarities, k=i*tofind_k, largest=True, dim=0)
        pred = tk.indices.reshape(-1)
        num_true_pos = len(set(pred.cpu().tolist()) & cmty_remain_members_set)
        # rcl = round(num_true_pos / (i * tofind_k), 4)
        rcl = round(num_true_pos / len(cmty_remain_members_set), 4)
        rcls.append(rcl)
    return rcls
