# Create by lcquan@nwsuaf.edu.cn on 2022/4/15
from load_dataset_and_preprocess import *
from torch.nn import Linear
from torch.optim import AdamW
from bdm import *
import time
import argparse

import warnings
warnings.filterwarnings("ignore")

if __name__ == "__main__":
    time_start = time.time()
    # -------------------load parameters start-------------------------------
    parser = argparse.ArgumentParser(description='load parameter for running and evaluating \
                                                 expand_by_cluter_contrastive algorithms')
    parser.add_argument('--dataset_name', '-d', type=str, default='lasftm-asia',
                        help='Data set to be used')
    parser.add_argument('--community_index', '-c', type=int, default=0,
                        help='Index of community (label) used in evaluating')
    parser.add_argument('--sample_seed', '-s', type=int, default=1,
                        help='random seed for sample seedset from community')
    parser.add_argument('--seed_nodes_num', '-n', type=int, default=5,
                        help='Number of community members to be used as seedset')
    parser.add_argument('--test_pct', '-t', type=float, default=1.00,
                        help='Percentage of non-seed members to be used as test set')
    parser.add_argument('--use_high_degree', '-u', type=str, default="False",
                        help='Flag to indicate using highest degree members as seeds or not')
    parser.add_argument('--result_path', '-r', type=str, default='./result/',
                        help='Path of result file')
    args = parser.parse_args()

    dataset_name = args.dataset_name
    community_index = args.community_index
    sample_seed = args.sample_seed
    seed_nodes_num = args.seed_nodes_num
    test_pct = args.test_pct
    use_high_degree = (args.use_high_degree == "True")
    result_path = args.result_path
    recall_result_path = result_path + "deep-metric/" + "metric_" + dataset_name + str(seed_nodes_num) + "_recall.txt"
    in_ego_result_path = result_path + "deep-metric/" + "metric_" + dataset_name + str(seed_nodes_num) +  "_in_ego.txt"
    out_ego_result_path = result_path + "deep-metric/" + "metric_" + dataset_name + str(seed_nodes_num) +  "_out_ego.txt"
    loss_result_path = result_path + "deep-metric/" + "metric_" + dataset_name + str(seed_nodes_num) +  "_loss.txt"
    if use_high_degree:
        recall_result_path = result_path + "deep-metric/use_high_degree/" + "metric_" + dataset_name + str(seed_nodes_num) +  "_recall.txt"
        in_ego_result_path = result_path + "deep-metric/use_high_degree/" + "metric_" + dataset_name + str(seed_nodes_num) +  "_in_ego.txt"
        out_ego_result_path = result_path + "deep-metric/use_high_degree/" + "metric_" + dataset_name + str(seed_nodes_num) +  "_out_ego.txt"
        loss_result_path = result_path + "deep-metric/use_high_degree/" + "metric_" + dataset_name + str(seed_nodes_num) +  "_loss.txt"
    # -------------------load parameters end---------------------------------

    # use CUDA_VISIBLE_DEVICES to select gpu
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # load data

    data = load_dataset(dataset_name)
    # transform multi-labels into positive-negative
    data = make_pu_dataset(data, use_high_degree=use_high_degree, pos_index=[community_index], sample_seed=sample_seed,
                           seed_nodes_num=seed_nodes_num, test_pct=test_pct)
    # permanently move in gpy memory
    data = data.to(device)

    # prepare transforms
    drop_edge_p_1, drop_feat_p_1, drop_edge_p_2, drop_feat_p_2 = agmt_dict[dataset_name]
    transform_1 = get_graph_drop_transform(drop_edge_p_1, drop_feat_p_1)
    transform_2 = get_graph_drop_transform(drop_edge_p_2, drop_feat_p_2)

    # build networks
    input_size = data.x.size(1)
    encoder = GCN([input_size] + [32,16], batchnorm=True)   # 32, 16, 32
    predictor = MLP_Predictor(16, 16, 32)
    discriminator = Linear(16, 2, bias=True)
    model = BDM(encoder, predictor, discriminator).to(device)

    # optimizer
    optimizer = AdamW(model.trainable_parameters(), lr=5e-4, weight_decay=1e-5)

    # scheduler
    lr_scheduler = CosineDecayScheduler(5e-4, 1000, 10000)
    mm_scheduler = CosineDecayScheduler(1 - 0.99, 0, 10000)

    tofind_k = int((data.y.sum(0) - data.train_mask.sum(0)) * 0.1)
    seed_nodes = data.train_seed_nodes
    seed_node_set = set(seed_nodes.cpu().tolist())
    cmty_members = data.y.nonzero(as_tuple=False).view(-1)  # only care about positive nodes
    cmty_members_set = set(cmty_members.cpu().tolist())
    cmty_remain_members_set = cmty_members_set - seed_node_set

    def train(step):
        model.train()

        # update learning rate
        lr = lr_scheduler.get(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # update momentum
        mm = 1 - mm_scheduler.get(step)

        # forward
        optimizer.zero_grad()

        x1, x2 = transform_1(data), transform_2(data)

        q1, y2 = model(x1, x2)
        q2, y1 = model(x2, x1)

        # self contrastive loss
        self_loss = 2 - cosine_similarity(q1, y2.detach(), dim=-1).mean() - cosine_similarity(q2, y1.detach(), dim=-1).mean()

        # clustering loss
        cluster_loss1 = compute_cluster_loss(q1, data.train_seed_nodes)
        cluster_loss2 = compute_cluster_loss(q2, data.train_seed_nodes)
        cluster_loss = cluster_loss1 + cluster_loss2

        if self_loss > cluster_loss/2:
            loss = self_loss
        else:
            loss = cluster_loss/2

        #loss = self_loss + cluster_loss/2
        loss.backward()

        # update online network
        optimizer.step()
        # update target network
        model.update_target_network(mm)

    def eval(epoch):
        model.eval()
        x1, x2 = transform_1(data), transform_2(data)
        q1, y2 = model(x1, x2)
        q2, y1 = model(x2, x1)
        return find_k_nodes_eval(q1, q2, tofind_k, seed_nodes, cmty_remain_members_set)

    # train and test
    for epoch in range(1, 10001):
        train(epoch-1)
    rcls = eval(epoch)

    prefex = "c_idx{},s_s{},".format(community_index,sample_seed)
    # print(prefex + rcls.__str__()[1:-1])
    f = open(recall_result_path, 'a+')
    # head_row = "cmt_index, sample_seed, int_ext_rt, seed_int_rt, val_acc, \
    #           k_rcl, 2k_rcl, 3k_rcl, 4k_rcl, 5k_rcl, 6k_rcl, 7k_rcl"
    f.write(prefex + rcls.__str__()[1:-1])
    f.write("\n")
    f.close()
    time_end = time.time()
    print('time cost', time_end - time_start, 's')
