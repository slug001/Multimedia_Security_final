#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import numpy as np
from torchvision import datasets, transforms

def mnist_iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def mnist_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_shards, num_imgs = 200, 300
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.train_labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    idxs = idxs_labels[0,:]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users


def cifar_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


from torch.utils.data import Subset
def cifar_noniid(dataset_label, num_clients, num_classes, q):
    """
    Sample Non-I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    并保证返回的 dict_users 有 0..num_clients-1 连续 key，即使该用户分到零样本也会得到一个空列表。
    """
    # 第一阶段：把样本先按照 label 分到 10 个“组”
    proportion = non_iid_distribution_group(dataset_label, num_clients, num_classes, q)
    # 第二阶段：再根据每个组把样本分配给各 client
    dict_users = non_iid_distribution_client(proportion, num_clients, num_classes)

    # 如果 client_id 少于 num_clients，这里补全那些没有分到样本的 client
    for uid in range(num_clients):
        if uid not in dict_users:
            dict_users[uid] = set()

    return dict_users

def non_iid_distribution_group(dataset_label, num_clients, num_classes, q):
    dict_users, all_idxs = {}, [i for i in range(len(dataset_label))]
    # 第一步：把 dict_users 临时当作“组编号 → 该组样本索引集合”
    for i in range(num_classes):
        dict_users[i] = set([])
    for k in range(num_classes):
        idx_k = np.where(np.array(dataset_label) == k)[0]
        num_idx_k = len(idx_k)
        
        # 从 label=k 的所有样本中，随机抽出 q 比例给组 k
        selected_q_data = set(np.random.choice(idx_k, int(num_idx_k*q) , replace=False))
        dict_users[k] = dict_users[k]|selected_q_data
        idx_k = list(set(idx_k) - selected_q_data)
        all_idxs = list(set(all_idxs) - selected_q_data)
        # 再把剩下的按照 (1-q)/(num_classes-1) 分给不是 k 的那些组
        for other_group in range(num_classes):
            if other_group == k:
                continue
            selected_not_q_data = set(np.random.choice(idx_k, int(num_idx_k*(1-q)/(num_classes-1)) , replace=False))
            dict_users[other_group] = dict_users[other_group]|selected_not_q_data
            idx_k = list(set(idx_k) - selected_not_q_data)
            all_idxs = list(set(all_idxs) - selected_not_q_data)
    print(len(all_idxs),' samples are remained')
    print('random put those samples into groups')
    # 把剩下 all_idxs（无法整除的部分）再平均分配给各组
    num_rem_each_group = len(all_idxs) // num_classes
    for i in range(num_classes):
        selected_rem_data = set(np.random.choice(all_idxs, num_rem_each_group, replace=False))
        dict_users[i] = dict_users[i]|selected_rem_data
        all_idxs = list(set(all_idxs) - selected_rem_data)
    print(len(all_idxs),' samples are remained after relocating')
    return dict_users


def non_iid_distribution_client(group_proportion, num_clients, num_classes):
    """
    将“组 → 样本索引集”分配给 num_clients 个 client，
    client_id 从 0 顺次到 num_clients-1，保证 key 连续。
    """
    num_each_group = num_clients // num_classes
    num_data_each_client = len(group_proportion[0]) // num_each_group
    dict_users = {}
    client_id = 0
    for i in range(num_classes):
        group_data = list(group_proportion[i])
        for j in range(num_each_group):
            # 每个 client 拿 num_data_each_client 张样本
            selected_data = set(np.random.choice(group_data, num_data_each_client, replace=False))
            dict_users[client_id] = selected_data
            group_data = list(set(group_data) - selected_data)
            client_id += 1
    print(len(group_data),' samples are remained')
    return dict_users


def check_data_each_client(dataset_label, client_data_proportion, num_client, num_classes):
    for client in client_data_proportion.keys():
        client_data = dataset_label[list(client_data_proportion[client])]
        print('client', client, 'distribution information:')
        for i in range(num_classes):
            print('class ', i, ':', len(client_data[client_data==i])/len(client_data))


if __name__ == '__main__':
    dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))
                                   ]))
    num = 100
    d = mnist_noniid(dataset_train, num)

    # 如果用脚本单独测试：

    # 模拟 CIFAR-10 labels 长度，比如 50000 张图
    dataset_label = np.random.randint(0, 10, size=50000).tolist()
    num_clients = 20
    num_classes = 10
    q = 0.1

    dict_users = cifar_noniid(dataset_label, num_clients, num_classes, q)
    print("最终 dict_users.keys()：", sorted(dict_users.keys()))
    for uid in range(num_clients):
        print(f"Client {uid} 拿到 {len(dict_users[uid])} 张样本")
