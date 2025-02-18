from collections import defaultdict
import random
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
import numpy as np
from src.datasets.common import (
    get_balanced_data_incremental_subset_indices,
    get_class_incremental_classes_and_subset_indices,
)
from src.heads import get_classification_head, build_subset_classification_head



# 定义一个全局函数
class TargetTransform:
    def __init__(self, class_map):
        self.class_map = class_map

    def __call__(self, t):
        return self.class_map[t]


def get_dataset_and_classifier_for_split(dataset, split_idx, text_encoder, args, remap_labels=True, return_classifier=True,classnames=None):
    if args.split_strategy == 'data':
        train_subset_indices, test_subset_indices = \
            get_balanced_data_incremental_subset_indices(
                dataset.train_dataset, args.n_splits, split_idx
            )
        dataset.train_dataset = torch.utils.data.Subset(dataset.train_dataset, train_subset_indices)
        # it does not make sense to split test in data-incremental
        # dataset.test_dataset = torch.utils.data.Subset(dataset.test_dataset, test_subset_indices)
        if return_classifier:
            classification_head = get_classification_head(args, args.dataset)
    elif args.split_strategy == 'class':
        classes, train_subset_indices, test_subset_indices = \
            get_class_incremental_classes_and_subset_indices(
                dataset, args.n_splits, split_idx
            )
        #用于从原始数据集中创建一个子集。它需要一个数据集（如 torchvision.datasets 中的任何数据集）和一个索引列表作为参数，返回一个新的子集数据集。
        dataset.train_dataset = Subset(dataset.train_dataset, train_subset_indices)
        dataset.test_dataset = Subset(dataset.test_dataset, test_subset_indices)

        # if remap_labels:
        #     class_map = {c: idx for idx, c in enumerate(sorted(classes))}
        #     dataset.train_dataset.dataset.target_transform = lambda t : class_map[t]
        #     dataset.test_dataset.dataset.target_transform = lambda t : class_map[t]
        # 修改 target_transform 逻辑
        """
        通过 class_map 将数据集中的标签（类别）映射到新的标签索引，常用于数据集划分或标签重新组织的场景。
        例如，在处理不连续的标签时，你可能需要重新为类别编号，以便于模型训练。通过设置 target_transform，
        ***每当加载一个样本时，标签会被自动映射到新的索引***
        """

        #还是在这里对数据集进行non-iid的划分

        # data_indices = dataset.train_dataset.indices
        #
        # y_train = np.array([dataset.train_dataset.targets[idx] for idx in data_indices])  # 提取标签
        # beta,n_parties = 0.4,3
        # # 调用 partition_data 进行 Non-IID 划分
        # net_dataidx_map = partition_data(y_train, beta, n_parties)

        #修改的结束
        if remap_labels:
            class_map = {c: idx for idx, c in enumerate(sorted(classes))}

            # 使用 TargetTransform 实例
            dataset.train_dataset.dataset.target_transform = TargetTransform(class_map)
            dataset.test_dataset.dataset.target_transform = TargetTransform(class_map)

        if return_classifier:
            classification_head = build_subset_classification_head(
                text_encoder.model, args.dataset, classes, args.data_location, args.device,classnames=classnames
            )
    else:
        raise NotImplementedError()
    
    # dataloaders
    dataset.train_loader = torch.utils.data.DataLoader(
        dataset.train_dataset, batch_size=dataset.train_loader.batch_size,
        shuffle=True, num_workers=dataset.train_loader.num_workers
    )        
    dataset.test_loader = torch.utils.data.DataLoader(
        dataset.test_dataset, batch_size=dataset.test_loader.batch_size,
        shuffle=False, num_workers=dataset.test_loader.num_workers
    )

    return (dataset, classification_head) if return_classifier else dataset

def distribution_based_label_skew(y_train, beta=0.4, n_parties=3,n_shot=64,data_cap=1024):
    """
    对训练数据进行 Non-IID 划分
    :param y_train: 标签列表
    :param beta: 划分的非IID程度，0表示完全IID，越大表示越非IID
    :param n_parties: 客户端数量
    :return: 划分后的数据索引
    """
    # 按照标签进行分组
    label_dict = defaultdict(list)
    for idx, label in enumerate(y_train):
        label_dict[label].append(idx)

    # 对每个标签类进行 Non-IID 划分
    client_data = {i: [] for i in range(n_parties)}
    client_val_data = {i: [] for i in range(n_parties)}
    for label, indices in label_dict.items():
        # 按照 beta 划分每一类数据
        num_samples = len(indices)
        # 计算每个客户端分到的样本数
        partitions = np.random.dirichlet([beta] * n_parties) * num_samples
        partitions = [int(p) for p in partitions]

        # 将每一类的数据分配到各个客户端
        start = 0
        for client_id in range(n_parties):
            end = start + partitions[client_id]
            client_data[client_id].extend(indices[start:end])
            client_val_data[client_id].extend(select_random_samples(client_data[client_id],num_samples=n_shot))
            start = end
    for client_id in range(n_parties):
        random.shuffle(client_val_data[client_id])
        # 对于数据量很多的情况，限制最大数据量
        client_val_data[client_id] = client_val_data[client_id][:data_cap]
    return client_data,client_val_data


def select_random_samples(client_ids, num_samples=64):
    # 如果 client_ids 长度大于 num_samples，就随机选取 num_samples 个元素
    if len(client_ids) > num_samples:
        selected_ids = random.sample(client_ids, num_samples)
    else:
        selected_ids = client_ids  # 如果长度小于等于 num_samples，则选择所有
    return selected_ids



def create_non_iid_dataloaders(train_loader, n_parties=3, beta=0.4):
    # 获取标签信息
    y_train = []
    for _, labels in train_loader:
        y_train.extend(labels.cpu().numpy())

    # 使用 Non-IID 划分
    client_data = distribution_based_label_skew(y_train, beta=beta, n_parties=n_parties)

    # 创建每个客户端的 DataLoader
    party_loaders = {}
    for party_id, indices in client_data.items():
        # 从原始数据集中创建子集
        subset = Subset(train_loader.dataset, indices)
        party_loader = DataLoader(subset, batch_size=train_loader.batch_size, shuffle=True,
                                  num_workers=train_loader.num_workers)
        party_loaders[party_id] = party_loader
    # 测试每个 party_loader 的 targets
    # for party_id, loader in party_loaders.items():
    #     party_targets = [y for _, y in loader.dataset]
    #     print(f"Client {party_id}: mapped targets = {sorted(set(party_targets))}")

    return party_loaders


def create_non_iid_dataloaders_with_val(train_loader, n_parties=3, beta=0.4, n_shot=64, val_data_cap=1024):
    # 获取标签信息
    y_train = []
    for _, labels in train_loader:
        y_train.extend(labels.cpu().numpy())

    # 使用 Non-IID 划分
    client_data,client_val_data = distribution_based_label_skew(y_train, beta=beta, n_parties=n_parties)

    # 创建每个客户端的 DataLoader
    party_loaders = {}
    val_loaders = {}  # 用来保存每个客户端的验证数据集

    for party_id, indices in client_data.items():
        # 从原始数据集中创建子集
        subset = Subset(train_loader.dataset, indices)
        party_loader = DataLoader(subset, batch_size=train_loader.batch_size, shuffle=True,
                                  num_workers=train_loader.num_workers)
        party_loaders[party_id] = party_loader

        # 创建每个客户端的验证集（n_shot per class）


        # 测试每个 party_loader 的 targets
        # party_targets = [y for _, y in party_loader.dataset]
        # print(f"Client {party_id}: mapped targets = {sorted(set(party_targets))}")

    for party_id, indices in client_val_data.items():
        # 从原始数据集中创建子集
        subset = Subset(train_loader.dataset, indices)
        val_loader = DataLoader(subset, batch_size=train_loader.batch_size, shuffle=True,
                                  num_workers=train_loader.num_workers)
        val_loaders[party_id] = val_loader

        # 创建每个客户端的验证集（n_shot per class）


        # 测试每个 party_loader 的 targets
        # party_targets_val = [y for _, y in val_loader.dataset]
        # # print(f"Client {party_id}: mapped targets = {sorted(set(party_targets_val))}")
    return party_loaders, val_loaders


