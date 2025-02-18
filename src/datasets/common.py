import os
import torch
import json
import glob
import collections
import random
from PIL import Image
from typing import Dict, Optional, Tuple, Callable, List, Union, cast
import numpy as np
from tqdm import tqdm
import math

import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader, Sampler



def get_balanced_data_incremental_subset_indices(dataset, n_splits, split_idx):
    n_classes = torch.unique(torch.tensor(dataset.targets)).shape[0]
    
    def get_subset_indices(X_dataset):
        subset_indices = []
        
        for c in range(n_classes):
            mask = [_c == c for _c in X_dataset.targets]
            samples_from_c = torch.tensor(mask).nonzero().flatten()
            start_idx = int(split_idx * len(samples_from_c) / n_splits)
            end_idx = int((split_idx + 1) * len(samples_from_c) / n_splits)
            subset_indices.append(samples_from_c[start_idx : end_idx])
            
        return subset_indices
    
    train_subset_indices = get_subset_indices(dataset.train_dataset)
    test_subset_indices = get_subset_indices(dataset.test_dataset)
    
    return train_subset_indices, test_subset_indices

"""
train_mask: 对应训练数据集的每个样本，如果该样本的类别在当前划分的 classes 中，则标记为 True，否则为 False。
test_mask: 对应测试数据集的每个样本，同样是根据样本类别是否在当前划分的 classes 中，来决定该样本是否属于当前划分。
train_mask 和 test_mask 是布尔列表，nonzero() 方法返回值为 True（即当前划分类别的样本）的索引。
.flatten() 将索引展平成一维张量，方便后续操作。
其实也就是按照给定的classes_order来对数据进行划分，现在我需要加入noniid
"""

#
def get_class_incremental_classes_and_subset_indices(dataset, n_splits, split_idx):
    n_classes = torch.unique(torch.tensor(dataset.train_dataset.targets)).shape[0]
    # assert n_classes % n_splits == 0
    assert 0 <= split_idx < n_splits
    #这里是对应着cifar100中的classes
    start_class_idx = math.floor(n_classes / n_splits * split_idx)
    end_class_idx = math.floor(n_classes / n_splits * (split_idx+1))
    class_order = dataset.default_class_order if \
            hasattr(dataset, 'default_class_order') else \
                list(range(n_classes))
    classes = sorted(class_order[start_class_idx:end_class_idx])
    """
    train_mask 是一个布尔列表，表示训练集中的每个样本的标签是否在当前任务的类别 classes 中。
    test_mask 也类似，表示测试集中的每个样本的标签是否在当前任务的类别中。
    """
    train_mask = [c in classes for c in dataset.train_dataset.targets]
    test_mask = [c in classes for c in dataset.test_dataset.targets]

    """
    使用 train_mask 和 test_mask，找到当前任务所需的训练集和测试集的索引。nonzero() 返回所有非零元素（即 True 值）的索引，
    然后通过 flatten() 将其转化为一维张量，得到当前分割任务所需的样本索引。
    """
    train_subset_indices = torch.tensor(train_mask).nonzero().flatten()
    test_subset_indices = torch.tensor(test_mask).nonzero().flatten()
        
    return classes, train_subset_indices, test_subset_indices

    
class SubsetSampler(Sampler):
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (i for i in self.indices)

    def __len__(self):
        return len(self.indices)

class ImageFolderWithPaths(datasets.ImageFolder):
    def __init__(self, path, transform, flip_label_prob=0.0):
        super().__init__(path, transform)
        self.flip_label_prob = flip_label_prob
        if self.flip_label_prob > 0:
            print(f'Flipping labels with probability {self.flip_label_prob}')
            num_classes = len(self.classes)
            for i in range(len(self.samples)):
                if random.random() < self.flip_label_prob:
                    new_label = random.randint(0, num_classes-1)
                    self.samples[i] = (
                        self.samples[i][0],
                        new_label
                    )

    def __getitem__(self, index):
        image, label = super(ImageFolderWithPaths, self).__getitem__(index)
        return {
            'images': image,
            'labels': label,
            'image_paths': self.samples[index][0]
        }


def maybe_dictionarize(batch):
    if isinstance(batch, dict):
        return batch

    if len(batch) == 2:
        batch = {'images': batch[0], 'labels': batch[1]}
    elif len(batch) == 3:
        batch = {'images': batch[0], 'labels': batch[1], 'metadata': batch[2]}
    else:
        raise ValueError(f'Unexpected number of elements: {len(batch)}')

    return batch


def get_features_helper(image_encoder, dataloader, device):
    all_data = collections.defaultdict(list)

    image_encoder = image_encoder.to(device)
    image_encoder = torch.nn.DataParallel(image_encoder, device_ids=[x for x in range(torch.cuda.device_count())])
    image_encoder.eval()

    with torch.no_grad():
        for batch in tqdm(dataloader):
            batch = maybe_dictionarize(batch)
            features = image_encoder(batch['images'].cuda())

            all_data['features'].append(features.cpu())

            for key, val in batch.items():
                if key == 'images':
                    continue
                if hasattr(val, 'cpu'):
                    val = val.cpu()
                    all_data[key].append(val)
                else:
                    all_data[key].extend(val)

    for key, val in all_data.items():
        if torch.is_tensor(val[0]):
            all_data[key] = torch.cat(val).numpy()

    return all_data


def get_features(is_train, image_encoder, dataset, device):
    split = 'train' if is_train else 'val'
    dname = type(dataset).__name__
    if image_encoder.cache_dir is not None:
        cache_dir = f'{image_encoder.cache_dir}/{dname}/{split}'
        cached_files = glob.glob(f'{cache_dir}/*')
    if image_encoder.cache_dir is not None and len(cached_files) > 0:
        print(f'Getting features from {cache_dir}')
        data = {}
        for cached_file in cached_files:
            name = os.path.splitext(os.path.basename(cached_file))[0]
            data[name] = torch.load(cached_file)
    else:
        print(f'Did not find cached features at {cache_dir}. Building from scratch.')
        loader = dataset.train_loader if is_train else dataset.test_loader
        data = get_features_helper(image_encoder, loader, device)
        if image_encoder.cache_dir is None:
            print('Not caching because no cache directory was passed.')
        else:
            os.makedirs(cache_dir, exist_ok=True)
            print(f'Caching data at {cache_dir}')
            for name, val in data.items():
                torch.save(val, f'{cache_dir}/{name}.pt')
    return data


class FeatureDataset(Dataset):
    def __init__(self, is_train, image_encoder, dataset, device):
        self.data = get_features(is_train, image_encoder, dataset, device)

    def __len__(self):
        return len(self.data['features'])

    def __getitem__(self, idx):
        data = {k: v[idx] for k, v in self.data.items()}
        data['features'] = torch.from_numpy(data['features']).float()
        return data


def get_dataloader(dataset, is_train, args, image_encoder=None):
    if image_encoder is not None:
        feature_dataset = FeatureDataset(is_train, image_encoder, dataset, args.device)
        dataloader = DataLoader(feature_dataset, batch_size=args.batch_size, shuffle=is_train)
    else:
        dataloader = dataset.train_loader if is_train else dataset.test_loader
    return dataloader
