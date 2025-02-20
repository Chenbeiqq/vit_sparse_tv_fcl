import sys
import inspect
import random
from collections import defaultdict, Counter

import torch
import copy
import numpy  as np
from torch.utils.data.dataset import random_split, Subset

from src.datasets.cars import Cars
from src.datasets.cifar10 import CIFAR10
from src.datasets.cifar100 import CIFAR100
from src.datasets.cub200 import CUB200, CUB200CustomTemplates
from src.datasets.domainnet import DomainNet
from src.datasets.dtd import DTD
from src.datasets.eurosat import EuroSAT, EuroSATVal
from src.datasets.gtsrb import GTSRB
from src.datasets.imagenet import ImageNet
from src.datasets.imagenetr import ImageNetR
from src.datasets.mnist import MNIST
from src.datasets.resisc45 import RESISC45
from src.datasets.stl10 import STL10
from src.datasets.svhn import SVHN
from src.datasets.sun397 import SUN397
from src.datasets.tinyImagenet import TinyImageNet
#导入类名，后续可以直接生成class
registry = {
    name: obj for name, obj in inspect.getmembers(sys.modules[__name__], inspect.isclass)
}


class GenericDataset(object):
    def __init__(self):
        self.train_dataset = None
        self.train_loader = None
        self.test_dataset = None
        self.test_loader = None
        self.classnames = None


def create_k_shot_dataset(dataset, num_shots=64, data_cap=1024):
    dataset = dataset.train_dataset
    targets = dataset.targets if hasattr(dataset, 'targets') else dataset.labels

    # Prepare to collect indices for each class
    class_indices = defaultdict(list)

    # Optimization: Use a counter to keep track of counts per class
    class_counts = Counter()
    # class_counts = {}

    # Collect indices
    for idx, label in enumerate(targets):
        label = label.item() if isinstance(label, torch.Tensor) else label
        if class_counts[label] < num_shots:
            class_indices[label].append(idx)
            class_counts[label] += 1
            # Break early if all classes have enough samples
            # print(class_counts.values())
            # if all(count >= num_shots for count in class_counts.values()):
            #     break

    # Collect exactly num_shots indices for each class
    selected_indices = []
    for label, indices in class_indices.items():
        # 对于每个类别，如果样本数量超过 num_shots，则随机选择 num_shots 个样本。否则，选择所有样本。
        if len(indices) > num_shots:
            selected_indices.extend(np.random.choice(indices, num_shots, replace=False))
        else:
            selected_indices.extend(indices)

    random.shuffle(selected_indices)
    # for tasks with many classes, set a cap for total data for mask training
    selected_indices = selected_indices[:data_cap]

    # Create a new dataset from the selected indices
    subset_dataset = Subset(dataset, selected_indices)
    return subset_dataset


def split_train_into_train_val(dataset, new_dataset_class_name, batch_size, num_workers, val_fraction, max_val_samples=None, seed=0):
    assert val_fraction > 0. and val_fraction < 1.
    total_size = len(dataset.train_dataset)
    val_size = int(total_size * val_fraction)
    if max_val_samples is not None:
        val_size = min(val_size, max_val_samples)
    train_size = total_size - val_size

    assert val_size > 0
    assert train_size > 0

    lengths = [train_size, val_size]

    trainset, valset = random_split(
        dataset.train_dataset,
        lengths,
        generator=torch.Generator().manual_seed(seed)
    )
    if new_dataset_class_name == 'MNISTVal':
        assert trainset.indices[0] == 36044


    new_dataset = None

    new_dataset_class = type(new_dataset_class_name, (GenericDataset, ), {})
    new_dataset = new_dataset_class()

    new_dataset.train_dataset = trainset
    new_dataset.train_loader = torch.utils.data.DataLoader(
        new_dataset.train_dataset,
        shuffle=True,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    new_dataset.test_dataset = valset
    new_dataset.test_loader = torch.utils.data.DataLoader(
        new_dataset.test_dataset,
        batch_size=batch_size,
        num_workers=num_workers
    )

    new_dataset.classnames = copy.copy(dataset.classnames)

    return new_dataset


def get_dataset(dataset_name, preprocess, location, batch_size=128, num_workers=12, val_fraction=0.1, max_val_samples=5000, subset_config=None):
    if dataset_name.endswith('Val'):
        # Handle val splits
        if dataset_name in registry:
            dataset_class = registry[dataset_name]
        else:
            base_dataset_name = dataset_name.split('Val')[0]
            base_dataset = get_dataset(base_dataset_name, preprocess, location, batch_size, num_workers)
            dataset = split_train_into_train_val(
                base_dataset, dataset_name, batch_size, num_workers, val_fraction, max_val_samples)
            return dataset
    else:
        assert dataset_name in registry, f'Unsupported dataset: {dataset_name}. Supported datasets: {list(registry.keys())}'
        dataset_class = registry[dataset_name]
    dataset = dataset_class(
        preprocess, location=location, batch_size=batch_size, num_workers=num_workers, subset_config=subset_config
    )
    return dataset
