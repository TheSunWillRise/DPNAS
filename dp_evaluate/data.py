import torch
from torchvision import datasets, transforms
import numpy as np


def get_data(name, augment=False):
    if name == "cifar10":
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        if augment:
            train_transforms = [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(32, 4),
                    transforms.ToTensor(),
                    normalize,
                ]
            print('\n', train_transforms, '\n')
        else:
            train_transforms = [
                transforms.ToTensor(),
                normalize,
            ]

        train_set = datasets.CIFAR10(root="../data", train=True,
                                     transform=transforms.Compose(train_transforms),
                                     download=True)

        test_set = datasets.CIFAR10(root="../data", train=False,
                                    transform=transforms.Compose(
                                        [transforms.ToTensor(), normalize]
                                    ))

    elif name == "fmnist":
        train_set = datasets.FashionMNIST(root='../data', train=True,
                                          transform=transforms.ToTensor(),
                                          download=True)

        test_set = datasets.FashionMNIST(root='../data', train=False,
                                         transform=transforms.ToTensor(),
                                         download=True)

    elif name == "mnist":
        train_set = datasets.MNIST(root='../data/', train=True,
                                   transform=transforms.ToTensor(),
                                   download=True)

        test_set = datasets.MNIST(root='../data', train=False,
                                  transform=transforms.ToTensor(),
                                  download=True)

    else:
        raise ValueError(f"unknown dataset {name}")

    return train_set, test_set


class PoissonSampler(torch.utils.data.Sampler):
    def __init__(self, num_examples, batch_size):
        self.inds = np.arange(num_examples)
        self.batch_size = batch_size
        self.num_batches = int(np.ceil(num_examples / batch_size))
        self.sample_rate = self.batch_size / (1.0 * num_examples)
        super().__init__(None)

    def __iter__(self):
        # select each data point independently with probability `sample_rate`
        for i in range(self.num_batches):
            batch_idxs = np.random.binomial(n=1, p=self.sample_rate, size=len(self.inds))
            # if sample number not equal to batch size, privacy engine will report error
            while np.sum(batch_idxs) != self.batch_size:
                batch_idxs = np.random.binomial(n=1, p=self.sample_rate, size=len(self.inds))
            batch = self.inds[batch_idxs.astype(np.bool)]
            np.random.shuffle(batch)
            yield batch

    def __len__(self):
        return self.num_batches


def random_sample_loader(loader, device, drop_last=False, sample_batches=False):
    datas = []
    targets = []

    for (data, target) in loader:
        data, target = data.to(device), target.to(device)
        datas.append(data)
        targets.append(target)

    datas = torch.cat(datas, axis=0)
    targets = torch.cat(targets, axis=0)
    data = torch.utils.data.TensorDataset(datas, targets)
    if sample_batches:
        sampler = PoissonSampler(len(datas), loader.batch_size)
        return torch.utils.data.DataLoader(data, batch_sampler=sampler,
                                           num_workers=0, pin_memory=False)
    else:
        shuffle = isinstance(loader.sampler, torch.utils.data.RandomSampler)
        return torch.utils.data.DataLoader(data,
                                           batch_size=loader.batch_size,
                                           shuffle=shuffle,
                                           num_workers=0,
                                           pin_memory=False,
                                           drop_last=drop_last)

