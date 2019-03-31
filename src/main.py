import os
import sys
import argparse

import numpy as np
from PIL import Image
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from tqdm import tqdm

import torch
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

from sklearn.manifold import TSNE
from matplotlib import pyplot as plt

from dataloader import Dataset, BalancedBatchSampler
from network import EmbeddingNet
from loss import OnlineTripletLoss


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, required=True)
    parser.add_argument('--min_images', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--input_size', type=int, default=224)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--dims', type=int, default=32)
    return parser.parse_args()


def fit(train_loader, test_loader, model, criterion, optimizer, scheduler, n_epochs, cuda):
    for epoch in range(1, n_epochs + 1):
        scheduler.step()

        train_loss = train_epoch(train_loader, model, criterion, optimizer, cuda)
        print('Epoch: {}/{}, Average train loss: {:.4f}'.format(epoch, n_epochs, train_loss))

        accuracy = test_epoch(train_loader, test_loader, model, cuda)
        print('Epoch: {}/{}, Accuracy: {:.4f}'.format(epoch, n_epochs, accuracy))


def train_epoch(train_loader, model, criterion, optimizer, cuda):
    model.train()
    losses = []

    for batch_idx, data in tqdm(enumerate(train_loader), total=len(train_loader), desc='Training', file=sys.stdout):
        samples, targets = data
        if cuda:
            samples = samples.cuda()
            targets = targets.cuda()

        optimizer.zero_grad()
        outputs = model(samples)

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    return np.mean(losses)


def test_epoch(train_loader, test_loader, model, cuda):
    model.eval()

    train_embeddings, train_targets = extract_embeddings(train_loader, model, cuda)
    test_embeddings, test_targets = extract_embeddings(test_loader, model, cuda)

    knn = KNeighborsClassifier(n_neighbors=5, n_jobs=4).fit(train_embeddings, train_targets)
    predicted = knn.predict(test_embeddings)
    accuracy = metrics.accuracy_score(test_targets, predicted)

    return accuracy


def extract_embeddings(loader, model, cuda):
    embeddings = []
    targets = []
    with torch.no_grad():
        for sample, target in tqdm(loader, total=len(loader), desc='Testing', file=sys.stdout):
            if cuda:
                sample = sample.cuda()

            output = model.get_embedding(sample)

            embeddings.append(output.cpu().numpy())
            targets.append(target)
    embeddings = np.vstack(embeddings)
    targets = np.concatenate(targets)

    return embeddings, targets


def plot_embeddings(dataset, embeddings, targets, title=''):
    embeddings = TSNE(n_components=2).fit_transform(embeddings)
    plt.figure(figsize=(10, 10))
    for cls in np.random.choice(dataset.classes, 10):
        i = dataset.class_to_idx[cls]
        inds = np.where(targets == i)[0]
        plt.scatter(embeddings[inds, 0], embeddings[inds, 1], alpha=0.5)
    plt.legend(dataset.classes)
    plt.title(title)
    plt.savefig('{}_embeddings.png'.format(title))


def main():
    args = parse_args()
    print(vars(args))

    cuda = torch.cuda.is_available()
    if cuda:
        print('Device: {}'.format(torch.cuda.get_device_name(0)))

    train_transform = transforms.Compose([
        transforms.Resize(args.input_size, interpolation=Image.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    valid_transform = transforms.Compose([
        transforms.Resize(args.input_size, interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    train_set = Dataset(os.path.join(args.dataset_dir, 'train'), train_transform, min_images=args.min_images)
    train_batch_sampler = BalancedBatchSampler(train_set.targets, n_classes=10, n_samples=10)
    train_loader = DataLoader(train_set, batch_sampler=train_batch_sampler, num_workers=4)
    print(train_set)

    test_set = Dataset(os.path.join(args.dataset_dir, 'test'), transform=valid_transform, min_images=args.min_images)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=4)
    print(test_set)

    model = EmbeddingNet(args.dims)
    if cuda:
        model = model.cuda()
    print(model)

    criterion = OnlineTripletLoss(margin=1.0)
    optimizer = Adam(model.parameters(), lr=1e-4)
    scheduler = StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)

    fit(train_loader, test_loader, model, criterion, optimizer, scheduler, args.epochs, cuda)

    train_embeddings, train_targets = extract_embeddings(train_loader, model, cuda)
    plot_embeddings(train_set, train_embeddings, train_targets, title='train')

    test_embeddings, test_targets = extract_embeddings(test_loader, model, cuda)
    plot_embeddings(test_set, test_embeddings, test_targets, title='test')


if __name__ == '__main__':
    main()
