import sys
import argparse

import numpy as np
from PIL import Image
from sklearn.neighbors import NearestNeighbors
from scipy import stats
from sklearn import metrics
from tqdm import tqdm

import torch
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torchvision.datasets import ImageFolder

from sklearn.manifold import TSNE
from matplotlib import pyplot as plt

from dataloader import SiameseDataset, TripletDataset
from network import SiameseNet, TripletNet
from loss import ContrastiveLoss, TripletLoss


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--dims', type=int, default=32)
    parser.add_argument('--batch_size', type=int, default=32)
    return parser.parse_args()


def fit(train_loader, test_loader, model, criterion, optimizer, scheduler, n_epochs, cuda):
    for epoch in range(1, n_epochs + 1):
        scheduler.step()

        train_loss = train_epoch(train_loader, model, criterion, optimizer, cuda)
        print('Epoch: {}/{}, Average train loss: {:.4f}'.format(epoch, n_epochs, train_loss))

        accuracy = test_epoch(test_loader, model, cuda)
        print('Epoch: {}/{}, Accuracy: {:.4f}'.format(epoch, n_epochs, accuracy))


def train_epoch(train_loader, model, criterion, optimizer, cuda):
    model.train()
    losses = []

    for batch_idx, data in tqdm(enumerate(train_loader), total=len(train_loader), desc='Training', file=sys.stdout):
        # (sample1, sample2), target = data
        (anchor, positive, negative), _ = data
        if cuda:
            #    sample1 = sample1.cuda()
            #    sample2 = sample2.cuda()
            #    target = target.cuda()
            anchor = anchor.cuda()
            positive = positive.cuda()
            negative = negative.cuda()

        optimizer.zero_grad()
        # output1, output2 = model(sample1, sample2)
        output1, output2, output3 = model(anchor, positive, negative)

        # loss = criterion(output1, output2, target)
        loss = criterion(output1, output2, output3)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    return np.mean(losses)


def test_epoch(test_loader, model, cuda):
    model.eval()

    embeddings, actual = extract_embeddings(test_loader, model, cuda)

    nbrs = NearestNeighbors(n_neighbors=5, n_jobs=4).fit(embeddings)
    distances, indices = nbrs.kneighbors(embeddings)
    nearest = np.empty(shape=indices.shape)
    for x in range(indices.shape[0]):
        for y in range(indices.shape[1]):
            nearest[x, y] = test_loader.dataset.targets[indices[x, y]]

    # TODO: remove skipping first index when using test dataset
    predicted, _ = stats.mode(nearest[:, 1:], axis=1)
    accuracy = metrics.accuracy_score(actual, predicted)

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
            targets.append(target.reshape([-1, 1]))
    embeddings = np.vstack(embeddings)
    targets = np.vstack(targets)

    return embeddings, targets


def plot_embeddings(dataset, embeddings, targets):
    embeddings = TSNE(n_components=2).fit_transform(embeddings)
    for cls in np.random.choice(dataset.classes, 10):
        i = dataset.class_to_idx[cls]
        inds = np.where(targets == i)[0]
        plt.scatter(embeddings[inds, 0], embeddings[inds, 1], alpha=0.5)
    plt.legend(dataset.classes)
    plt.savefig('embeddings.png')


def main():
    args = parse_args()

    cuda = torch.cuda.is_available()
    if cuda:
        print('Device: {}'.format(torch.cuda.get_device_name(0)))

    train_transform = transforms.Compose([
        transforms.Resize(224, interpolation=Image.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    valid_transform = transforms.Compose([
        transforms.Resize(224, interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    #train_set = SiameseDataset(args.dataset_dir, train_transform)
    train_set = TripletDataset(args.dataset_dir, train_transform)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4)
    print(train_set)

    # TODO: use test dataset
    test_set = ImageFolder(args.dataset_dir, transform=valid_transform)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=4)
    print(test_set)

    #model = SiameseNet(args.dims)
    model = TripletNet(args.dims)
    if cuda:
        model = model.cuda()

    #criterion = ContrastiveLoss(margin=1.)
    criterion = TripletLoss(margin=1.)
    optimizer = Adam(model.parameters(), lr=1e-4)
    scheduler = StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)

    fit(train_loader, test_loader, model, criterion, optimizer, scheduler, args.epochs, cuda)

    embeddings, targets = extract_embeddings(test_loader, model, cuda)
    plot_embeddings(test_set, embeddings, targets)


if __name__ == '__main__':
    main()
