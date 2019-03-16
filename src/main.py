import argparse
import sys

import numpy as np
from PIL import Image
from tqdm import tqdm, trange
from sklearn.neighbors import NearestNeighbors
import pandas as pandas
import ml_metrics

import torch
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torchvision.datasets import ImageFolder

from dataloader import SiameseDataset
from network import SiameseNet
from loss import ContrastiveLoss


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--dims', type=int, default=16)
    parser.add_argument('--dim', type=int, default=16)
    parser.add_argument('--batch-size', type=int, default=32)
    return parser.parse_args()


def main():
    args = parse_args()
    print(vars(args))

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

    train_set = SiameseDataset(args.dataset_dir, train_transform)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4)
    print(train_set)

    model = SiameseNet(args.dims)
    if cuda:
        model = model.cuda()

    criterion = ContrastiveLoss(margin=1.)
    optimizer = Adam(model.parameters())
    scheduler = StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)

    table = []

    for epoch in trange(1, args.epochs + 1, desc="Training", file=sys.stdout):
        scheduler.step()
        losses = []
        with tqdm(enumerate(train_loader), desc='Epoch {}/{}'.format(epoch, args.epochs), total=len(train_loader),
                  file=sys.stdout) as pbar:

            for batch_idx, data in pbar:
                (sample1, sample2), target = data
                if cuda:
                    sample1 = sample1.cuda()
                    sample2 = sample2.cuda()
                    target = target.cuda()

                optimizer.zero_grad()
                output1, output2 = model(sample1, sample2)

                loss = criterion(output1, output2, target)
                loss.backward()
                optimizer.step()

                losses.append(loss.item())

                pbar.set_postfix(loss=loss.item())

        table.append([epoch, np.mean(losses)])

    df = pandas.DataFrame(table, columns=['epoch', 'mean_loss'])
    df.set_index('epoch', inplace=True)
    print('\nTraining stats:')
    print(df)

    test_set = ImageFolder(args.dataset_dir, transform=valid_transform)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=4)

    model.eval()
    embeddings = None
    with torch.no_grad():
        for sample, target in tqdm(test_loader, desc='Predict', total=len(test_loader), file=sys.stdout):
            if cuda:
                sample = sample.cuda()

            output = model.get_embedding(sample)

            if embeddings is None:
                embeddings = output.cpu().numpy()
            else:
                embeddings = np.vstack([embeddings, output.cpu().numpy()])

    nbrs = NearestNeighbors(n_neighbors=10, algorithm='ball_tree').fit(embeddings)
    distances, indices = nbrs.kneighbors(embeddings)
    for x in range(indices.shape[0]):
        for y in range(indices.shape[1]):
            indices[x, y] = train_set.targets[indices[x, y]]
    actual = np.array(test_set.targets).reshape([-1, 1])

    # TODO remove skipping first index when using test dataset
    print('K=1:', ml_metrics.mapk(actual, indices[:, 1:], k=1))
    print('K=5:', ml_metrics.mapk(actual, indices[:, 1:], k=5))
    print('K=10:', ml_metrics.mapk(actual, indices[:, 1:], k=10))


if __name__ == '__main__':
    main()
