from torchvision.datasets import ImageFolder

import numpy as np

POSITIVE_NEGATIVE_RATIO = 0.5
LENGTH_FACTOR = 0.1


class SiameseDataset(ImageFolder):

    def __init__(self, root, transform):
        super(SiameseDataset, self).__init__(root, transform)
        print('Found {} images belonging to {} classes'.format(len(self), len(self.classes)))

        self.label_to_idxs = {label: np.where(np.array(self.targets) == self.class_to_idx[label])[0] for label in
                              self.classes}

    def __getitem__(self, index):
        # TODO should we use a seed?

        target = np.random.random_sample() < POSITIVE_NEGATIVE_RATIO
        if target == 0:
            siamese_label = self.classes[self.targets[index]]
        else:
            siamese_label = np.random.choice(list(set(self.classes) - {self.classes[self.targets[index]]}))
        siamese_index = np.random.choice(self.label_to_idxs[siamese_label])

        sample1 = self.loader(self.samples[index][0])
        sample2 = self.loader(self.samples[siamese_index][0])
        if self.transform is not None:
            sample1 = self.transform(sample1)
            sample2 = self.transform(sample2)

        return (sample1, sample2), target

    def __len__(self):
        return int(super().__len__() * LENGTH_FACTOR)
