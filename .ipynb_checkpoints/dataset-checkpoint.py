from torchvision import datasets

class MyDataset(datasets.ImageFolder):
    def __init__(self, root, transform):
        super(MyDataset, self).__init__(root, transform)

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target