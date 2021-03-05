from enum import Enum
import pandas as pd
import cv2
from torch.utils.data import DataLoader, Dataset
import torch
import torchvision

class FruitType(Enum):
    APPLE=1
    MANGO=2
    ALMOND=3

def read_data(FruitType):
    PATH="data/acfr-fruit-dataset/apples"
    training_set_names = pd.read_csv(f"{PATH}/sets/train.txt", names=["image_id"])
    test_set_names = pd.read_csv(f"{PATH}/sets/test.txt", names=["image_id"])
    val_set_names = pd.read_csv(f"{PATH}/sets/val.txt", names=["image_id"])
    train_val_set_names = pd.read_csv(f"{PATH}/sets/train_val.txt", names=["image_id"])

    return (training_set_names, test_set_names, val_set_names, train_val_set_names)

def read_image_annotations(filename):
    PATH = "data/acfr-fruit-dataset/apples"
    filename_annotations = PATH + '/annotations/' + filename + '.csv'
    filename_image = PATH + '/images/' + filename + '.png'

    image = cv2.imread(filename_image, cv2.IMREAD_COLOR)
    annotations = pd.read_csv(filename_annotations)

    return image, annotations

class FruitDataset(Dataset):
    def __init__(self, data, PATH):
        super().__init__()

        self.data = data
        self.PATH = PATH

    def __getitem__(self, index):
        image_id= self.data['image_id'][index]
        print(image_id)
        filename_image = self.PATH + '/images/' + image_id + '.png'
        filename_annotations = self.PATH + '/annotations/' + image_id + '.csv'

        image = cv2.imread(filename_image, cv2.IMREAD_COLOR)
        annotations = pd.read_csv(filename_annotations)

        #Create boxes
        boxes = 0

        labels = torch.ones((self.data.shape[0],), dtype=torch.int64)
        isfruit = torch.zeros((self.data.shape[0],), dtype=torch.int64)

        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['image_id'] = torch.tensor([index])

        return image, target, image_id

    def __len__(self):
        return self.data.shape[0]

def collate_fn(batch):
    return tuple(zip(*batch))

PATH = "data/acfr-fruit-dataset/apples"
training_data = read_data(FruitType.APPLE)[0]
train_dataset = FruitDataset(training_data, PATH)

train_data_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=4,
    collate_fn = collate_fn
)

print(train_data_loader)