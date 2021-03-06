import numpy as np
import pandas as pd
import cv2
from torch.utils.data import Dataset
import torch


PATH = "data/acfr-fruit-dataset/almonds"

def read_data():
    training_set_names = pd.read_csv(f"{PATH}/sets/train.txt", names=["image_id"])
    test_set_names = pd.read_csv(f"{PATH}/sets/test.txt", names=["image_id"])
    val_set_names = pd.read_csv(f"{PATH}/sets/val.txt", names=["image_id"])
    train_val_set_names = pd.read_csv(f"{PATH}/sets/train_val.txt", names=["image_id"])
    return (training_set_names, test_set_names, val_set_names, train_val_set_names)

def read_image_annotations(filename):
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
        filename_image = self.PATH + '/images/' + image_id + '.png'
        filename_annotations = self.PATH + '/annotations/' + image_id + '.csv'

        image = cv2.imread(filename_image)
        image = np.moveaxis(image, -1, 0)
        image = torch.from_numpy(image) / 255.0
        annotations = pd.read_csv(filename_annotations)

        area = torch.empty((1,))


        if annotations.empty == False:
            boxes = pd.DataFrame({'x_start': annotations['x'], 'y_start': annotations['y'],
                                'x_end': (annotations['x']+annotations['dx']),
                                'y_end': (annotations['y']+annotations['dy'])})
            boxes = torch.tensor(boxes.values)
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        else:
            boxes = torch.empty((0,4))

        labels = torch.ones((self.data.shape[0],), dtype=torch.int64)
        iscrowd = torch.zeros((self.data.shape[0],), dtype=torch.int64)

        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['image_id'] = torch.tensor([index])
        target['iscrowd'] = iscrowd
        target['area'] = area

        return image, target

    def __len__(self):
        return self.data.shape[0]

def collate_fn(batch):
    return tuple(zip(*batch))