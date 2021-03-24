from enum import Enum
import pandas as pd
import cv2
from torch.utils.data import DataLoader, Dataset
import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np

class FruitType(Enum):
    APPLE=1
    MANGO=2
    ALMOND=3

def read_data(FruitType):
    PATH="data/acfr-fruit-dataset/almonds"
    training_set_names = pd.read_csv(f"{PATH}/sets/train.txt", names=["image_id"])
    test_set_names = pd.read_csv(f"{PATH}/sets/test.txt", names=["image_id"])
    val_set_names = pd.read_csv(f"{PATH}/sets/val.txt", names=["image_id"])
    train_val_set_names = pd.read_csv(f"{PATH}/sets/train_val.txt", names=["image_id"])

    return (training_set_names, test_set_names, val_set_names, train_val_set_names)

def read_image_annotations(filename):
    PATH = "data/acfr-fruit-dataset/almonds"
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
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0
        # image = np.moveaxis(image, -1, 0)
        annotations_df = pd.read_csv(filename_annotations)
        print(annotations_df)
        # boxes = pd.DataFrame({'x_start': [0], 'y_start': [0],
        #                       'x_end': [0], 'y_end': [0]})

        boxes = np.genfromtxt(filename_annotations, delimiter=',')

        # if annotations.empty == False:
            # boxes = pd.DataFrame({'x_start': annotations['x'], 'y_start': annotations['y'],
            #                     'x_end': (annotations['x']+annotations['dx']),
            #                     'y_end': (annotations['y']+annotations['dy'])})

        labels = torch.ones((self.data.shape[0],), dtype=torch.int64)
        isfruit = torch.zeros((self.data.shape[0],), dtype=torch.int64)

        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['image_id'] = torch.tensor([index])
        target['isfruit'] = isfruit

        return image, target, image_id

    def __len__(self):
        return self.data.shape[0]

def collate_fn(batch):
    return tuple(zip(*batch))

PATH = "data/acfr-fruit-dataset/almonds"
training_data = read_data(FruitType.APPLE)[0]
train_dataset = FruitDataset(training_data, PATH)

train_data_loader = DataLoader(
    train_dataset,
    # batch_size=1,
    # shuffle=False,
    # num_workers=0,
)

images, target, image_id = next(iter(train_data_loader))
image2, target2, image_id2 = train_dataset.__getitem__(0)
fig, ax = plt.subplots(1, 1, figsize=(16, 8))

image = images[0]

for idx, val in enumerate(target['boxes']):
    # x_start = int(val[1].item())
    # y_start = int(val[2].item())
    # x_end = int(val[1].item() + val[3].item())
    # y_end = int(val[2].item() + val[4].item())
    x_start = int(val[0][1].item())
    y_start = int(val[0][2].item())
    x_end = int(val[0][1].item() + val[0][3].item())
    y_end = int(val[0][2].item() + val[0][4].item())
    print(x_start)
    print(y_start)
    print(x_end)
    print(y_end)
    # cv2.rectangle(image2, (20,30), (20,30), (255, 0, 0))
    cv2.rectangle(image, (int(x_start), int(y_start)), (int(x_end), int(y_end)), (0, 255, 0), 5)

ax.set_axis_off()
ax.imshow(image2)
plt.show()