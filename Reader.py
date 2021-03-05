from enum import Enum
import pandas as pd
import cv2

class FruitType(Enum):
    APPLE=1
    MANGO=2
    ALMOND=3

def read_data(FruitType):
    PATH="data/acfr-fruit-dataset/apples"
    training_set_names = pd.read_csv(f"{PATH}/sets/train.txt", names=["image_id"])
    test_set_names = pd.read_csv(f"{PATH}/sets/test.txt", ["image_id"])
    val_set_names = pd.read_csv(f"{PATH}/sets/val.txt", ["image_id"])
    train_val_set_names = pd.read_csv(f"{PATH}/sets/train_val.txt", ["image_id"])

    return training_set_names, test_set_names, val_set_names, train_val_set_names

def read_image_annotations(filename):
    PATH = "data/acfr-fruit-dataset/apples"
    filename_annotations = PATH + '/annotations/' + filename + '.csv'
    filename_image = PATH + '/images/' + filename + '.png'

    image = cv2.imread(filename_image, cv2.IMREAD_COLOR)
    annotations = pd.read_csv(filename_annotations)

    return image, annotations

class FruitDataset()


read_image_annotations('20130320T004354.849492.Cam6_52')