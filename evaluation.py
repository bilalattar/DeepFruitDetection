import torch

from Reader import read_data, FruitDataset
from detection import utils
from detection.engine import evaluate

MODEL_PATH = 'model_27.pt'
DATA_PATH = "data/acfr-fruit-dataset/almonds"

def evaluate_model():
    device = torch.device('cuda')
    model = torch.load(MODEL_PATH)
    test_data = read_data()[1]
    dataset_test = FruitDataset(test_data, DATA_PATH)
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=1,
        collate_fn=utils.collate_fn)
    evaluate(model, data_loader_test, device=device)

if __name__ == '__main__':
    evaluate_model()