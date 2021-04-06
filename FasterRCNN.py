import cv2
import numpy as np
import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator
from torch.utils.data import DataLoader
from Plotter import plot_bounding_boxes
from Reader import FruitDataset, read_data
from detection import utils
from detection.engine import train_one_epoch, evaluate
from torch.utils.tensorboard import SummaryWriter


def main():
    PATH = "data/acfr-fruit-dataset/almonds"
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda')

    # load resnet faster rcnn
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, progress=True, num_classes=2,
                                                                 pretrained_backbone=False)

    # use our dataset and defined transformations
    training_data = read_data()[0]
    dataset = FruitDataset(training_data, PATH)
    test_data = read_data()[1]
    dataset_test = FruitDataset(test_data, PATH)
    val_data = read_data()[2]
    dataset_val = FruitDataset(val_data, PATH)

    # subset of training set
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:3])

    # define training, test, and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=True, num_workers=1,
        collate_fn=utils.collate_fn)
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=1,
        collate_fn=utils.collate_fn)
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, batch_size=1, shuffle=False, num_workers=1,
        collate_fn=utils.collate_fn)

    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.001,
                                momentum=0.9, weight_decay=0.0005)

    # let's train it for x epochs
    num_epochs = 40000
    writer = SummaryWriter()
    maxAcc = 0.0
    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=100)
        # put the model in evaluation mode
        model.eval()
        # evaluate on the validation dataset and save model if accuracy is highest yet
        cocoEval = evaluate(model, data_loader_val, device=device)
        acc = cocoEval.coco_eval.get('bbox').accuracy
        if acc > maxAcc:
            maxAcc = acc
            torch.save(model, 'model_3_resnet.pt')
        writer.add_scalar('Loss/validation_3_resnet', acc, epoch)

    # evaluate test set
    evaluate(model, data_loader_test, device=device)
    print("That's it!")


# predict and plot image
def show_image(readed_img, model, device):
    with torch.no_grad():
        prediction = model([readed_img[0].to(device)])
        boxes = prediction[0].get('boxes')
        filename_image = 'data/acfr-fruit-dataset/almonds' + '/images/' + readed_img[1]['image_id'] + '.png'
        image = cv2.imread(filename_image)
        plot_bounding_boxes(image, boxes)


if __name__ == '__main__':
    main()
