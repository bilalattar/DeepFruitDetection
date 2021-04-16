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

    # load backbone
    vgg = torchvision.models.vgg16(pretrained=False)
    backbone = vgg.features[:-1]
    for param in backbone.parameters():
        param.require_grad = True

    # FasterRCNN needs to know the number of
    # output channels in a backbone. For mobilenet_v2, it's 1280
    # so we need to add it here
    backbone.out_channels = 512

    # let's make the RPN generate 5 x 3 anchors per spatial
    # location, with 5 different sizes and 3 different aspect
    # ratios. We have a Tuple[Tuple[int]] because each feature
    # map could potentially have different sizes and
    # aspect ratios
    anchor_generator = AnchorGenerator(sizes=((128, 256, 512),),  # smaller bounding boxes: 64, 128, 256
                                       aspect_ratios=((0.5, 1.0, 2.0),))

    # let's define what are the feature maps that we will
    # use to perform the region of interest cropping, as well as
    # the size of the crop after rescaling.
    # if your backbone returns a Tensor, featmap_names is expected to
    # be [0]. More generally, the backbone should return an
    # OrderedDict[Tensor], and in featmap_names you can choose which
    # feature maps to use.
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
                                                    output_size=9,
                                                    sampling_ratio=2)

    # put the pieces together inside a FasterRCNN model
    #For ResNet-50, use the following line
    # model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, progress=True, num_classes=2,
    #                                                              pretrained_backbone=False)
    model = FasterRCNN(backbone,
                       num_classes=2,
                       rpn_anchor_generator=anchor_generator,
                       box_roi_pool=roi_pooler,
                       min_size=600,
                       max_size=1000,
                       rpn_pre_nms_top_n_train=12000,
                       rpn_pre_nms_top_n_test=6000,
                       rpn_post_nms_top_n_train=2000,
                       rpn_post_nms_top_n_test=300,
                       rpn_nms_thresh=0.3,
                       rpn_fg_iou_thresh=0.7,
                       rpn_bg_iou_thresh=0.2,
                       rpn_batch_size_per_image=256,
                       rpn_positive_fraction=0.5,
                       box_batch_size_per_image=128,
                       box_positive_fraction=0.25,
                       box_score_thresh=0.1,
                       box_nms_thresh=0.3,
                       box_detections_per_img=100
                       )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)

    # use our dataset and defined transformations
    training_data = read_data()[0]
    dataset = FruitDataset(training_data, PATH)
    test_data = read_data()[1]
    dataset_test = FruitDataset(test_data, PATH)
    val_data = read_data()[2]
    dataset_val = FruitDataset(val_data, PATH)

    # subset of training set
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:27])

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
            torch.save(model, 'model_27.pt')
        writer.add_scalar('Loss/validation_27', acc, epoch)

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