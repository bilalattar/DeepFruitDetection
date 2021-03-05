import cv2
import matplotlib.pyplot as plt


def plot_bounding_boxes(image, annotations):
    fig, ax = plt.subplots(1, 1, figsize=(16, 8))
    annons = annotations.to_numpy()
    for box in annons:
        x_start = int(box[1])
        y_start = int(box[2])
        x_end = x_start + int(box[3])
        y_end = y_start + int(box[4])
        cv2.rectangle(image, (x_start,y_start), (x_end,y_end), (255, 0, 0))
    ax.set_axis_off()
    ax.imshow(image)
    plt.show()