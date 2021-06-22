import os
import cv2
import numpy as np
import torch


def crop_white_spaces_image(image):
    # Load image, grayscale, Gaussian blur, Otsu's threshold
    original = image.copy()
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(image, (25, 25), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Perform morph operations, first open to remove noise, then close to combine
    noise_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, noise_kernel, iterations=2)
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    close = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, close_kernel, iterations=3)

    # Find enclosing boundingbox and crop ROI
    coords = cv2.findNonZero(close)
    x, y, w, h = cv2.boundingRect(coords)
    cv2.rectangle(image, (x, y), (x + w, y + h), (36, 255, 12), 2)
    crop = original[y : y + h, x : x + w]

    return crop


def is_image_rgb(image):
    if len(image.shape) < 3:
        return False
    elif len(image.shape) == 3:
        return True


def crop_white_spaces_image_v2(image):
    is_rgb = is_image_rgb(image)

    if is_rgb:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    to_crop_from = gray

    gray = 255 * (gray < 128).astype(np.uint8)  # To invert the text to white
    coords = cv2.findNonZero(gray)  # Find all non-zero points (text)
    x, y, w, h = cv2.boundingRect(coords)  # Find minimum spanning bounding box
    rect = to_crop_from[
        y : y + h, x : x + w
    ]  # Crop the image - note we do this on the original image

    return rect


def resize_image(image, shape: tuple):
    return cv2.resize(image, shape)


def convert_image_to_binary(image, mode="median"):
    if mode == "median":
        median_value = np.median(image)
        threshold = int(median_value / 1.33)

        (_, image_binary) = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    elif isinstance(mode, int):
        (_, image_binary) = cv2.threshold(image, mode, 255, cv2.THRESH_BINARY)
        threshold = mode

    return int(threshold), image_binary


def convert_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def save_image(image, image_name, path):
    if not os.path.exists(path):
        os.makedirs(path)

    try:
        cv2.imwrite(os.path.join(path, "{}.pgm".format(image_name)), image)
    except Exception as e:
        print(e)


def load_checkpoint(path_of_model, epoch_checkpoint):
    return torch.load(
        os.path.join(path_of_model, "checkpoint_{}.pth".format(epoch_checkpoint))
    )


def multi_acc(y_pred, y_labels):
    y_pred_softmax = torch.log_softmax(y_pred, dim=1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim=1)

    correct_pred = (y_pred_tags == y_labels).float()
    acc = correct_pred.sum() / len(correct_pred)

    return acc * 100


def set_parameter_requires_grad(model, freeze=True):
    if freeze:
        for param in model.parameters():
            param.requires_grad = False

    return model


def reverse_black_white_keep_values(image):
    image[image == 255] = 1
    image[image == 0] = 255
    image[image == 1] = 0

    return image


def boolean_to_255(image):
    image[image == 1] = 255
    image[image == 0] = 0

    return image


def pixels_255_to_boolean(image):
    image[image == 0] = 1
    image[image == 255] = 0

    return image


def reverse_black_white_boolean_values(image):
    image[image == 0] = 0
    image[image == 255] = 1

    return image
