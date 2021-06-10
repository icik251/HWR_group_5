import os
from utils import convert_image_to_binary
from data_loader import DataLoader
import numpy, imageio, elasticdeform

import numpy as np
import cv2
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt

X = numpy.zeros((200, 300))
X[::10, ::10] = 1

# apply deformation with a random 3 x 3 grid
X_deformed = elasticdeform.deform_random_grid(X, sigma=25, points=3)

imageio.imsave("test_X.png", X)
imageio.imsave("test_X_deformed.png", X_deformed)


def elastic_transform(image, alpha_range, sigma, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_.
     .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
        Convolutional Neural Networks applied to Visual Document Analysis", in
        Proc. of the International Conference on Document Analysis and
        Recognition, 2003.

    # Arguments
        image: Numpy array with shape (height, width, channels).
        alpha_range: Float for fixed value or [lower, upper] for random value from uniform distribution.
            Controls intensity of deformation.
        sigma: Float, sigma of gaussian filter that smooths the displacement fields.
        random_state: `numpy.random.RandomState` object for generating displacement fields.
    """

    if random_state is None:
        random_state = np.random.RandomState(None)

    if np.isscalar(alpha_range):
        alpha = alpha_range
    else:
        alpha = np.random.uniform(low=alpha_range[0], high=alpha_range[1])

    shape = image.shape
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha

    x, y, z = np.meshgrid(
        np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing="ij"
    )
    indices = (
        np.reshape(x + dx, (-1, 1)),
        np.reshape(y + dy, (-1, 1)),
        np.reshape(z, (-1, 1)),
    )

    return map_coordinates(image, indices, order=1, mode="reflect").reshape(shape)


# Load images for the first time
data_loader = DataLoader()
dict_of_results = data_loader.get_characters_style_based(
    "D:\\PythonProjects\\HWR_group_5\\data\\style_classification\\characters_for_style_classification"
)

for style_class, dict_of_letters in dict_of_results.items():
    for letter_key, value_list in dict_of_letters.items():
        list_of_resized_samples = list()
        for idx, sample in enumerate(value_list):
            # cropped_image = crop_white_spaces_image_v2(sample)
            # resized_image = white_bg_square(Image.fromarray(sample))
            _, binary_img = convert_image_to_binary(sample)

            # binary_img[binary_img == 0] = 1
            # binary_img[binary_img == 255] = 0
            # binary_img[binary_img == 1] = 255
            np_list = np.array(value_list)
            num_examples = 10
            ed_sample = np.expand_dims(
                np_list[np.random.choice(np_list.shape[0], num_examples)], -1
            )
            sigma = 2
            alpha = 8

            X_deformed = elastic_transform(ed_sample, alpha_range=alpha, sigma=sigma)

            path_to_save_image = os.path.join(
                "data\\style_classification\\testing_augment",
                style_class,
                letter_key,
            )

            imageio.imsave(
                "data\\style_classification\\testing_augment\\test_X_{}_{}.png".format(
                    sigma, alpha
                ),
                ed_sample,
            )
            imageio.imsave(
                "data\\style_classification\\testing_augment\\test_X_deformed_{}_{}.png".format(
                    sigma, alpha
                ),
                X_deformed,
            )

            # save_image(np.asarray(resized_image), idx, path_to_save_image)
            break
        break
    break
