import cv2
import os
from data_loader import DataLoader
import numpy as np


class CharacterProcessing:
    def __init__(self, dict_of_chars) -> None:
        self.dict_of_chars = dict_of_chars
        self.dict_of_structuring_elements = dict()

    def resize_image(self, image, shape: tuple):
        return cv2.resize(image, shape)

    def normalize_character_images_size(self, letter, list_of_samples, save_mode=True):
        shape_to_be = [0, 0]
        for sample in list_of_samples:
            shape_to_be[0] += sample.shape[0]
            shape_to_be[1] += sample.shape[1]

        shape_to_be = np.divide(shape_to_be, len(list_of_samples))
        shape_to_be = list(np.array(shape_to_be).astype("uint8"))
        shape_to_be = tuple(shape_to_be)

        list_of_normalized_images = list()

        for idx, sample in enumerate(list_of_samples):
            resized_img = self.resize_image(sample, shape_to_be)
            # convert to binary (not working here because there are no intersections,
            # in the case of 300 examples like Alef)
            # (thresh, resized_img_binary) = cv2.threshold(
            #    resized_img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
            # )
            if save_mode:
                self.save_normalized_character_images(letter, idx, resized_img)
            list_of_normalized_images.append(resized_img)

        return list_of_normalized_images

    def save_normalized_character_images(self, letter, img_name, img):
        if not os.path.exists(os.path.join("data\\resized", letter)):
            os.makedirs(os.path.join("data\\resized", letter))
        cv2.imwrite(
            os.path.join("data\\resized", letter, "{}.pgm".format(img_name)), img
        )

    def create_structuring_elements(self, save_mode=False):
        for letter, list_of_samples in self.dict_of_chars.items():

            list_of_samples_resized = self.normalize_character_images_size(
                letter, list_of_samples, save_mode=save_mode
            )

            # old init structuring element
            # structuring_element = self._init_structuring_element(list_of_samples)

            for idx, train_sample in enumerate(list_of_samples_resized):
                # init structuring element from first sample
                if idx == 0:
                    structuring_element = list_of_samples_resized[0].astype("uint32")
                    continue

                for i in range(train_sample.shape[0]):

                    # Old way that makes struct element with the shape of the
                    # biggest image in size
                    """
                    curr_row = train_sample[i, :]
                    added_array = np.full(
                        (structuring_element.shape[1] - len(curr_row),), 255
                    )

                    structuring_element[i, :] = np.add(
                        structuring_element[i, :],
                        np.concatenate((curr_row, added_array)),
                    )
                    """
                    try:
                        structuring_element[i, :] = np.add(
                            structuring_element[i, :], train_sample[i, :]
                        )
                    except Exception as e:
                        print(e)

            structuring_element = np.divide(
                structuring_element, len(list_of_samples_resized)
            )
            structuring_element = np.rint(structuring_element).astype("uint8")

            self.dict_of_structuring_elements[letter] = structuring_element

    def save_alpha_structuring_elements(self):
        if not os.path.exists(os.path.join("data\\alpha_structuring_elements_binary")):
            os.makedirs(os.path.join("data\\alpha_structuring_elements_binary"))

        for letter, structuring_element in self.dict_of_structuring_elements.items():
            median_value = np.median(structuring_element)
            threshold = int(median_value / 1.33)
            (_, structuring_element_binary) = cv2.threshold(
                structuring_element, threshold, 255, cv2.THRESH_BINARY
            )

            cv2.imwrite(
                os.path.join(
                    "data\\alpha_structuring_elements_binary", "{}_{}.pgm".format(letter, threshold)
                ),
                structuring_element_binary,
            )

    def get_structuring_elements(self):
        return self.dict_of_structuring_elements

    def build_structuring_elements(self, save_normalized=False):
        self.create_structuring_elements(save_normalized)
        self.save_alpha_structuring_elements()

    def _init_structuring_element(self, list_of_samples):
        # the images are different sizes and this results in bad structuring element
        curr_shape = [0, 0]
        for sample in list_of_samples:
            if sample.shape[0] > curr_shape[0]:
                curr_shape[0] = sample.shape[0]
            if sample.shape[1] > curr_shape[1]:
                curr_shape[1] = sample.shape[1]

        return np.full(tuple(curr_shape), 0)


data_loader = DataLoader()
dict_result = data_loader.get_characters_train_data(
    path="D:\\PythonProjects\\HWR_group_5\\data\\character_set_labeled")

char_processing = CharacterProcessing(dict_result)
char_processing.build_structuring_elements(save_normalized=True)
