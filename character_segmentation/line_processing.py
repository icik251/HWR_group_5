import cv2
import numpy as np
from skimage.morphology import medial_axis, skeletonize
from matplotlib import pyplot as plt
from pathlib import Path
import os

import torch

from utils import (
    convert_image_to_binary,
    convert_grayscale,
    reverse_black_white_boolean_values,
    reverse_black_white_keep_values,
    save_image,
)


class LineProcessing:
    def __init__(self, path_to_image_line) -> None:
        self.path_to_image_line = path_to_image_line
        self.initial_line = cv2.imread(self.path_to_image_line)
        self.gray_image = convert_grayscale(self.initial_line)
        self.path_to_line_folder = Path(self.path_to_image_line).parent.absolute()
        self.binary_image = None
        self.vertical_histogram = None
        self.line_width_array = None
        self.skel_line = None
        self.list_of_segmented_characters = list()

    def apply_vertical_projection(self, binary_treshold="median"):
        # skeletonize before vertical projection
        self.skeletonize_line(binary_treshold)

        (h, w) = self.binary_image.shape
        self.line_width_array = [0 for z in range(0, w)]

        # Record the peaks of each column
        for j in range(0, w):  # traverse a column
            for i in range(0, h):  # traverse a line
                if (
                    self.skel_line[i, j] == 0
                ):  # If the point is changed to a black point
                    self.line_width_array[
                        j
                    ] += 1  # The counter in this column is incremented by one
                    self.skel_line[i, j] = 255  # turn it to white after recording

        for j in range(0, w):  # Iterate through each column
            for i in range(
                (h - self.line_width_array[j]), h
            ):  # Blacken to the bottom from the top point where the column should turn black
                self.skel_line[i, j] = 0  # black

    def segment_characters(self, window_size=5):
        ## Pipeine:
        # convert line grayscale -> convert line binary otsu method -> skeletonize lee method ->
        # segment chars with window size=5 (add left borders (20 pix) to char on each segmentation) ->
        # fill missing parts of the characters with erosion, dilation -> remove characters with black/white ratio
        # lower than 5% -> TODO: If image is > 150 pixels on width, try to separate it on more characters because
        # most likely they are more chars in the image if it is > 150 width (empirically seen on 4 lines)

        curr_idx = 0
        self.list_of_segmented_characters = list()
        for column_idx in range(len(self.line_width_array) - window_size):
            if self.check_window(window_size, column_idx):

                self.list_of_segmented_characters.append(
                    self.add_borders_to_image(self.gray_image[:, curr_idx:column_idx])
                )

                curr_idx = column_idx

    def add_borders_to_image(self, image):
        padding_adder = torch.nn.ZeroPad2d((20, 0, 0, 0))
        reversed_black_white_sample = reverse_black_white_keep_values(image)
        padded_sample = np.array(
            padding_adder(torch.from_numpy(reversed_black_white_sample))
        )
        return reverse_black_white_keep_values(padded_sample)

    def remove_non_character_images(self):
        for character_image in self.list_of_segmented_characters[:]:
            number_of_white_pix = np.sum(character_image == 255)
            number_of_black_pix = np.sum(character_image == 0)
            # plt.imshow(character_image)
            # plt.show()
            # print(number_of_black_pix / number_of_white_pix)
            if (
                (number_of_black_pix / number_of_white_pix) < 0.05
                or character_image.shape[0] < 10
                or character_image.shape[1] < 10
            ):
                self.list_of_segmented_characters.remove(character_image)

    def fill_segmented_character(self):
        for idx in range(len(self.list_of_segmented_characters)):
            self.list_of_segmented_characters[idx] = self._fill_ruined_pixels(
                self.list_of_segmented_characters[idx]
            )

    def separate_images_above_width(self, width=150):
        list_of_results = list()
        for segmented_char in self.list_of_segmented_characters:
            if segmented_char.shape[1] < width:
                list_of_results.append(segmented_char)
            else:
                # boundaries_found = cv2.Canny(segmented_char, 30, 200)
                # plt.imshow(boundaries_found)
                # plt.show()

                _, binary_image = convert_image_to_binary(segmented_char, mode="otsu")
                reversed_binary_image = reverse_black_white_keep_values(binary_image)
                # getting mask with connectComponents
                ret, labels = cv2.connectedComponents(reversed_binary_image)
                for label in range(1, ret):
                    mask = np.array(labels, dtype=np.uint8)
                    mask[labels == label] = 255
                    # cv2.imshow('component',mask)
                    # cv2.waitKey(0)

                # getting ROIs with findContours
                contours = cv2.findContours(
                    reversed_binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )[0]

                # make them in left to right order
                h_list = list()
                for cnt in contours:
                    [x, y, w, h] = cv2.boundingRect(cnt)
                    if w * h > 250:
                        h_list.append([x, y, w, h])

                ziped_list = list(zip(*h_list))
                x_list = list(ziped_list[0])
                dict_of_x_h = dict(zip(x_list, h_list))
                x_list.sort()

                for x in x_list:
                    (x, y, w, h) = dict_of_x_h[x]
                    segmented_character_final = segmented_char[y : y + h, x : x + w]
                    list_of_results.append(segmented_character_final)

                    # cv2.imshow("ROI", segmented_character_final)
                    # cv2.waitKey(0)

                # cv2.destroyAllWindows()

        self.list_of_segmented_characters = list_of_results

    def segmentation_logic_2(self):
        thresh, self.binary_image = convert_image_to_binary(
            self.gray_image, mode="otsu"
        )

        _, binary_image = convert_image_to_binary(self.binary_image, mode="otsu")
        reversed_binary_image = reverse_black_white_keep_values(binary_image)
        # getting mask with connectComponents
        ret, labels = cv2.connectedComponents(reversed_binary_image)
        for label in range(1, ret):
            mask = np.array(labels, dtype=np.uint8)
            mask[labels == label] = 255
            # cv2.imshow('component',mask)
            # cv2.waitKey(0)

        # getting ROIs with findContours
        contours = cv2.findContours(
            reversed_binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )[0]
        for cnt in contours:
            (x, y, w, h) = cv2.boundingRect(cnt)
            segmented_character_final = self.binary_image[y : y + h, x : x + w]
            self.list_of_segmented_characters.append(segmented_character_final)

    def check_window(self, window_size, curr_col_idx):
        condition_met = True
        if self.line_width_array[curr_col_idx] == 0:
            for i in range(1, window_size + 1):
                if not self.line_width_array[curr_col_idx + i] >= 1:
                    condition_met = False

        else:
            condition_met = False

        return condition_met

    def skeletonize_line(self, binary_threshold):
        thresh, self.binary_image = convert_image_to_binary(
            self.gray_image, binary_threshold
        )

        save_image(
            self.binary_image,
            "line_binary",
            self.path_to_line_folder,
            extension="png",
        )

        """
        self.binary_image = self._fill_ruined_pixels(self.binary_image)

        save_image(
            self.binary_image,
            "line_filled_erosion_dilate",
            self.path_to_line_folder,
            extension="png",
        )
        """
        reversed_line = reverse_black_white_keep_values(self.binary_image)
        # kernel = np.ones((10, 10), np.uint8)
        # dilated_line = cv2.dilate(reversed_line, kernel, iterations=1)

        self.skel_line = skeletonize(reversed_line, method="lee")

        # undo_reverse_image
        self.skel_line = reverse_black_white_keep_values(self.skel_line)

        save_image(
            self.skel_line,
            "line_skel",
            self.path_to_line_folder,
            extension="png",
        )
        # plt.imshow(skel_image)
        # plt.show()

    def _fill_ruined_pixels(self, image):
        """
        contours, hier = cv2.findContours(
            self.binary_image.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
        )
        stencil = np.zeros(self.binary_image.shape).astype(self.binary_image.dtype)

        color = [255]

        # Fill out ALL Contours

        cv2.fillPoly(stencil, contours, color)

        result = cv2.bitwise_and(self.binary_image, stencil)

        save_image(
            result,
            "line_filled",
            self.path_to_line_folder,
            extension="png",
        )
        """
        kernel = np.ones((5, 5), np.uint8)

        img_erode = cv2.erode(image, kernel, iterations=1)
        img_filled = cv2.dilate(img_erode, kernel, iterations=1)

        return img_filled

    def save_segmented_characters(self):
        for idx, character_image in enumerate(self.list_of_segmented_characters):
            save_image(
                character_image,
                "char_{}".format(idx),
                self.path_to_line_folder,
                extension="png",
            )


line_processing = LineProcessing(
    "D:\\PythonProjects\\HWR_group_5\\data\\TESTING_some_image_name\\line_1\\line_1.png"
)

## Our main approach

"""
line_processing.apply_vertical_projection(binary_treshold="otsu")
line_processing.segment_characters(window_size=5)
line_processing.fill_segmented_character()
line_processing.separate_images_above_width()
line_processing.remove_non_character_images()
line_processing.save_segmented_characters()
"""

## Just contour approach
"""
line_processing.segmentation_logic_2()
line_processing.save_segmented_characters()
"""
