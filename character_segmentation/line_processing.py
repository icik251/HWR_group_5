import cv2
import numpy as np
from skimage.morphology import medial_axis, skeletonize
from matplotlib import pyplot as plt
from pathlib import Path
import os

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
        self.binary_image = None
        self.vertical_histogram = None
        self.line_width_array = None
        self.skel_line = None
        self.list_of_segmented_characters = list()

    def apply_vertical_projection(self, binary_treshold="median"):
        thresh, self.binary_image = convert_image_to_binary(
            self.gray_image, mode=binary_treshold
        )

        # skeletonize before vertical projection
        self.skeletonize_line()

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

        # self.vertical_histogram = self.skel_line

    def segment_characters(self, window_size=5):
        curr_idx = 0
        self.list_of_segmented_characters = list()
        for column_idx in range(len(self.line_width_array) - window_size):
            if self.check_window(window_size, column_idx):
                self.list_of_segmented_characters.append(
                    self.gray_image[:, curr_idx:column_idx]
                )
                curr_idx = column_idx

    def check_window(self, window_size, curr_col_idx):
        condition_met = True
        if self.line_width_array[curr_col_idx] == 0:
            for i in range(1, window_size + 1):
                if not self.line_width_array[curr_col_idx + i] >= 1:
                    condition_met = False

        else:
            condition_met = False

        return condition_met

    def skeletonize_line(self):
        thresh, self.binary_image = convert_image_to_binary(
            self.gray_image,
        )

        reversed_line = reverse_black_white_keep_values(self.binary_image)
        # kernel = np.ones((10, 10), np.uint8)
        # dilated_line = cv2.dilate(reversed_line, kernel, iterations=1)

        self.skel_line = skeletonize(reversed_line, method="lee")

        # undo_reverse_image
        self.skel_line = reverse_black_white_keep_values(self.skel_line)

        # plt.imshow(skel_image)
        # plt.show()

    def save_segmented_characters(self):
        path_to_line_folder = Path(self.path_to_image_line).parent.absolute()
        for idx, character_image in enumerate(self.list_of_segmented_characters):
            save_image(
                character_image,
                "char_{}".format(idx),
                path_to_line_folder,
                extension="png",
            )


line_processing = LineProcessing(
    "D:\\PythonProjects\\HWR_group_5\\data\\TESTING_some_image_name\\line_1\\line_1.png"
)

line_processing.apply_vertical_projection()
line_processing.segment_characters(window_size=10)
line_processing.save_segmented_characters()
