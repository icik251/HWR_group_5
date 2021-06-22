import os
from utils import crop_white_spaces_image_v2, resize_image
import matplotlib.pyplot as plt
import cv2


class CharacterProcessing:
    def __init__(
        self, path_to_image, resize_mode="smallest", model_mode="recognition"
    ) -> None:
        """Processing methods for a single image containing a character. Real-life scenario usage

        Args:
            path_to_image ([type]): Path to the image
            resize_mode (str, optional): The resizing modem possible: "smallest" and "average". Defaults to "smallest".
            model_mode (str, optional): The model for which the processing is done: possible: "recognition" and "style".
                                                                                                Defaults to "recognition".
        """
        self.path_to_image = path_to_image
        self.resize_mode = resize_mode
        self.model_mode = model_mode
        self.shape = None
        self.resized_image = None

        try:
            self.image = self._load_image()
        except Exception as e:
            print(e)
            print("Image can't be loaded")

    def _load_image(self):
        return cv2.imread(self.path_to_image, -1)

    def _load_shape(self):
        if self.resize_mode == "smallest":
            path_to_letter_folders = "data\\resizing_shapes\\normalized_smallest"
        elif self.resize_mode == "average":
            path_to_letter_folders = "data\\resizing_shapes\\normalized_avg"

        if self.model_mode == "recognition":
            path_to_shape = os.path.join(
                path_to_letter_folders, "Alef", "recognition_shape.txt"
            )
        elif self.model_mode == "style":
            curr_char = os.path.split(self.path_to_image)[1].split("_")[0]
            path_to_shape = os.path.join(
                path_to_letter_folders, curr_char, "style_shape.txt"
            )

        with open(path_to_shape, "r") as f:
            self.shape = f.readline().split(",")
            self.shape = tuple([int(item) for item in self.shape])
        f.close()

    def resize_image(self):
        self._load_shape()

        try:
            cropped_image = crop_white_spaces_image_v2(self.image)
        except Exception as e:
            print(e)
            cropped_image = self.image

        self.resized_image = resize_image(cropped_image, self.shape)

        # plt.imshow(self.resized_image)
        # plt.show()

        # plt.imshow(self.image)
        # plt.show()

    def get_image(self):
        return self.resized_image


RESIZE_MODE = "smallest"

for line_folder in os.listdir("data\\TESTING_some_image_name\\"):
    # if for recognition
    if line_folder.split("_")[0] == "line":
        for image_in_line_folder in os.listdir(
            os.path.join("data\\TESTING_some_image_name\\", line_folder)
        ):
            if image_in_line_folder.split("_")[0] == "char":
                path_to_image = os.path.join(
                    "data\\TESTING_some_image_name\\", line_folder, image_in_line_folder
                )
                character_processing = CharacterProcessing(
                    path_to_image, resize_mode=RESIZE_MODE, model_mode="recognition"
                )
                character_processing.resize_image()

    # else for style classification when letters are already recognized
    if line_folder.split("_")[0] == "recognized":
        for image_in_line_folder in os.listdir(
            os.path.join("data\\TESTING_some_image_name\\", line_folder)
        ):
            if image_in_line_folder.split("_")[1].split("_")[0] == "char":
                path_to_image = os.path.join(
                    "data\\TESTING_some_image_name\\", line_folder, image_in_line_folder
                )
                character_processing = CharacterProcessing(
                    path_to_image, resize_mode=RESIZE_MODE, model_mode="style"
                )
                character_processing.resize_image()
