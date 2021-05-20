import os
import numpy as np
import cv2

from character_processing import CharacterProcessing
from data_loader import DataLoader


class CharacterRecognition:
    def __init__(
        self, path_to_structuring_elements, threshold, dict_of_scrolls
    ) -> None:

        self.threshold = threshold
        self.dict_of_structuring_elements = dict()
        self.dict_of_scrolls = dict_of_scrolls

        for letter_se in os.listdir(path_to_structuring_elements):
            if letter_se.endswith("_{}.pgm".format(self.threshold)):
                curr_se = cv2.imread(
                    os.path.join(path_to_structuring_elements, letter_se), -1
                )
                self.dict_of_structuring_elements[letter_se.split(".")[0]] = curr_se

    def reverse_black_white(self, image):
        image_reversed = 255 - image
        return image_reversed

    def resize_image(self, image, mode="upscale", scaler=4):
        if mode == "upscale":
            resized_image = cv2.resize(
                image,
                dsize=(image.shape[1] * scaler, image.shape[0] * scaler),
                interpolation=cv2.INTER_CUBIC,
            )
        elif mode == "downscale":
            resized_image = cv2.resize(
                image,
                dsize=(image.shape[1] // scaler, image.shape[0] // scaler),
                interpolation=cv2.INTER_CUBIC,
            )

        return resized_image

    @staticmethod
    def inverse_image(image):
        image[image == 0] = 1
        image[image == 255] = 0

        return image

    def erode_on_image(self, searched_letter, scroll_name, resize_mode=None, scaler=1):

        path_to_save = self.create_path_if_not_exist(
            searched_letter, scroll_name, resize_mode, scaler
        )

        kernel = self.dict_of_structuring_elements[
            searched_letter + "_{}".format(self.threshold)
        ]

        if resize_mode is not None:
            # resize
            kernel = self.resize_image(kernel, mode=resize_mode, scaler=scaler)

        image_scroll = dict_of_scrolls[scroll_name]

        try:
            cropped_kernel = CharacterProcessing.crop_white_spaces_image_v2(kernel)
        except Exception as e:
            print(kernel)
            cropped_kernel = kernel

        _, bw_kernel = CharacterProcessing.convert_image_to_binary(cropped_kernel)

        final_kernel = CharacterRecognition.inverse_image(bw_kernel)
        final_image_scroll = CharacterRecognition.inverse_image(image_scroll)

        # kernel = np.ones((1, 1), np.uint8)

        result = cv2.erode(final_image_scroll, final_kernel, iterations=1)

        cv2.imwrite(
            os.path.join(path_to_save, "threshold_{}.pgm".format(self.threshold)),
            result,
        )

    def create_path_if_not_exist(
        self, searched_letter, scroll_name, resize_mode, scaler
    ):
        path_to_check = os.path.join(
            "data\\exracted_characters",
            scroll_name,
            searched_letter,
            str(resize_mode),
            str(scaler),
        )
        if not os.path.exists(path_to_check):
            os.makedirs(path_to_check)

        return path_to_check


data_loader = DataLoader()
dict_of_scrolls = data_loader.get_sea_scrolls_images(
    "D:\\PythonProjects\\HWR_group_5\\data\\dead_sea_scrolls_images"
)

dict_of_scrolls = dict()
dict_of_scrolls['nice_roi_2'] = cv2.imread(os.path.join("D:\\PythonProjects\\HWR_group_5\\data\\lines_nice", 'roi(2).png'), cv2.IMREAD_GRAYSCALE)

list_of_scalers = list(range(1, 5))
list_of_modes = ["upscale", "downscale", None]
for i in range(3, 11):
    for scaler in list_of_scalers:
        for mode in list_of_modes:
            character_recognition = CharacterRecognition(
                "D:\\PythonProjects\\HWR_group_5\\data\\structuring_elements_0.3\\final_structuring_elements_binary",
                threshold=i,
                dict_of_scrolls=dict_of_scrolls,
            )

            if mode is None:
                character_recognition.erode_on_image(
                    "Alef",
                    scroll_name="nice_roi_2",
                    resize_mode=mode,
                    scaler=1,
                )
            else:
                character_recognition.erode_on_image(
                    "Alef",
                    scroll_name="nice_roi_2",
                    resize_mode=mode,
                    scaler=scaler,
                )
