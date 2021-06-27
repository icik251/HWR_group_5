import os
from pathlib import Path
import cv2

"""
Take a look into these:
https://docs.opencv.org/3.4/d8/d6a/group__imgcodecs__flags.html
the flags that set the color when loading the image. Discuss with group.
"""

class DataLoader:
    def __init__(self) -> None:
        pass

    def get_characters_train_data(
        self, path: Path, list_of_letters: list = list(), num_samples: int = 300
    ) -> dict():
        """Loads all or chosen by user training images for letters.

        Args:
            path: Path to training characters folder
            list_of_letters: List of chosen letters from user. If you don't want
                to load all 27 letters, you can for example: ["Bet", "Dalet"].
                Defaults to empty list, which loads all 27 letters.
            num_samples: Number of samples per letter you want to load.
                Defaults to 300, which loads all samples for each letter.
        Returns:
            dict_of_result: Dictionary with key - letter name and value - list of
                loaded samples for the following letter.
        """

        dict_of_result = dict()

        for letter_dir_name in os.listdir(path):
            if len(list_of_letters) != 0 and letter_dir_name in list_of_letters:
                path_to_letter_dir = os.path.join(path, letter_dir_name)
                list_of_loaded_images = self._load_character_train_data(
                    path_to_letter_dir, num_samples
                )
                dict_of_result[letter_dir_name] = list_of_loaded_images

            elif len(list_of_letters) == 0:
                path_to_letter_dir = os.path.join(path, letter_dir_name)
                list_of_loaded_images = self._load_character_train_data(
                    path_to_letter_dir, num_samples
                )
                dict_of_result[letter_dir_name] = list_of_loaded_images

        return dict_of_result

    def _load_character_train_data(
        self, path_to_char_dir: Path, num_samples: int
    ) -> list:
        list_of_result = list()
        for idx, char_image in enumerate(os.listdir(path_to_char_dir)):
            if idx == num_samples:
                break

            if not char_image.endswith(".txt"):
                list_of_result.append(
                    cv2.imread(os.path.join(path_to_char_dir, char_image), -1)
                )

        return list_of_result

    def get_sea_scrolls_images(self, path: Path) -> list():
        """Get all binarized sea scrolls images.

        Args:
            path: Path to the sea scrolls images.

        Returns:
            list_of_result: List contatining all loaded images.
        """
        list_of_result = list()
        for image in os.listdir(path):
            if image.endswith("binarized.jpg"):
                list_of_result.append(cv2.imread(os.path.join(path, image), 0))

        return list_of_result

    def get_characters_style_based(self, path: Path, type_img="png"):
        dict_of_result = {"Archaic": dict(), "Hasmonean": dict(), "Herodian": dict()}

        for style_folder in os.listdir(path):
            curr_path_style = os.path.join(path, style_folder)
            for letter_folder in os.listdir(curr_path_style):
                curr_path_style_letter = os.path.join(curr_path_style, letter_folder)
                for img in os.listdir(curr_path_style_letter):
                    if img.endswith(type_img):

                        loaded_img = cv2.imread(
                            os.path.join(curr_path_style_letter, img), -1
                        )

                        if letter_folder not in dict_of_result[style_folder].keys():
                            dict_of_result[style_folder][letter_folder] = [loaded_img]
                        else:
                            dict_of_result[style_folder][letter_folder].append(
                                loaded_img
                            )

        return dict_of_result


