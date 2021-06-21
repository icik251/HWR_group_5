from os import path
import os
from data_loader import DataLoader
from matplotlib import pyplot as plt
import numpy as np
from utils import save_image


class CharacterAugmentation:
    def __init__(self, image, char_name, path_to_save) -> None:
        self.image = image
        self.char_name = char_name
        self.path_to_save = path_to_save
        self.dict_of_results = {"noise": list(), "rotation": list(), "sheering": list()}

    def noise_addition(self, num_of_augmented=10):
        size_array_letter = self.image.shape

        mean_array_letter = np.mean(self.image)
        var_array_letter = np.var(self.image)

        for _ in range(num_of_augmented):
            noisy_array = (
                self.image
                + np.sqrt(var_array_letter)
                * np.random.randn(size_array_letter[0], size_array_letter[1])
                + mean_array_letter
            )
            noisy_array_clipped = np.clip(
                noisy_array, 0, 255
            )  # we might get out of bounds due to noise

            noisy_array_clipped_int = np.rint(noisy_array_clipped).astype(np.uint8)
            self.dict_of_results["noise"].append(noisy_array_clipped_int)

            # plt.imshow(noisy_array_clipped)
            # plt.show()

    def save_augmented_images(self):
        for key_method, augmented_list in self.dict_of_results.items():
            for idx, agumented_image in enumerate(augmented_list):
                save_image(
                    agumented_image,
                    str(key_method) + "_" + str(idx),
                    os.path.join(self.path_to_save, self.char_name),
                )


"""
data_loader = DataLoader()
dict_of_results = data_loader.get_characters_train_data(
    "D:\\PythonProjects\\HWR_group_5\\data\\character_set_labeled\\", num_samples=10
)

char_augmentation = CharacterAugmentation(
    dict_of_results["Alef"][0],
    "Alef",
    "D:\\PythonProjects\\HWR_group_5\\data\\character_set_labeled_augmented\\",
)
char_augmentation.noise_addition()
char_augmentation.save_augmented_images()
"""
