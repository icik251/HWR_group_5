from os import path
import os

from PIL import Image
from data_loader import DataLoader
from matplotlib import pyplot as plt
import numpy as np
from utils import save_image
import random

random.seed(42)


class CharacterAugmentation:
    def __init__(self, image, char_name, path_to_save) -> None:
        self.image = image
        self.char_name = char_name
        self.path_to_save = path_to_save
        self.list_of_results = list()

    def apply_noise(self, num_of_augmented=10):
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
            self.list_of_results.append(noisy_array_clipped_int)

            # plt.imshow(noisy_array_clipped)
            # plt.show()

    def apply_rotation(self, boundaries=(-20, 20)):
        degrees_rotation = random.randint(boundaries[0], boundaries[1])

        img_pil = Image.fromarray(self.image)
        img_rotate_pil = img_pil.rotate(degrees_rotation)
        plt.imshow(img_rotate_pil)
        plt.show()

    def apply_elasticity(
        self,
    ):
        pass

    def save_augmented_images(self):
        for idx, agumented_image in enumerate(self.list_of_results):
            save_image(
                agumented_image,
                str(self.char_name) + "_" + str(idx),
                self.path_to_save,
            )


data_loader = DataLoader()
dict_of_results = data_loader.get_characters_train_data(
    "data\\processed_data\\character_recognition\\normalized_avg\\train"
)

for char_name, list_samples in dict_of_results.items():
    for sample in list_samples:
            
        char_augmentation = CharacterAugmentation(
            sample,
            char_name,
            "data\\processed_data\\character_recognition\\normalized_avg\\train_augmented",
        )
        
        char_augmentation.apply_noise()
        # char_augmentation.apply_rotation()
        char_augmentation.save_augmented_images()
