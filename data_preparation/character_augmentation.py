from os import path
import os

from PIL import Image
from data_loader import DataLoader
from matplotlib import pyplot as plt
import numpy as np
from utils import save_image
import random

random.seed(42)

global num_of_rotations_tried
global num_of_rotations_completed

num_of_rotations_tried = 0
num_of_rotations_completed = 0


class CharacterAugmentation:
    def __init__(
        self, image, char_name, num_of_augmented, path_to_save, style_class=None
    ) -> None:
        self.image = image
        self.char_name = char_name
        self.num_of_augmented = num_of_augmented
        self.path_to_save = path_to_save
        self.list_of_results = list()
        self.style_class = style_class

    def apply_noise(self):
        size_array_letter = self.image.shape

        mean_array_letter = np.mean(self.image)
        var_array_letter = np.var(self.image)

        for _ in range(self.num_of_augmented):
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

    def apply_rotation(self, image, boundaries=(-20, 20)):
        theta = random.randint(boundaries[0], boundaries[1])

        im = Image.fromarray(image)
        im_rotate = im.rotate(theta)
        # plt.imshow(im_rotate)
        # plt.show()
        rotated_image = np.array(im_rotate)

        stop_top_list = list()
        stop_bottom_list = list()
        stop_left_list = list()
        stop_right_list = list()

        try:

            for x in range(rotated_image.shape[0]):
                if rotated_image[x][0] == 255:
                    stop_left_list.append(x)

            for x in range(rotated_image.shape[0]):
                if rotated_image[x][rotated_image.shape[1] - 1] == 255:
                    stop_right_list.append(x)

            for y in range(rotated_image.shape[1]):
                if rotated_image[0][y] == 255:
                    stop_top_list.append(y)

            for y in range(rotated_image.shape[1]):
                if rotated_image[rotated_image.shape[0] - 1][y] == 255:
                    stop_bottom_list.append(y)

            for i in range(0, min(stop_left_list)):
                for k in range(0, (min(stop_top_list))):
                    if (
                        min(stop_top_list)
                        - (
                            i
                            * (
                                np.sqrt(
                                    (1 - np.square(np.cos(theta)))
                                    / (np.square(np.cos(theta)))
                                )
                            )
                        )
                    ) >= k - 5:

                        rotated_image[i][k] = 255

            for i in range(0, min(stop_right_list)):
                for k in range(0, rotated_image.shape[1] - max(stop_top_list)):
                    if (
                        rotated_image.shape[1]
                        - max(stop_top_list)
                        - (
                            i
                            * (
                                np.sqrt(
                                    (1 - np.square(np.sin(theta)))
                                    / (np.square(np.sin(theta)))
                                )
                            )
                        )
                    ) >= k - 5:
                        rotated_image[i][rotated_image.shape[1] - k - 1] = 255

            for i in range(0, rotated_image.shape[0] - max(stop_left_list)):
                for k in range(0, min(stop_bottom_list)):
                    if (
                        min(stop_bottom_list)
                        - (
                            i
                            * np.sqrt((1 - np.cos(np.square(theta))) / np.square(theta))
                        )
                    ) >= k - 5:
                        rotated_image[rotated_image.shape[0] - 1 - i][k] = 255

            for i in range(0, rotated_image.shape[0] - max(stop_right_list)):
                for k in range(0, rotated_image.shape[1] - max(stop_bottom_list)):
                    if (
                        rotated_image.shape[1]
                        - max(stop_top_list)
                        - (
                            i
                            * np.sqrt((1 - np.cos(np.square(theta))) / np.square(theta))
                        )
                    ) >= k - 5:
                        rotated_image[rotated_image.shape[0] - 1 - i][
                            rotated_image.shape[1] - 1 - k
                        ] = 255

            # plt.imshow(rotated_image)
            # plt.show()

        except Exception as e:
            print(e)
            return image, 0
            # plt.imshow(rotated_image)
            # plt.show()

        return rotated_image, 1

    def apply_elasticity(
        self,
    ):
        pass

    def logic(self):
        self.apply_noise()

        for idx in range(len(self.list_of_results)):
            random_bit = random.getrandbits(1)
            if random_bit:
                global num_of_rotations_tried
                num_of_rotations_tried += 1
                rotated_image, is_succeed = self.apply_rotation(
                    self.list_of_results[idx]
                )

                # check if rotation is completed
                if is_succeed:
                    global num_of_rotations_completed
                    num_of_rotations_completed += 1

            else:
                rotated_image = self.list_of_results[idx]

            self.list_of_results[idx] = rotated_image

    def save_augmented_images(self, list_of_names_idx):
        if self.style_class is not None:
            path_to_save = os.path.join(
                self.path_to_save, self.style_class, str(self.char_name)
            )
        else:
            path_to_save = os.path.join(self.path_to_save, str(self.char_name))

        for agumented_image, idx in zip(self.list_of_results, list_of_names_idx):
            save_image(
                agumented_image,
                str(idx),
                path_to_save,
            )


"""
# CHARACTER RECOGNITION train for hyper-parameter tuning
data_loader = DataLoader()
dict_of_results = data_loader.get_characters_train_data(
    "data\\processed_data\\character_recognition\\normalized_smallest\\train"
)

max_number_of_samples_in_the_set = len(dict_of_results["Alef"])
num_of_augmented_per_sample = 20

# Replicate samples which don't have the required number

for char_name, list_of_samples in dict_of_results.items():
    new_list_samples = list_of_samples
    while len(new_list_samples) < max_number_of_samples_in_the_set:
        new_list_samples += list_of_samples

    new_list_samples = new_list_samples[:max_number_of_samples_in_the_set]
    dict_of_results[char_name] = new_list_samples

for char_name, list_samples in dict_of_results.items():

    num_of_augmented_per_class = 0
    start_idx_name = 0

    for sample in list_samples:
        char_augmentation = CharacterAugmentation(
            sample,
            char_name,
            num_of_augmented_per_sample,
            "data\\processed_data\\character_recognition\\normalized_smallest\\train_augmented",
        )

        num_of_augmented_per_class += num_of_augmented_per_sample

        list_of_idx_names = list(
            range(
                num_of_augmented_per_class - num_of_augmented_per_sample,
                num_of_augmented_per_class,
            )
        )

        char_augmentation.logic()
        char_augmentation.save_augmented_images(list_of_idx_names)

        if num_of_augmented_per_class % 1000 == 0:
            print(
                "Number of augmented {} for character {}".format(
                    num_of_augmented_per_class, char_name
                )
            )

    print(
        "Augmentation completed for {}. Rotations completed/tried: {}/{}, Number of augmented: {}".format(
            char_name,
            num_of_rotations_completed,
            num_of_rotations_tried,
            num_of_augmented_per_class,
        )
    )

    num_of_rotations_tried = 0
    num_of_rotations_completed = 0
    """

# CHARACTER RECOGNITION concat train and val without data augmentation as this is the final model
data_loader = DataLoader()
dict_of_results_train = data_loader.get_characters_train_data(
    "data\\processed_data\\character_recognition\\normalized_smallest\\train"
)
dict_of_results_val = data_loader.get_characters_train_data(
    "data\\processed_data\\character_recognition\\normalized_smallest\\val"
)

base_path = "data\\processed_data\\character_recognition\\normalized_smallest\\train_val_augmented"

# Merge train and val dicts before replicating
for letter_key, samples_list in dict_of_results_train.items():
    if letter_key in dict_of_results_val.keys():
        dict_of_results_train[letter_key] += dict_of_results_val[letter_key]
            
for letter_key, samples_list in dict_of_results_train.items():
    if not os.path.exists(os.path.join(base_path, letter_key)):
        os.makedirs(os.path.join(base_path, letter_key))
    
    for idx, sample in enumerate(samples_list):
        save_image(sample, idx, os.path.join(base_path, letter_key))

"""
## STYLE CLASSIFICATION train split for hyperparameter tuning

data_loader = DataLoader()
dict_of_results = data_loader.get_characters_style_based(
    "data\\processed_data\\style_classification\\normalized_avg\\train",
    type_img="pgm",
)

dict_of_char_num_of_samples_per_style = dict()
for style_class, dict_of_letters in dict_of_results.items():
    for letter_key, samples_list in dict_of_letters.items():
        if letter_key not in dict_of_char_num_of_samples_per_style.keys():
            dict_of_char_num_of_samples_per_style[letter_key] = {
                "Archaic": 0,
                "Hasmonean": 0,
                "Herodian": 0,
            }

        dict_of_char_num_of_samples_per_style[letter_key][style_class] = len(
            samples_list
        )

# Set number of samples per class per letter (60 samples per letter before augmenting)
for (
    char_key,
    dict_of_styles_num_samples,
) in dict_of_char_num_of_samples_per_style.items():
    for style_key, num_samples in dict_of_styles_num_samples.items():
        if style_key == "Archaic" and num_samples > 0:
            dict_of_char_num_of_samples_per_style[char_key][style_key] = 20
        elif style_key == "Archaic" and num_samples == 0:
            dict_of_char_num_of_samples_per_style[char_key]["Hasmonean"] = 30
            dict_of_char_num_of_samples_per_style[char_key]["Herodian"] = 30


num_of_augmented_per_sample = 20

# Replicate samples which don't have the required number
for style_class, dict_of_letters in dict_of_results.items():
    for letter_key, samples_list in dict_of_letters.items():
        new_list_samples = samples_list
        while (
            len(new_list_samples)
            < dict_of_char_num_of_samples_per_style[letter_key][style_class]
        ):
            new_list_samples += samples_list

        new_list_samples = new_list_samples[
            : dict_of_char_num_of_samples_per_style[letter_key][style_class]
        ]

        dict_of_results[style_class][letter_key] = new_list_samples


for style_class, dict_of_letters in dict_of_results.items():
    for char_name, list_samples in dict_of_letters.items():

        num_of_augmented_per_class = 0
        start_idx_name = 0

        for sample in list_samples:
            char_augmentation = CharacterAugmentation(
                sample,
                char_name,
                num_of_augmented_per_sample,
                "data\\processed_data\\style_classification\\normalized_avg\\train_augmented",
                style_class,
            )

            num_of_augmented_per_class += num_of_augmented_per_sample

            list_of_idx_names = list(
                range(
                    num_of_augmented_per_class - num_of_augmented_per_sample,
                    num_of_augmented_per_class,
                )
            )

            char_augmentation.logic()
            char_augmentation.save_augmented_images(list_of_idx_names)

            if num_of_augmented_per_class % 100 == 0:
                print(
                    "Number of augmented {} for character {}".format(
                        num_of_augmented_per_class, char_name
                    )
                )

        print(
            "Augmentation completed for {}. Rotations completed/tried: {}/{}, Number of augmented: {}".format(
                char_name,
                num_of_rotations_completed,
                num_of_rotations_tried,
                num_of_augmented_per_class,
            )
        )

        num_of_rotations_tried = 0
        num_of_rotations_completed = 0
        """

"""
## STYLE CLASSIFICATION train-val training set combination for final model training
data_loader = DataLoader()
dict_of_results_train = data_loader.get_characters_style_based(
    "data\\processed_data\\style_classification\\normalized_avg\\train",
    type_img="pgm",
)

dict_of_results_val = data_loader.get_characters_style_based(
    "data\\processed_data\\style_classification\\normalized_avg\\val",
    type_img="pgm",
)


dict_of_char_num_of_samples_per_style = dict()
for style_class, dict_of_letters in dict_of_results_train.items():
    for letter_key, samples_list in dict_of_letters.items():
        if letter_key not in dict_of_char_num_of_samples_per_style.keys():
            dict_of_char_num_of_samples_per_style[letter_key] = {
                "Archaic": 0,
                "Hasmonean": 0,
                "Herodian": 0,
            }

        dict_of_char_num_of_samples_per_style[letter_key][style_class] = len(
            samples_list
        )

        if letter_key in dict_of_results_val[style_class].keys():
            dict_of_char_num_of_samples_per_style[letter_key][style_class] += len(
                dict_of_results_val[style_class][letter_key]
            )

# Set number of samples per class per letter (66 samples per letter before augmenting)
for (
    char_key,
    dict_of_styles_num_samples,
) in dict_of_char_num_of_samples_per_style.items():
    for style_key, num_samples in dict_of_styles_num_samples.items():
        if style_key == "Archaic" and num_samples > 0:
            dict_of_char_num_of_samples_per_style[char_key][style_key] = 22
        elif style_key == "Archaic" and num_samples == 0:
            dict_of_char_num_of_samples_per_style[char_key]["Hasmonean"] = 33
            dict_of_char_num_of_samples_per_style[char_key]["Herodian"] = 33


num_of_augmented_per_sample = 20

# Merge train and val dicts before replicating
for style_class, dict_of_letters in dict_of_results_train.items():
    for letter_key, samples_list in dict_of_letters.items():
        if letter_key in dict_of_results_val[style_class].keys():
            dict_of_letters[letter_key] += dict_of_results_val[style_class][letter_key]
            dict_of_results_train[style_class] = dict_of_letters
        
# Replicate samples which don't have the required number
for style_class, dict_of_letters in dict_of_results_train.items():
    for letter_key, samples_list in dict_of_letters.items():
        new_list_samples = samples_list
        while (
            len(new_list_samples)
            < dict_of_char_num_of_samples_per_style[letter_key][style_class]
        ):
            new_list_samples += samples_list

        new_list_samples = new_list_samples[
            : dict_of_char_num_of_samples_per_style[letter_key][style_class]
        ]

        dict_of_results_train[style_class][letter_key] = new_list_samples


for style_class, dict_of_letters in dict_of_results_train.items():
    for char_name, list_samples in dict_of_letters.items():

        num_of_augmented_per_class = 0
        start_idx_name = 0

        for sample in list_samples:
            char_augmentation = CharacterAugmentation(
                sample,
                char_name,
                num_of_augmented_per_sample,
                "data\\processed_data\\style_classification\\normalized_avg\\train_val_augmented",
                style_class,
            )

            num_of_augmented_per_class += num_of_augmented_per_sample

            list_of_idx_names = list(
                range(
                    num_of_augmented_per_class - num_of_augmented_per_sample,
                    num_of_augmented_per_class,
                )
            )

            char_augmentation.logic()
            char_augmentation.save_augmented_images(list_of_idx_names)

            if num_of_augmented_per_class % 100 == 0:
                print(
                    "Number of augmented {} for character {}".format(
                        num_of_augmented_per_class, char_name
                    )
                )

        print(
            "Augmentation completed for {}. Rotations completed/tried: {}/{}, Number of augmented: {}".format(
                char_name,
                num_of_rotations_completed,
                num_of_rotations_tried,
                num_of_augmented_per_class,
            )
        )

        num_of_rotations_tried = 0
        num_of_rotations_completed = 0
        

# Delete random samples from the augmented images to make them 400
base_path = "data\\processed_data\\style_classification\\normalized_avg\\train_val_augmented"

for style_folder in os.listdir(base_path):
    for letter_dir in os.listdir(os.path.join(base_path, style_folder)):
        full_path_to_curr_samples = os.path.join(base_path, style_folder, letter_dir)
        samples_dir_iter = os.listdir(full_path_to_curr_samples)
        num_of_samples = len(samples_dir_iter)
        if num_of_samples == 440:
            num_of_samples_to_be_removed = 40
        else:
            num_of_samples_to_be_removed = 60
            
        list_of_to_be_removed_samples = random.sample(range(0, num_of_samples-1), num_of_samples_to_be_removed)
        
        for sample in samples_dir_iter:
            if int(sample.split('.')[0]) in list_of_to_be_removed_samples:
                os.remove(os.path.join(full_path_to_curr_samples, sample))
                
                """