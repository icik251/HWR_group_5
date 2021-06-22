import os
import numpy as np
from data_loader import DataLoader

from utils import (
    crop_white_spaces_image_v2,
    resize_image,
    save_image,
)


class DataProcessing:
    def __init__(self, dict_of_images, mode="recognition") -> None:
        """It is used to split and process the data

        Args:
            dict_of_images ([type]): Images loaded from the data loader class
            mode (str, optional): Mode of processing, possible: "recognition" and "style". Defaults to "recognition".
        """

        self.dict_of_images = dict_of_images
        self.mode = mode

    def split_train_val_test(self, train_split=0.8, val_split=0.1):
        if self.mode == "style":
            for style_class, dict_of_letters in self.dict_of_images.items():
                for letter_key, value_list in dict_of_letters.items():
                    idx_boundary_train = int(len(value_list) * train_split)
                    idx_boundary_val = int(len(value_list) * (train_split + val_split))
                    list_of_trains = value_list[:idx_boundary_train]
                    list_of_vals = value_list[idx_boundary_train:idx_boundary_val]
                    list_of_tests = value_list[idx_boundary_val:]
                    self.dict_of_images[style_class][letter_key] = {
                        "train": list_of_trains,
                        "val": list_of_vals,
                        "test": list_of_tests,
                    }

        elif self.mode == "recognition":
            for letter_key, value_list in self.dict_of_images.items():
                idx_boundary_train = int(len(value_list) * train_split)
                idx_boundary_val = int(len(value_list) * (train_split + val_split))
                list_of_trains = value_list[:idx_boundary_train]
                list_of_vals = value_list[idx_boundary_train:idx_boundary_val]
                list_of_tests = value_list[idx_boundary_val:]
                self.dict_of_images[letter_key] = {
                    "train": list_of_trains,
                    "val": list_of_vals,
                    "test": list_of_tests,
                }

    def normalize_data(self, normalization_type="smallest", save_mode=True):
        """Resize the data

        Args:
            normalization_type (str, optional): Mode of resizing, possible modes: "smallest" and "average". Defaults to "smallest".
            save_mode (bool, optional): Save or not the resulted images. Defaults to True.
        """
        if self.mode == "recognition":
            if normalization_type == "average":
                self._normalize_reco_data_avg_size(save_mode=save_mode)
            elif normalization_type == "smallest":
                self._normalize_reco_data_smallest_size(save_mode=save_mode)

        elif self.mode == "style":
            if normalization_type == "average":
                self._normalize_style_data_avg_size(save_mode=save_mode)
            elif normalization_type == "smallest":
                self._normalize_style_data_smallest_size(save_mode=save_mode)

    def save_used_shape(self, shape, path_to_save):
        # create path to save
        if not os.path.exists(path_to_save):
            os.makedirs(path_to_save)

        with open(
            os.path.join(path_to_save, "{}_shape.txt".format(self.mode)), "w"
        ) as f:
            f.write("{},{}".format(shape[0], shape[1]))
        f.close()

    def _normalize_reco_data_avg_size(self, save_mode=True):
        new_shape = [0, 0]
        count_samples = 0
        for char_name, dict_partitions_images in self.dict_of_images.items():
            for sample in dict_partitions_images["train"]:
                try:
                    cropped_train_sample = crop_white_spaces_image_v2(sample)
                except Exception as e:
                    print(e)

                    cropped_train_sample = sample
                # resize method expects (width, height), that's why we reverse them here already
                new_shape[0] += cropped_train_sample.shape[1]
                new_shape[1] += cropped_train_sample.shape[0]
                count_samples += 1

        new_shape = np.divide(new_shape, count_samples)
        new_shape = list(np.array(new_shape).astype("uint8"))
        new_shape = tuple(new_shape)

        for char_name, dict_partitions_images in self.dict_of_images.items():
            for partition_key, list_of_samples in dict_partitions_images.items():
                for idx, sample in enumerate(list_of_samples):
                    try:
                        cropped_image = crop_white_spaces_image_v2(sample)
                    except Exception as e:
                        print(e)
                        cropped_image = sample

                    resized_image = resize_image(cropped_image, new_shape)

                    if save_mode:
                        path_to_save_image = os.path.join(
                            "data\\processed_data\\character_recognition\\normalized_avg",
                            partition_key,
                            char_name,
                        )
                        path_to_save_shape = os.path.join(
                            "data\\resizing_shapes\\normalized_avg",
                            char_name,
                        )
                        save_image(resized_image, idx, path_to_save_image)
                        self.save_used_shape(new_shape, path_to_save_shape)

    def _normalize_reco_data_smallest_size(self, save_mode=True):
        new_shape = [1000, 1000]
        for char_name, dict_partitions_images in self.dict_of_images.items():
            for sample in dict_partitions_images["train"]:
                try:
                    cropped_train_sample = crop_white_spaces_image_v2(sample)
                except Exception as e:
                    print(e)
                    cropped_train_sample = sample

                # resize method expects (width, height), that's why we reverse them here already
                if cropped_train_sample.shape[1] < new_shape[0]:
                    new_shape[0] = cropped_train_sample.shape[1]
                if cropped_train_sample.shape[0] < new_shape[1]:
                    new_shape[1] = cropped_train_sample.shape[0]

        new_shape = list(np.array(new_shape).astype("uint8"))
        new_shape = tuple(new_shape)

        for char_name, dict_partitions_images in self.dict_of_images.items():
            for partition_key, list_of_samples in dict_partitions_images.items():
                for idx, sample in enumerate(list_of_samples):
                    try:
                        cropped_image = crop_white_spaces_image_v2(sample)
                    except Exception as e:
                        print(e)
                        cropped_image = sample

                    resized_image = resize_image(cropped_image, new_shape)

                    if save_mode:
                        path_to_save_image = os.path.join(
                            "data\\processed_data\\character_recognition\\normalized_smallest",
                            partition_key,
                            char_name,
                        )
                        path_to_save_shape = os.path.join(
                            "data\\resizing_shapes\\normalized_smallest",
                            char_name,
                        )
                        save_image(resized_image, idx, path_to_save_image)
                        self.save_used_shape(new_shape, path_to_save_shape)

    def _normalize_style_data_avg_size(self, save_mode=True):
        dict_of_char_shape = dict()

        for style_class, dict_of_letters in self.dict_of_images.items():
            for char_name, dict_partitions_images in dict_of_letters.items():
                for sample in dict_partitions_images["train"]:
                    try:
                        cropped_image = crop_white_spaces_image_v2(sample)
                    except Exception as e:
                        print(e)
                        cropped_image = sample

                    # resize method expects (width, height), that's why we reverse them here already
                    if not char_name in dict_of_char_shape.keys():
                        dict_of_char_shape[char_name] = [
                            [sample.shape[1], sample.shape[0]],
                            1,
                        ]
                    else:
                        dict_of_char_shape[char_name][0][0] += sample.shape[1]
                        dict_of_char_shape[char_name][0][1] += sample.shape[0]
                        # count samples
                        dict_of_char_shape[char_name][1] += 1

        for style_class, dict_of_letters in self.dict_of_images.items():
            for char_name, dict_partitions_images in dict_of_letters.items():
                for partition_key, list_of_samples in dict_partitions_images.items():
                    for idx, sample in enumerate(list_of_samples):
                        try:
                            cropped_image = crop_white_spaces_image_v2(sample)
                        except Exception as e:
                            print(e)
                            cropped_image = sample

                        new_shape = np.divide(
                            dict_of_char_shape[char_name][0],
                            dict_of_char_shape[char_name][1],
                        )
                        new_shape = list(np.array(new_shape).astype("uint8"))
                        new_shape = tuple(new_shape)
                        resized_image = resize_image(cropped_image, new_shape)

                        if save_mode:
                            path_to_save_image = os.path.join(
                                "data\\processed_data\\style_classification\\normalized_avg",
                                partition_key,
                                style_class,
                                char_name,
                            )
                            path_to_save_shape = os.path.join(
                                "data\\resizing_shapes\\normalized_avg",
                                char_name,
                            )
                            save_image(resized_image, idx, path_to_save_image)
                            self.save_used_shape(new_shape, path_to_save_shape)

    def _normalize_style_data_smallest_size(self, save_mode=True):
        dict_of_char_shape = dict()

        for style_class, dict_of_letters in self.dict_of_images.items():
            for char_name, dict_partitions_images in dict_of_letters.items():
                for sample in dict_partitions_images["train"]:
                    try:
                        cropped_train_sample = crop_white_spaces_image_v2(sample)
                    except Exception as e:
                        print(e)
                        cropped_train_sample = sample

                    # resize method expects (width, height), that's why we reverse them here already
                    if not char_name in dict_of_char_shape.keys():
                        dict_of_char_shape[char_name] = [
                            [
                                cropped_train_sample.shape[1],
                                cropped_train_sample.shape[0],
                            ],
                            1,
                        ]
                    else:
                        # resize method expects (width, height), that's why we reverse them here already
                        if (
                            cropped_train_sample.shape[1]
                            < dict_of_char_shape[char_name][0][0]
                        ):
                            dict_of_char_shape[char_name][0][
                                0
                            ] = cropped_train_sample.shape[1]
                        if (
                            cropped_train_sample.shape[0]
                            < dict_of_char_shape[char_name][0][1]
                        ):
                            dict_of_char_shape[char_name][0][
                                1
                            ] = cropped_train_sample.shape[0]

        for style_class, dict_of_letters in self.dict_of_images.items():
            for char_name, dict_partitions_images in dict_of_letters.items():
                for partition_key, list_of_samples in dict_partitions_images.items():
                    for idx, sample in enumerate(list_of_samples):
                        try:
                            cropped_image = crop_white_spaces_image_v2(sample)
                        except Exception as e:
                            print(e)
                            cropped_image = sample

                        new_shape = list(
                            np.array(dict_of_char_shape[char_name][0]).astype("uint8")
                        )
                        new_shape = tuple(new_shape)
                        resized_image = resize_image(cropped_image, new_shape)

                        if save_mode:
                            path_to_save_image = os.path.join(
                                "data\\processed_data\\style_classification\\normalized_smallest",
                                partition_key,
                                style_class,
                                char_name,
                            )
                            path_to_save_shape = os.path.join(
                                "data\\resizing_shapes\\normalized_smallest",
                                char_name,
                            )
                            save_image(resized_image, idx, path_to_save_image)
                            self.save_used_shape(new_shape, path_to_save_shape)


"""
data_loader = DataLoader()
# dict_of_results = data_loader.get_characters_train_data(
#    "D:\\PythonProjects\\HWR_group_5\\data\\character_set_labeled\\"
# )
dict_of_results = data_loader.get_characters_style_based(
    "D:\\PythonProjects\\HWR_group_5\\data\\style_classification\\characters_for_style_classification\\",
)

data_processing = DataProcessing(dict_of_results, mode="style")
data_processing.split_train_val_test()
data_processing.normalize_data(normalization_type="smallest")
"""
