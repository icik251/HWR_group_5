import os
import cv2
import numpy as np
from data_loader import DataLoader
from utils import (
    crop_white_spaces_image_v2,
    resize_image,
    crop_white_spaces_image,
    save_image,
)
import torch
import torch.utils.data as data_utils


class StyleBasedTrainDataProcessing:
    def __init__(self, dict_of_style_based) -> None:
        self.dict_of_style_based = dict_of_style_based

        self.style2idx = {"Archaic": 0, "Hasmonean": 1, "Herodian": 2}

    def split_train_test(self, train_split=0.9):
        for style_class, dict_of_letters in self.dict_of_style_based.items():
            for letter_key, value_list in dict_of_letters.items():
                idx_boundary = int(len(value_list) * train_split)
                list_of_trains = value_list[:idx_boundary]
                list_of_tests = value_list[idx_boundary:]
                self.dict_of_style_based[style_class][letter_key] = {
                    "train": list_of_trains,
                    "test": list_of_tests,
                }

    def normalize_data(self, save_mode=False):
        new_shape = [0, 0]
        count_samples = 0
        for style_class, dict_of_letters in self.dict_of_style_based.items():
            for letter_key, value_list in dict_of_letters.items():
                for sample in value_list:
                    # resize method expects (width, height), that's why we reverse them here already
                    new_shape[0] += sample.shape[1]
                    new_shape[1] += sample.shape[0]
                    count_samples += 1

        new_shape = np.divide(new_shape, count_samples)
        new_shape = list(np.array(new_shape).astype("uint8"))
        new_shape = tuple(new_shape)

        for style_class, dict_of_letters in self.dict_of_style_based.items():
            for letter_key, value_list in dict_of_letters.items():
                list_of_resized_samples = list()
                for idx, sample in enumerate(value_list):
                    cropped_image = crop_white_spaces_image_v2(sample)
                    resized_image = resize_image(cropped_image, new_shape)

                    if save_mode:
                        path_to_save_image = os.path.join(
                            "data\\style_classification\\character_for_style_classification_normalized",
                            style_class,
                            letter_key,
                        )
                        save_image(resized_image, idx, path_to_save_image)

                    list_of_resized_samples.append(resized_image)
                self.dict_of_style_based[style_class][
                    letter_key
                ] = list_of_resized_samples

    def get_data_loaders(self):
        list_of_train_images = list()
        list_of_train_labels = list()

        list_of_test_images = list()
        list_of_test_labels = list()

        for style_class, dict_of_letters in self.dict_of_style_based.items():
            for letter_key, dict_train_test in dict_of_letters.items():
                list_of_train_images += dict_train_test["train"]
                list_of_train_labels += [self.style2idx[style_class]] * len(
                    dict_train_test["train"]
                )

                list_of_test_images += dict_train_test["test"]
                list_of_test_labels += [self.style2idx[style_class]] * len(
                    dict_train_test["test"]
                )

        train_dataset = CustomDataset(list_of_train_images, list_of_train_labels)
        train_dataloader = data_utils.DataLoader(train_dataset, batch_size=5)

        test_dataset = CustomDataset(list_of_test_images, list_of_test_labels)
        test_dataloader = data_utils.DataLoader(test_dataset, batch_size=1)
        
        return train_dataloader, test_dataloader


# create custom dataset class
class CustomDataset(data_utils.Dataset):
    def __init__(self, samples, labels):
        self.samples = torch.FloatTensor(samples)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        data = self.samples[idx].unsqueeze(0)
        label = self.labels[idx]
        sample = [data, label]
        return sample
