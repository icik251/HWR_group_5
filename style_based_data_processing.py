import os
import cv2
import math
import numpy as np
import PIL
from torchvision.transforms.transforms import RandomAffine, RandomRotation
from data_loader import DataLoader
from utils import (
    crop_white_spaces_image_v2,
    resize_image,
    crop_white_spaces_image,
    save_image,
    reverse_black_white
)
import torchvision.transforms as transforms
import torch
import torch.utils.data as data_utils


class StyleBasedTrainDataProcessing:
    def __init__(self, dict_of_style_based) -> None:
        self.dict_of_style_based = dict_of_style_based

        self.style2idx = {"Archaic": 0, "Hasmonean": 1, "Herodian": 2}

    def split_train_test(self, train_split=0.7):
        for style_class, dict_of_letters in self.dict_of_style_based.items():
            for letter_key, value_list in dict_of_letters.items():
                idx_boundary = int(len(value_list) * train_split)
                list_of_trains = value_list[:idx_boundary]
                list_of_tests = value_list[idx_boundary:]
                self.dict_of_style_based[style_class][letter_key] = {
                    "train": list_of_trains,
                    "test": list_of_tests,
                }

    def normalize_data_avg_size(self, save_mode=False):
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
                            "data\\style_classification\\character_for_style_classification_normalized_avg",
                            style_class,
                            letter_key,
                        )
                        save_image(resized_image, idx, path_to_save_image)

                    list_of_resized_samples.append(resized_image)
                self.dict_of_style_based[style_class][
                    letter_key
                ] = list_of_resized_samples
                
    def normalize_data_smallest_size(self, save_mode=False):
        new_shape = [1000, 1000]
        for style_class, dict_of_letters in self.dict_of_style_based.items():
            for letter_key, value_list in dict_of_letters.items():
                for sample in value_list:
                    # resize method expects (width, height), that's why we reverse them here already
                    if sample.shape[1] < new_shape[0]:
                        new_shape[0] = sample.shape[1]    
                    if sample.shape[0] < new_shape[1]:
                        new_shape[1] = sample.shape[0]
        
        new_shape = list(np.array(new_shape).astype("uint8"))
        new_shape = tuple(new_shape)
        
        for style_class, dict_of_letters in self.dict_of_style_based.items():
            for letter_key, value_list in dict_of_letters.items():
                list_of_resized_samples = list()
                for idx, sample in enumerate(value_list):
                    #cropped_image = crop_white_spaces_image_v2(sample)
                    resized_image = resize_image(sample, new_shape)

                    if save_mode:
                        path_to_save_image = os.path.join(
                            "data\\style_classification\\character_for_style_classification_normalized_smallest",
                            style_class,
                            letter_key,
                        )
                        save_image(resized_image, idx, path_to_save_image)

                    list_of_resized_samples.append(resized_image)
                self.dict_of_style_based[style_class][
                    letter_key
                ] = list_of_resized_samples
                
    def normalize_data_padding(self, save_mode=False):
        biggest_shape = [0, 0]
        for style_class, dict_of_letters in self.dict_of_style_based.items():
            for letter_key, value_list in dict_of_letters.items():
                for sample in value_list:
                    # resize method expects (width, height), that's why we reverse them here already
                    if sample.shape[1] > biggest_shape[0]:
                        biggest_shape[0] = sample.shape[1]    
                    if sample.shape[0] > biggest_shape[1]:
                        biggest_shape[1] = sample.shape[0]
        
        for style_class, dict_of_letters in self.dict_of_style_based.items():
            for letter_key, value_list in dict_of_letters.items():
                list_of_resized_samples = list()
                for idx, sample in enumerate(value_list):
                    width_diff = biggest_shape[0] - sample.shape[1]
                    height_diff = biggest_shape[1] - sample.shape[0]
                    
                    left_pad, right_pad, top_pad, bottom_pad = self.get_paddings(height_diff, width_diff)
                    
                    padding_adder = torch.nn.ZeroPad2d((left_pad, right_pad, top_pad, bottom_pad))
                    reversed_black_white_sample = reverse_black_white(sample)
                    padded_sample = np.array(padding_adder(torch.from_numpy(reversed_black_white_sample)))
                    rollback_black_white_sample = reverse_black_white(padded_sample)
                    
                    if save_mode:
                        path_to_save_image = os.path.join(
                            "data\\style_classification\\character_for_style_classification_normalized_padded",
                            style_class,
                            letter_key,
                        )
                        save_image(rollback_black_white_sample, idx, path_to_save_image)

                    list_of_resized_samples.append(rollback_black_white_sample)
                self.dict_of_style_based[style_class][
                    letter_key
                ] = list_of_resized_samples
    
    def get_paddings(self, height_diff, width_diff):
        left_pad = math.floor(width_diff / 2)
        right_pad = math.ceil(width_diff / 2)
        
        top_pad = math.floor(height_diff / 2)
        bottom_pad = math.ceil(height_diff / 2)
        
        return left_pad, right_pad, top_pad, bottom_pad
        
    def get_data_loaders(self, batch_size=100):
        # for data augmentation
        transform_train = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.RandomAffine(20),
                transforms.RandomRotation(20, resample=PIL.Image.BILINEAR),
                transforms.Normalize((0.5), (0.5)),
            ]
        )
        
        transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5), (0.5)),
            ]
        )
        
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

        train_dataset = CustomDataset(list_of_train_images, list_of_train_labels, transform=transform_train)
        train_dataloader = data_utils.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        test_dataset = CustomDataset(list_of_test_images, list_of_test_labels, transform=transform_test)
        test_dataloader = data_utils.DataLoader(test_dataset, batch_size=1)
        
        return train_dataloader, test_dataloader



# create custom dataset class
class CustomDataset(data_utils.Dataset):
    def __init__(self, samples, labels, transform=None):
        self.samples = samples
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        data = self.samples[idx]
        label = self.labels[idx]
        
        if self.transform:
            data = self.transform(data)
        
        label = torch.from_numpy(np.array(label))
        label = label.type(torch.LongTensor)
        
        sample = [data, label]
            
        return sample
