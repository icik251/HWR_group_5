import os
import numpy as np
import torchvision.transforms as transforms
import torch
import torch.utils.data as data_utils


class RecognitionDataPrepTensor:
    def __init__(self) -> None:
        self.char2idx = {
            "Alef": 0,
            "Ayin": 1,
            "Bet": 2,
            "Dalet": 3,
            "Gimel": 4,
            "He": 5,
            "Het": 6,
            "Kaf": 7,
            "Kaf-final": 8,
            "Lamed": 9,
            "Mem": 10,
            "Mem-medial": 11,
            "Nun-final": 12,
            "Nun-medial": 13,
            "Pe": 14,
            "Pe-final": 15,
            "Qof": 16,
            "Resh": 17,
            "Samekh": 18,
            "Shin": 19,
            "Taw": 20,
            "Tet": 21,
            "Tsadi-final": 22,
            "Tsadi-medial": 23,
            "Waw": 24,
            "Yod": 25,
            "Zayin": 26,
        }

    def get_data_loader_production(self, list_of_images):
        transform_production = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5), (0.5)),
            ]
        )

        # To use the Dataset class without changing it
        fake_labels = [0] * len(list_of_images)

        production_dataset = CustomDataset(
            list_of_images, fake_labels, transform=transform_production
        )
        production_dataloader = data_utils.DataLoader(
            production_dataset, batch_size=1, shuffle=False
        )

        return production_dataloader

    def get_data_loaders_training(
        self, dict_of_train, dict_of_test, train_batch_size=100
    ):
        # for data augmentation
        transform_train = transforms.Compose(
            [
                transforms.ToTensor(),
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

        for train_letter_key, list_of_letter_train_samples in dict_of_train.items():
            list_of_train_images += list_of_letter_train_samples
            list_of_train_labels += [self.char2idx[train_letter_key]] * len(
                list_of_letter_train_samples
            )

        for test_letter_key, list_of_letter_test_samples in dict_of_test.items():
            list_of_test_images += list_of_letter_test_samples
            list_of_test_labels += [self.char2idx[test_letter_key]] * len(
                list_of_letter_test_samples
            )

        train_dataset = CustomDataset(
            list_of_train_images, list_of_train_labels, transform=transform_train
        )
        train_dataloader = data_utils.DataLoader(
            train_dataset, batch_size=train_batch_size, shuffle=True
        )

        test_dataset = CustomDataset(
            list_of_test_images, list_of_test_labels, transform=transform_test
        )
        test_dataloader = data_utils.DataLoader(
            test_dataset, batch_size=1, shuffle=False
        )

        return train_dataloader, test_dataloader


class StyleDataPrepTensor:
    def __init__(self) -> None:
        self.style2idx = {"Archaic": 0, "Hasmonean": 1, "Herodian": 2}

    def get_data_loader_production(self, list_of_images):
        transform_production = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5), (0.5)),
            ]
        )

        # To use the Dataset class without changing it
        fake_labels = [0] * len(list_of_images)

        production_dataset = CustomDataset(
            list_of_images, fake_labels, transform=transform_production
        )
        production_dataloader = data_utils.DataLoader(
            production_dataset, batch_size=1, shuffle=False
        )

        return production_dataloader

    def get_data_loaders_training(
        self, dict_of_train, dict_of_test, number_of_samples_per_style=1000
    ):
        # calculating this because we want each batch to contain only one letter and its multiple styles.
        # Because the letter images are resized with respect to their class.
        train_batch_size = 3 * number_of_samples_per_style
        
        # for data augmentation
        transform_train = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5), (0.5)),
            ]
        )
        transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5), (0.5)),
            ]
        )

        dict_of_regrouped_train = dict()

        for train_style_class, dict_of_train_letters in dict_of_train.items():
            for (
                train_letter_key,
                list_of_letter_train_samples,
            ) in dict_of_train_letters.items():

                list_of_letter_train_labels = [self.style2idx[train_style_class]] * len(
                    list_of_letter_train_samples
                )

                if train_letter_key not in dict_of_regrouped_train.keys():
                    dict_of_regrouped_train[train_letter_key] = (
                        list_of_letter_train_samples,
                        list_of_letter_train_labels,
                    )
                else:
                    dict_of_regrouped_train[train_letter_key][
                        0
                    ] += list_of_letter_train_samples
                    dict_of_regrouped_train[train_letter_key][
                        1
                    ] += list_of_letter_train_labels
        
        dict_of_regrouped_test = dict()

        for test_style_class, dict_of_test_letters in dict_of_test.items():
            for (
                test_letter_key,
                list_of_letter_test_samples,
            ) in dict_of_test_letters.items():

                list_of_letter_test_labels = [self.style2idx[test_style_class]] * len(
                    list_of_letter_test_samples
                )

                if test_letter_key not in dict_of_regrouped_test.keys():
                    dict_of_regrouped_test[test_letter_key] = (
                        list_of_letter_test_samples,
                        list_of_letter_test_labels,
                    )
                else:
                    dict_of_regrouped_test[test_letter_key][
                        0
                    ] += list_of_letter_test_samples
                    dict_of_regrouped_test[test_letter_key][
                        1
                    ] += list_of_letter_test_labels


        list_of_train_images = list()
        list_of_train_labels = list()

        list_of_test_images = list()
        list_of_test_labels = list()
        
        for train_samples_labels in dict_of_regrouped_train.values():
            list_of_train_images += train_samples_labels[0]
            list_of_train_labels += train_samples_labels[1]
            
        for test_samples_labels in dict_of_regrouped_test.values():
            list_of_test_images += test_samples_labels[0]
            list_of_test_labels += test_samples_labels[1]
        
        train_dataset = CustomDataset(
            list_of_train_images, list_of_train_labels, transform=transform_train
        )
        train_dataloader = data_utils.DataLoader(
            train_dataset, batch_size=train_batch_size, shuffle=False
        )

        test_dataset = CustomDataset(
            list_of_test_images, list_of_test_labels, transform=transform_test
        )
        test_dataloader = data_utils.DataLoader(
            test_dataset, batch_size=1, shuffle=False
        )

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
