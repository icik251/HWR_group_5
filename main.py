from data_processing import DataProcessing
import os
from data_loader import DataLoader
from torch import nn
import torch.optim as optim
import copy

if __name__ == "__main__":

    # Load the data
    data_loader = DataLoader()
    dict_of_results_recognizer = data_loader.get_characters_train_data(
        "D:\\PythonProjects\\HWR_group_5\\data\\character_set_labeled\\"
    )
    dict_of_results_style = data_loader.get_characters_style_based(
        "D:\\PythonProjects\\HWR_group_5\\data\\style_classification\\characters_for_style_classification\\",
    )

    # Process the data both for recognition and style classification task
    # Process smallest for recognizer
    data_processing = DataProcessing(dict_of_results_recognizer, mode="recognition")
    data_processing.split_train_val_test()
    data_processing.normalize_data(normalization_type="smallest")

    # Process average for recognizer
    data_processing = DataProcessing(dict_of_results_recognizer, mode="recognition")
    data_processing.split_train_val_test()
    data_processing.normalize_data(normalization_type="average")

    # Process smallest for style
    data_processing = DataProcessing(dict_of_results_style, mode="style")
    data_processing.split_train_val_test()
    data_processing.normalize_data(normalization_type="smallest")

    # Process average for style
    data_processing = DataProcessing(dict_of_results_style, mode="style")
    data_processing.split_train_val_test()
    data_processing.normalize_data(normalization_type="average")

    # Augment data after it has been processed
