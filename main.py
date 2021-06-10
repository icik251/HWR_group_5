import os
from mnist_model import MNISTModel
from style_based_data_processing import StyleBasedTrainDataProcessing
from data_loader import DataLoader
from model import Model
from torch import nn
import torch.optim as optim
import copy

if __name__ == "__main__":
    """
    seed = 42
    mnist_model = MNISTModel(root_path=".\\data\\MNIST", seed=seed)
    mnist_model.prepate_data()
    path_checkpoints = "D:\\PythonProjects\\HWR_group_5\\data\\MNIST\\MNIST\\checkpoints"
    patience = 7

    epochs = 50
    learning_rate = 0.01
    optimizer = optim.Adam(mnist_model.get_model_params(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    mnist_model.train(epochs, optimizer, criterion, patience, path_checkpoints)
    """
    # Load images for the first time
    data_loader = DataLoader()
    dict_of_results = data_loader.get_characters_style_based(
        "D:\\PythonProjects\\HWR_group_5\\data\\style_classification\\characters_for_style_classification"
    )

    # Pipeline for different experiments
    list_of_size_normalizations = ["padding", "average", "smallest"]
    list_of_train_splits = [0.8, 0.9]
    list_of_batch_sizes = [100]
    list_of_freeze_layers = [True, False]

    for normalization_type in list_of_size_normalizations:
        for train_split in list_of_train_splits:
            for batch_size in list_of_batch_sizes:
                for freeze_bool in list_of_freeze_layers:
                    # Preprocess images to be same size and split train/test
                    style_based_train_data_processing = StyleBasedTrainDataProcessing(
                        copy.deepcopy(dict_of_results)
                    )
                    style_based_train_data_processing.normalize_data(
                        normalization_type, save_mode=True
                    )
                    style_based_train_data_processing.split_train_test(
                        train_split=train_split
                    )
                    (
                        train_loader,
                        test_loader,
                    ) = style_based_train_data_processing.get_data_loaders(
                        batch_size=batch_size
                    )

                    # Train and evaluate model
                    # Load model with activation func and dropout_rate
                    path_to_checkpoint = (
                        "data\\MNIST\\MNIST\\checkpoints\\checkpoint_14.pth"
                    )
                    model_obj = Model(
                        path_to_checkpoint, freeze_layers=freeze_bool, seed=42
                    )

                    # Path to save model and everything realated to it
                    model_folder = "norm_{}_split_{}_batch_{}_freeze_{}".format(
                        normalization_type, train_split, batch_size, freeze_bool
                    )
                    path_to_save_model = os.path.join(
                        "D:\\PythonProjects\\HWR_group_5\\data\\style_classification\\models",
                        model_folder,
                    )

                    patience = 20
                    epochs = 100
                    criterion = nn.CrossEntropyLoss()
                    learning_rate = 0.01
                    optimizer = optim.Adam(
                        model_obj.get_model_params(), lr=learning_rate
                    )

                    # Train
                    model_obj.train(
                        epochs,
                        optimizer,
                        criterion,
                        train_loader,
                        test_loader,
                        patience,
                        path_to_save_model,
                    )
