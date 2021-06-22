import os
from mnist_model import MNISTModel
from data_prep_tensor import RecognitionDataPrepTensor, StyleDataPrepTensor
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

    # RECOGNITION
    # Load preprocessed images for recognition, smallest method
    """
    data_loader = DataLoader()
    dict_of_reco_train = data_loader.get_characters_train_data(
        "data\\processed_data\\character_recognition\\normalized_smallest\\train"
    )

    dict_of_reco_val = data_loader.get_characters_train_data(
        "data\\processed_data\\character_recognition\\normalized_smallest\\val"
    )

    reco_data_prep = RecognitionDataPrepTensor()
    train_loader, test_loader = reco_data_prep.get_data_loaders_training(
        dict_of_train=dict_of_reco_train,
        dict_of_test=dict_of_reco_val,
        train_batch_size=20,
    )

    normalization_type = "smallest"
    freeze_bool = False

    # Train and evaluate model
    # Load model with activation func and dropout_rate
    path_to_checkpoint = "data\\MNIST\\MNIST\\checkpoints\\checkpoint_14.pth"
    model_obj = Model(mode="recognition", model_path_to_load=None, freeze_layers=freeze_bool, seed=42)

    # Path to save model and everything realated to it
    model_folder = "norm_{}_freeze_{}".format(normalization_type, freeze_bool)

    path_to_save_model = os.path.join(
        "data\\models\\character_recognition",
        model_folder,
    )

    patience = 20
    delta = 0
    epochs = 100
    criterion = nn.CrossEntropyLoss()
    learning_rate = 0.01
    optimizer = optim.Adam(model_obj.get_model_params(), lr=learning_rate)

    # Train
    model_obj.train(
        epochs,
        optimizer,
        criterion,
        train_loader,
        test_loader,
        patience,
        delta,
        path_to_save_model,
    )
    """
    
    # STYLE CLASSIFICATION
    data_loader = DataLoader()
    dict_of_style_train = data_loader.get_characters_style_based(
        "data\\processed_data\\style_classification\\normalized_smallest\\train", type_img='pgm'
    )

    dict_of_style_val = data_loader.get_characters_style_based(
        "data\\processed_data\\style_classification\\normalized_smallest\\val", type_img='pgm'
    )

    style_data_prep = StyleDataPrepTensor()
    train_loader, test_loader = style_data_prep.get_data_loaders_training(
        dict_of_train=dict_of_style_train,
        dict_of_test=dict_of_style_val,
        number_of_samples_per_style=2,
    )

    normalization_type = "smallest"
    freeze_bool = False

    # Train and evaluate model
    # Load model with activation func and dropout_rate
    path_to_checkpoint = "data\\MNIST\\MNIST\\checkpoints\\checkpoint_14.pth"
    model_obj = Model(mode="style", model_path_to_load=None, freeze_layers=freeze_bool, seed=42)

    # Path to save model and everything realated to it
    model_folder = "norm_{}_freeze_{}".format(normalization_type, freeze_bool)

    path_to_save_model = os.path.join(
        "data\\models\\style_classification",
        model_folder,
    )

    patience = 20
    delta = 0
    epochs = 100
    criterion = nn.CrossEntropyLoss()
    learning_rate = 0.01
    optimizer = optim.Adam(model_obj.get_model_params(), lr=learning_rate)

    # Train
    model_obj.train(
        epochs,
        optimizer,
        criterion,
        train_loader,
        test_loader,
        patience,
        delta,
        path_to_save_model,
    )
