import os
from data_loader import DataLoader
from mnist_model import MNISTModel
from data_prep_tensor import RecognitionDataPrepTensor, StyleDataPrepTensor
from model import Model
from torch import nn
import torch.optim as optim
import copy

if __name__ == "__main__":
    
    """
    # RECOGNITION final model - BEST MODEL
    # norm_smallest_batch_32_augmented_train_mnist_True_freeze_False_optim_SGD_lr_0.01
    final_recognition_model_name = "norm_smallest_batch_32_augmented_train_mnist_True_freeze_False_optim_SGD_lr_0.01"
    
    list_of_splitted = final_recognition_model_name.split("_")
    normalization_type = list_of_splitted[1]
    batch_size = int(list_of_splitted[3])
    augmented_dir = "train_val_non_augmented"
    is_mnist = (list_of_splitted[7])
    freeze_bool = list_of_splitted[9]
    optimizer_name = list_of_splitted[11]
    learning_rate = float(list_of_splitted[13])
    
    if is_mnist == "True":
        is_mnist = True
    else:
        is_mnist = False
        
    if freeze_bool == "True":
        freeze_bool = True
    else:
        freeze_bool = False

    data_loader = DataLoader()
    dict_of_style_train = data_loader.get_characters_train_data(
        "data\\processed_data\\character_recognition\\normalized_{}\\{}".format(
            normalization_type, augmented_dir
        )
    )

    dict_of_style_val = data_loader.get_characters_train_data(
        "data\\processed_data\\character_recognition\\normalized_{}\\test".format(
            normalization_type
        )
    )

    if is_mnist:
        path_to_checkpoint = "data\\MNIST\\MNIST\\checkpoints\\checkpoint_14.pth"
    else:
        path_to_checkpoint = None
        freeze_bool = False

    # Path to save model and everything realated to it
    model_folder = (
        "norm_{}_batch_{}_augmented_{}_mnist_{}_freeze_{}_optim_{}_lr_{}".format(
            normalization_type,
            batch_size,
            augmented_dir,
            is_mnist,
            freeze_bool,
            optimizer_name,
            learning_rate,
        )
    )

    path_to_save_model = os.path.join(
        "data\\models\\character_recognition_final",
        model_folder,
    )

    # Check if variation of parameters for model is trained already
    if os.path.exists(path_to_save_model):
        print("Model {} already trained. Skipping...".format(model_folder))

    style_data_prep = RecognitionDataPrepTensor()
    (train_loader, test_loader,) = style_data_prep.get_data_loaders_training(
        dict_of_train=copy.deepcopy(dict_of_style_train),
        dict_of_test=copy.deepcopy(dict_of_style_val),
        train_batch_size=batch_size,
    )

    model_obj = Model(
        mode="recognition",
        model_path_to_load=path_to_checkpoint,
        freeze_layers=freeze_bool,
        seed=42,
    )

    patience = 15
    delta = 0
    epochs = 300
    criterion = nn.CrossEntropyLoss()

    if optimizer_name == "Adam":
        optimizer = optim.Adam(model_obj.get_model_params(), lr=learning_rate)
    elif optimizer_name == "SGD":
        optimizer = optim.SGD(model_obj.get_model_params(), lr=learning_rate)

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
    
    
    # STYLE final model training
    final_style_model_name = ""
    
    list_of_splitted = final_style_model_name.split("_")
    normalization_type = list_of_splitted[1]
    batch_size = int(list_of_splitted[3])
    augmented_dir = "train_val_augmented"
    is_mnist = (list_of_splitted[8])
    freeze_bool = list_of_splitted[10]
    optimizer_name = list_of_splitted[12]
    learning_rate = float(list_of_splitted[14])
    
    if is_mnist == "True":
        is_mnist = True
    else:
        is_mnist = False
        
    if freeze_bool == "True":
        freeze_bool = True
    else:
        freeze_bool = False

    data_loader = DataLoader()
    dict_of_style_train = data_loader.get_characters_style_based(
        "data\\processed_data\\style_classification\\normalized_{}\\{}".format(
            normalization_type, augmented_dir
        ),
        type_img="pgm",
    )

    dict_of_style_val = data_loader.get_characters_style_based(
        "data\\processed_data\\style_classification\\normalized_{}\\test".format(
            normalization_type
        ),
        type_img="pgm",
    )

    if is_mnist:
        path_to_checkpoint = "data\\MNIST\\MNIST\\checkpoints\\checkpoint_14.pth"
    else:
        path_to_checkpoint = None
        freeze_bool = False

    # Path to save model and everything realated to it
    model_folder = (
        "norm_{}_batch_{}_augmented_{}_mnist_{}_freeze_{}_optim_{}_lr_{}".format(
            normalization_type,
            batch_size,
            augmented_dir,
            is_mnist,
            freeze_bool,
            optimizer_name,
            learning_rate,
        )
    )

    path_to_save_model = os.path.join(
        "data\\models\\style_classification_final",
        model_folder,
    )

    # Check if variation of parameters for model is trained already
    if os.path.exists(path_to_save_model):
        print("Model {} already trained. Skipping...".format(model_folder))

    style_data_prep = StyleDataPrepTensor()
    (train_loader, test_loader,) = style_data_prep.get_data_loaders_training(
        dict_of_train=copy.deepcopy(dict_of_style_train),
        dict_of_test=copy.deepcopy(dict_of_style_val),
        train_batch_size=batch_size,
    )

    model_obj = Model(
        mode="style",
        model_path_to_load=path_to_checkpoint,
        freeze_layers=freeze_bool,
        seed=42,
    )

    patience = 15
    delta = 0
    epochs = 300
    criterion = nn.CrossEntropyLoss()

    if optimizer_name == "Adam":
        optimizer = optim.Adam(model_obj.get_model_params(), lr=learning_rate)
    elif optimizer_name == "SGD":
        optimizer = optim.SGD(model_obj.get_model_params(), lr=learning_rate)

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