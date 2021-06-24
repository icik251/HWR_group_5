import os
from mnist_model import MNISTModel
from data_prep_tensor import RecognitionDataPrepTensor, StyleDataPrepTensor
from data_loader import DataLoader
from model import Model
from torch import nn
import torch.optim as optim
import copy

# Pipeline doing all variations of experiments for both style classification and character recognition

if __name__ == "__main__":
    # MNIST Training (don't uncomment, already trained)
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

    """
    # RECOGNITION PIPELINE to try different params
    # Load preprocessed images for recognition, smallest method
    
    # Pipeline for different experiments
    list_of_size_normalizations = ["average", "smallest"]
    list_of_batch_sizes = [128, 256]
    list_of_freeze_layers = [True, False]
    list_of_augmented = ["train_augmented", "train"]
    list_of_mnist = [True, False]
    list_of_optimizers = ["Adam", "SGD"]
    list_of_learning_rates = [0.01, 0.001]

    for normalization_type in list_of_size_normalizations:
        for augmented_dir in list_of_augmented:

            normalization_type = "average"
            if normalization_type == "average":
                normalization_type = "avg"

            data_loader = DataLoader()
            dict_of_reco_train = data_loader.get_characters_train_data(
                "data\\processed_data\\character_recognition\\normalized_{}\\{}".format(
                    normalization_type, augmented_dir
                )
            )

            dict_of_reco_val = data_loader.get_characters_train_data(
                "data\\processed_data\\character_recognition\\normalized_{}\\val".format(
                    normalization_type
                )
            )
            for batch_size in list_of_batch_sizes:
                for freeze_bool in list_of_freeze_layers:
                    for is_mnist in list_of_mnist:
                        for optimizer_name in list_of_optimizers:
                            for learning_rate in list_of_learning_rates:
                                
                                # without augmented batch size set to smt low
                                if augmented_dir == "train":
                                    batch_size = 32
                                    
                                if is_mnist:
                                    path_to_checkpoint = "data\\MNIST\\MNIST\\checkpoints\\checkpoint_14.pth"
                                else:
                                    path_to_checkpoint = None
                                    freeze_bool = False
                                    
                                # Path to save model and everything realated to it
                                model_folder = "norm_{}_batch_{}_augmented_{}_mnist_{}_freeze_{}_optim_{}_lr_{}".format(
                                    normalization_type,
                                    batch_size,
                                    augmented_dir,
                                    is_mnist,
                                    freeze_bool,
                                    optimizer_name,
                                    learning_rate,
                                )

                                path_to_save_model = os.path.join(
                                    "data\\models\\character_recognition",
                                    model_folder,
                                )
                                
                                # Check if variation of parameters for model is trained already
                                if os.path.exists(path_to_save_model):
                                    print("Model {} already trained. Skipping...".format(model_folder))
                                    break
                                    
                                reco_data_prep = RecognitionDataPrepTensor()
                                (
                                    train_loader,
                                    test_loader,
                                ) = reco_data_prep.get_data_loaders_training(
                                    dict_of_train=copy.deepcopy(dict_of_reco_train),
                                    dict_of_test=copy.deepcopy(dict_of_reco_val),
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
                                epochs = 100
                                criterion = nn.CrossEntropyLoss()

                                if optimizer_name == "Adam":
                                    optimizer = optim.Adam(
                                        model_obj.get_model_params(), lr=learning_rate
                                    )
                                elif optimizer_name == "SGD":
                                    optimizer = optim.SGD(
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
                                    delta,
                                    path_to_save_model,
                                )
    
    """
    # STYLE CLASSIFICATION
    # SYLE CLASSIFICATION to try different params
    # Load preprocessed images for recognition, smallest method

    # Pipeline for different experiments
    list_of_size_normalizations = ["average", "smallest"]
    list_of_batch_sizes = [100, 300]
    list_of_freeze_layers = [True, False]
    list_of_augmented = ["train_augmented"]
    list_of_mnist = [True, False]
    list_of_optimizers = ["Adam", "SGD"]
    list_of_learning_rates = [0.01, 0.001]

    for normalization_type in list_of_size_normalizations:
        for augmented_dir in list_of_augmented:

            if normalization_type == "average":
                normalization_type = "avg"

            data_loader = DataLoader()
            dict_of_style_train = data_loader.get_characters_style_based(
                "data\\processed_data\\style_classification\\normalized_{}\\{}".format(
                    normalization_type, augmented_dir
                ), type_img="pgm"
            )

            dict_of_style_val = data_loader.get_characters_style_based(
                "data\\processed_data\\style_classification\\normalized_{}\\val".format(
                    normalization_type
                ), type_img="pgm"
            )
            for batch_size in list_of_batch_sizes:
                for freeze_bool in list_of_freeze_layers:
                    for is_mnist in list_of_mnist:
                        for optimizer_name in list_of_optimizers:
                            for learning_rate in list_of_learning_rates:
                                
                                if is_mnist:
                                    path_to_checkpoint = "data\\MNIST\\MNIST\\checkpoints\\checkpoint_14.pth"
                                else:
                                    path_to_checkpoint = None
                                    freeze_bool = False
                                
                                # Path to save model and everything realated to it
                                model_folder = "norm_{}_batch_{}_augmented_{}_mnist_{}_freeze_{}_optim_{}_lr_{}".format(
                                    normalization_type,
                                    batch_size,
                                    augmented_dir,
                                    is_mnist,
                                    freeze_bool,
                                    optimizer_name,
                                    learning_rate,
                                )
                                
                                path_to_save_model = os.path.join(
                                    "data\\models\\style_classification",
                                    model_folder,
                                )
                                
                                # Check if variation of parameters for model is trained already
                                if os.path.exists(path_to_save_model):
                                    print("Model {} already trained. Skipping...".format(model_folder))
                                    break
                                    
                                style_data_prep = StyleDataPrepTensor()
                                (
                                    train_loader,
                                    test_loader,
                                ) = style_data_prep.get_data_loaders_training(
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
                                epochs = 100
                                criterion = nn.CrossEntropyLoss()

                                if optimizer_name == "Adam":
                                    optimizer = optim.Adam(
                                        model_obj.get_model_params(), lr=learning_rate
                                    )
                                elif optimizer_name == "SGD":
                                    optimizer = optim.SGD(
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
                                    delta,
                                    path_to_save_model,
                                )