import os
from mnist_model import MNISTModel
from data_prep_tensor import RecognitionDataPrepTensor, StyleDataPrepTensor
from data_loader import DataLoader
from model import Model
from torch import nn
import torch.optim as optim
import copy

#norm_avg_batch_300_augmented_train_augmented_mnist_True_freeze_False_optim_Adam_lr_0.001
list_of_size_normalizations = ["average"]
list_of_batch_sizes = [300]
list_of_freeze_layers = [False]
list_of_augmented = ["train_augmented"]
list_of_mnist = [True]
list_of_optimizers = ["Adam"]
list_of_learning_rates = [0.001]

for normalization_type in list_of_size_normalizations:
    for augmented_dir in list_of_augmented:

        if normalization_type == "average":
            normalization_type = "avg"

        data_loader = DataLoader()
        dict_of_style_train = data_loader.get_characters_style_based(
            "data\\processed_data\\style_classification\\normalized_{}\\{}".format(
                normalization_type, augmented_dir
            ),
            type_img="pgm",
        )

        dict_of_style_val = data_loader.get_characters_style_based(
            "data\\processed_data\\style_classification\\normalized_{}\\val".format(
                normalization_type
            ),
            type_img="pgm",
        )
        for batch_size in list_of_batch_sizes:
            for freeze_bool in list_of_freeze_layers:
                for is_mnist in list_of_mnist:
                    for optimizer_name in list_of_optimizers:
                        for learning_rate in list_of_learning_rates:

                            if is_mnist:
                                path_to_checkpoint = (
                                    "data\\MNIST\\MNIST\\checkpoints\\checkpoint_14.pth"
                                )
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
                                print(
                                    "Model {} already trained. Skipping...".format(
                                        model_folder
                                    )
                                )
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
