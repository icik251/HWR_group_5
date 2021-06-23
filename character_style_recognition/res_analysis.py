# Pipeline for different experiments
list_of_size_normalizations = ["average", "smallest"]
list_of_batch_sizes = [128, 256, 512]
list_of_freeze_layers = [True, False]
list_of_augmented = ["train_augmented", "train"]
list_of_mnist = [True, False]
list_of_optimizers = ["Adam", "SGD"]
list_of_learning_rates = [0.01, 0.001]

import os
import torch

base_path = "D:\\PythonProjects\\HWR_group_5\\data\\models\\character_recognition"
dict_of_results_checkpoints = dict()
for normalization_type in list_of_size_normalizations:
    for augmented_dir in list_of_augmented:
        for batch_size in list_of_batch_sizes:
            for freeze_bool in list_of_freeze_layers:
                for is_mnist in list_of_mnist:
                    for optimizer_name in list_of_optimizers:
                        for learning_rate in list_of_learning_rates:
                            normalization_type = "average"
                            if normalization_type == "average":
                                normalization_type = "avg"

                            # Path to load model and everything realated to it
                            model_folder = "norm_{}_batch_{}_augmented_{}_mnist_{}_freeze_{}_optim_{}_lr_{}".format(
                                normalization_type,
                                batch_size,
                                augmented_dir,
                                is_mnist,
                                freeze_bool,
                                optimizer_name,
                                learning_rate,
                            )

                            path_of_model = os.path.join(base_path, model_folder)

                            curr_best_checkpoint = 0
                            try:
                                for check_file in os.listdir(path_of_model):
                                    if check_file.endswith(".pth"):
                                        curr_checkppoint = int(
                                            check_file.split("_")[1].split(".")[0]
                                        )
                                    if curr_checkppoint > curr_best_checkpoint:
                                        curr_best_checkpoint = curr_checkppoint

                                    checkpoint = torch.load(
                                        os.path.join(
                                            path_of_model,
                                            "checkpoint_{}.pth".format(
                                                curr_best_checkpoint
                                            ),
                                        )
                                    )
                                    curr_results = checkpoint["results"]
                                    dict_of_results_checkpoints[
                                        model_folder
                                    ] = curr_results
                            except Exception as e:
                                pass


import matplotlib.pyplot as plt

for k, v in dict_of_results_checkpoints.items():
    epochs = range(len(v["test_acc"]))
    plt.plot(epochs, v["test_loss"], label="Test")
    plt.plot(epochs, v["train_loss"], label="Train")
    plt.legend()
    plt.title(k)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    # plt.show()
    plt.savefig(
        "D:\\PythonProjects\\HWR_group_5\\character_style_recognition\\plots\\{}.png".format(
            k
        )
    )
    plt.clf()
