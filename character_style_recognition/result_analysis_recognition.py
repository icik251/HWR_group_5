import os
import torch

base_path = "data\\models\\character_recognition_final"
dict_of_results_checkpoints = dict()

for model_dir in os.listdir(base_path):
    path_of_model = os.path.join(base_path, model_dir)
    is_ea = False
    for check_file in os.listdir(path_of_model):
        if check_file.endswith(".pth"):
            is_check_int = False
            ea_int_optimal = check_file.split("_")[1].split(".pth")[0]
            try:
                _ = int(ea_int_optimal)
                is_check_int = True
            except Exception as e:
                pass

            if ea_int_optimal == "ea" or is_check_int:
                checkpoint = torch.load(
                    os.path.join(
                        path_of_model, "checkpoint_{}.pth".format(ea_int_optimal)
                    )
                )
                curr_results = checkpoint["results"]
                dict_of_results_checkpoints[model_dir] = curr_results
                is_ea = True

    if is_ea is False:
        checkpoint = torch.load(os.path.join(path_of_model, "checkpoint_optimal.pth"))
        curr_results = checkpoint["results"]
        dict_of_results_checkpoints[model_dir + "_" + "noEA"] = curr_results


import matplotlib.pyplot as plt
import numpy as np

for k, v in dict_of_results_checkpoints.items():
    epochs = range(len(v["test_acc"]))

    fig, (ax1, ax2) = plt.subplots(2)

    fig.suptitle(k)
    ax1.plot(epochs, v["test_loss"], label="Val")
    ax1.plot(epochs, v["train_loss"], label="Train")

    ax2.plot(epochs, v["test_acc"], label="Val")
    ax2.plot(epochs, v["train_acc"], label="Train")

    ax2.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax2.set_ylabel("Accuracy")

    # plt.subplot(epochs, v['test_loss'], label='Val')
    # plt.subplot(epochs, v['train_loss'], label='Train')
    # plt.legend()
    # plt.title(k)
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss')
    plt.legend()

    # plt.subplot(epochs, v['test_acc'], label='Val')
    # plt.subplot(epochs, v['train_acc'], label='Train')

    # plt.show()
    plt.savefig("data\\plots\\character_recognition_final\\{}.png".format(k))
    plt.cla()
