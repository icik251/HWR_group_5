from torch._C import device
from early_stopping import EarlyStopping
import torch.nn as nn
import torch.nn.functional as F
import torch
from datetime import date, datetime
import os
from utils import multi_acc, set_parameter_requires_grad, save_image
from mnist_model import MNISTResNet
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd


class ConvNet(nn.Module):
    def __init__(self, dropout_rate, activation_function_type):
        super(ConvNet, self).__init__()

        if activation_function_type == "relu":
            activation_function = nn.ReLU()
        elif activation_function_type == "sigmoid":
            activation_function = nn.Sigmoid()
        elif activation_function_type == "leaky_relu":
            activation_function = nn.LeakyReLU()

        self.layer1 = nn.Sequential(
            # 3 is the rbg
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            activation_function,
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            activation_function,
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(128),
            activation_function,
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.drop_out = nn.Dropout(p=dropout_rate)
        self.fc1 = nn.Linear(4480, 1000)
        self.fc2 = nn.Linear(1000, 3)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return torch.log_softmax(out, dim=1)


class Model:
    def __init__(
        self, mode="recognition", model_path_to_load=None, freeze_layers=True, seed=42
    ) -> None:
        self.mode = mode
        self.seed = seed
        self.model_path_to_load = model_path_to_load
        self.style2idx = {"Archaic": 0, "Hasmonean": 1, "Herodian": 2}
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
        self.checkpoint = None
        self.model = None

        self.load_model()
        if freeze_layers:
            self.model = set_parameter_requires_grad(self.model, freeze=freeze_layers)

        if mode == "recognition":
            self.modify_output_layer(num_classes=27)
        elif mode == "style":
            self.modify_output_layer(num_classes=3)

        ## Move model to cuda if available
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def load_model(self):
        self.model = MNISTResNet()
        if self.model_path_to_load is not None:
            self.checkpoint = torch.load(self.model_path_to_load)
            self.model.load_state_dict(self.checkpoint["state_dict"])

    def modify_output_layer(self, num_classes):
        self.model.fc = nn.Sequential(nn.Linear(512, num_classes))

    def save_confusion_matrix(self, list_of_labels, list_of_predictions, path_to_save):
        if self.mode == "recognition":
            list_of_classes = self.char2idx.keys()
        elif self.mode == "style":
            list_of_classes = self.style2idx.keys()

        # Confusion matrix
        list_of_labels = list_of_labels.cpu().detach().numpy()
        list_of_predictions = list_of_predictions.cpu().detach().numpy()
        conf_mat = confusion_matrix(list_of_labels, list_of_predictions)
        df_cm = pd.DataFrame(
            conf_mat,
            index=[item for item in list_of_classes],
            columns=[item for item in list_of_classes],
        )
        # plt.figure(figsize = (10,7))
        cm_plot = sn.heatmap(df_cm, annot=True)
        cm_plot.figure.savefig(path_to_save)

    def train(
        self,
        epochs,
        optimizer,
        criterion,
        train_loader,
        test_loader,
        patience,
        delta,
        path_checkpoints,
    ):

        torch.manual_seed(self.seed)
        ## CHANGE THIS: dict should be loaded and not created as new if model is loaded to be trained
        dict_of_results = {
            "train_acc": list(),
            "train_loss": list(),
            "test_acc": list(),
            "test_loss": list(),
            "convergence_time": list(),
        }

        # initialize the early_stopping object
        early_stopping = EarlyStopping(
            patience=patience, verbose=True, delta=delta, path=path_checkpoints
        )

        self.model.to(self.device)
        with torch.cuda.device(self.device.index):
            for epoch in range(1, epochs + 1):
                epoch_start_time = datetime.now()
                # TRAINING
                train_epoch_loss = 0
                train_epoch_acc = 0
                self.model.train()
                for X_train_batch, y_train_batch in train_loader:
                    X_train_batch, y_train_batch = (
                        X_train_batch.to(self.device),
                        y_train_batch.to(self.device),
                    )
                    optimizer.zero_grad()

                    y_train_pred = self.model(X_train_batch)

                    train_loss = criterion(y_train_pred, y_train_batch)
                    train_acc = multi_acc(y_train_pred, y_train_batch)

                    train_loss.backward()
                    optimizer.step()

                    train_epoch_loss += train_loss.item()
                    train_epoch_acc += train_acc.item()

                # VALIDATION
                with torch.no_grad():

                    test_epoch_loss = 0
                    test_epoch_acc = 0

                    # Initialize the prediction and label lists(tensors)
                    pred_list = torch.zeros(0, dtype=torch.long, device=self.device)
                    label_list = torch.zeros(0, dtype=torch.long, device=self.device)

                    self.model.eval()
                    for X_test_batch, y_test_batch in test_loader:
                        X_test_batch, y_test_batch = (
                            X_test_batch.to(self.device),
                            y_test_batch.to(self.device),
                        )

                        y_test_pred = self.model(X_test_batch)

                        test_loss = criterion(y_test_pred, y_test_batch)
                        test_acc = multi_acc(y_test_pred, y_test_batch)

                        test_epoch_loss += test_loss.item()
                        test_epoch_acc += test_acc.item()

                        # For creating a confusion matrix
                        y_pred_softmax = torch.log_softmax(y_test_pred, dim=1)
                        _, y_pred_tags = torch.max(y_pred_softmax, dim=1)
                        # Append batch prediction results
                        pred_list = torch.cat([pred_list, y_pred_tags.view(-1)])
                        label_list = torch.cat([label_list, y_test_batch.view(-1)])

                epoch_end_time = datetime.now()

                dict_of_results["train_acc"].append(
                    round(train_epoch_acc / len(train_loader), 3)
                )
                dict_of_results["train_loss"].append(
                    round(train_epoch_loss / len(train_loader), 3)
                )
                dict_of_results["test_acc"].append(
                    round(test_epoch_acc / len(test_loader), 3)
                )
                dict_of_results["test_loss"].append(
                    round(test_epoch_loss / len(test_loader), 3)
                )
                dict_of_results["convergence_time"].append(
                    epoch_end_time - epoch_start_time
                )

                if epoch % 1 == 0:
                    print(
                        "Epoch: {} | Train Loss: {} |  Test Loss: {} | Train acc: {} | Test acc: {} | Time taken: {}".format(
                            epoch,
                            round(train_epoch_loss / len(train_loader), 3),
                            round(test_epoch_loss / len(test_loader), 3),
                            round(train_epoch_acc / len(train_loader), 3),
                            round(test_epoch_acc / len(test_loader), 3),
                            epoch_end_time - epoch_start_time,
                        )
                    )

                is_saved = early_stopping(
                    test_epoch_loss, self.model, epoch, optimizer, dict_of_results
                )
                if is_saved:
                    path_to_save = os.path.join(
                        path_checkpoints, "cm_plot_{}.png".format(epoch)
                    )
                    # self.save_test_images(test_loader, criterion, path_to_save)
                    self.save_confusion_matrix(label_list, pred_list, path_to_save)

                if early_stopping.early_stop:
                    print("Early stopping")
                    break

    """
    def save_test_images(self, test_loader, criterion, path_to_save):
        # VALIDATION
        with torch.no_grad():
            test_epoch_loss = 0
            test_epoch_acc = 0

            self.model.eval()
            count = 0
            for X_test_batch, y_test_batch in test_loader:
                X_test_batch, y_test_batch = (
                    X_test_batch.to(self.device),
                    y_test_batch.to(self.device),
                )

                y_test_pred = self.model(X_test_batch)

                test_loss = criterion(y_test_pred, y_test_batch)
                test_acc = multi_acc(y_test_pred, y_test_batch)
                test_epoch_loss += test_loss.item()
                test_epoch_acc += test_acc.item()

                y_pred_softmax = torch.log_softmax(y_test_pred, dim=1)
                _, y_pred_tag = torch.max(y_pred_softmax, dim=1)

                y_pred_tag = y_pred_tag.detach().cpu().item()
                y_test_batch = y_test_batch.detach().cpu().item()

                X_test_batch = torch.squeeze(X_test_batch)
                sample_numpy = X_test_batch.detach().cpu().numpy()
                sample_numpy = boolean_to_255(sample_numpy)

                image_name = "{}_Pred_{}_Real_{}".format(
                    count, self.idx2style[y_pred_tag], self.idx2style[y_test_batch]
                )

                save_image(sample_numpy, image_name, path_to_save)
                count += 1
    """

    def get_model_params(self):
        return self.model.parameters()

    def save_model_results(epoch, model, optimizer, dict_of_results, file_name):
        state = {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "results": dict_of_results,
        }

        # Check if folder structure is created, if not - create it
        if not os.path.isdir("Results"):
            os.makedirs("Results")
        torch.save(state, os.path.join("Results", file_name + ".pth"))
