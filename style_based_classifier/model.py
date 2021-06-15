from early_stopping import EarlyStopping
import torch.nn as nn
import torch.nn.functional as F
import torch
from datetime import date, datetime
import os
from utils import boolean_to_255, multi_acc, set_parameter_requires_grad, save_image
# from mnist_model import MNISTResNet


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
    def __init__(self, model_path_to_load, freeze_layers=True, seed=42) -> None:
        self.seed = seed
        self.model_path_to_load = model_path_to_load
        self.idx2style = {0: "Archaic", 1: "Hasmonean", 2: "Herodian"}
        self.checkpoint = None
        self.model = None

        self.load_model()
        if freeze_layers:
            self.model = set_parameter_requires_grad(self.model, freeze=freeze_layers)
        self.modify_output_layer()

        ## Move model to cuda if available
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def load_model(self):
        self.model = MNISTResNet()
        self.checkpoint = torch.load(self.model_path_to_load)
        self.model.load_state_dict(self.checkpoint["state_dict"])

    def modify_output_layer(self):
        self.model.fc = nn.Sequential(nn.Linear(512, 3))

    def train(
        self,
        epochs,
        optimizer,
        criterion,
        train_loader,
        test_loader,
        patience,
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
            patience=patience, verbose=True, path=path_checkpoints
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
                        path_checkpoints, "checkpoint_{}".format(epoch)
                    )
                    self.save_test_images(test_loader, criterion, path_to_save)

                if early_stopping.early_stop:
                    print("Early stopping")
                    break

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
