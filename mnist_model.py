from early_stopping import EarlyStopping

from datetime import datetime
import os
from torch import nn
from torchvision.datasets import MNIST
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.models.resnet import ResNet, BasicBlock, Bottleneck
import torch.optim as optim
from utils import multi_acc


class MNISTResNet(ResNet):
    def __init__(self, num_classes=10):
        super(MNISTResNet, self).__init__(
            BasicBlock, [2, 2, 2, 2], num_classes=num_classes
        )  # Based on ResNet18
        # super(MNISTResNet, self).__init__(BasicBlock, [3, 4, 6, 3], num_classes=10) # Based on ResNet34
        # super(MNISTResNet, self).__init__(Bottleneck, [3, 4, 6, 3], num_classes=10) # Based on ResNet50
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=1, padding=3, bias=False)


class MNISTModel:
    def __init__(self, root_path, seed) -> None:
        self.root_path = root_path
        self.seed = seed
        self.train_loader = None
        self.test_loader = None
        self.model = MNISTResNet()

        ## Move model to cuda if available
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def prepate_data(self):
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5), (0.5)),
            ]
        )

        train_set = torchvision.datasets.MNIST(
            root=self.root_path, train=True, download=True, transform=transform
        )
        self.train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=500, shuffle=True, num_workers=2
        )

        test_set = torchvision.datasets.MNIST(
            root=self.root_path, train=False, download=True, transform=transform
        )
        self.test_loader = torch.utils.data.DataLoader(
            test_set, batch_size=500, shuffle=False, num_workers=2
        )

    def train(self, epochs, optimizer, criterion, patience, path_checkpoints):

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
                for X_train_batch, y_train_batch in self.train_loader:
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
                    for X_test_batch, y_test_batch in self.test_loader:
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
                    round(train_epoch_acc / len(self.train_loader), 3)
                )
                dict_of_results["train_loss"].append(
                    round(train_epoch_loss / len(self.train_loader), 3)
                )
                dict_of_results["test_acc"].append(
                    round(test_epoch_acc / len(self.test_loader), 3)
                )
                dict_of_results["test_loss"].append(
                    round(test_epoch_loss / len(self.test_loader), 3)
                )
                dict_of_results["convergence_time"].append(
                    epoch_end_time - epoch_start_time
                )

                if epoch % 1 == 0:
                    print(
                        "Epoch: {} | Train Loss: {} |  Test Loss: {} | Train acc: {} | Test acc: {} | Time taken: {}".format(
                            epoch,
                            round(train_epoch_loss / len(self.train_loader), 3),
                            round(test_epoch_loss / len(self.test_loader), 3),
                            round(train_epoch_acc / len(self.train_loader), 3),
                            round(test_epoch_acc / len(self.test_loader), 3),
                            epoch_end_time - epoch_start_time,
                        )
                    )

                early_stopping(
                    test_epoch_loss, self.model, epoch, optimizer, dict_of_results
                )
                if early_stopping.early_stop:
                    print("Early stopping")
                    break

    def get_model_params(self):
        return self.model.parameters()
