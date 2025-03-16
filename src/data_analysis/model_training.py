"""Helper functions and classes for training the fashionMNIST model from the fashion_mnist_model.py file."""
import os
import json
from pathlib import Path
from typing import Any
from datetime import datetime

import numpy as np
from sklearn.model_selection import train_test_split

import torch
from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset, Dataset
import torchvision

from src.data_analysis.mnist_constants import (
    FASHION_MNIST_INPUT_DIM,
    FASHION_MNIST_OUTPUT_DIM,
    FASHION_MNIST_MEAN,
    FASHION_MNIST_STD,
    MNIST_DIGITS_MEAN,
    MNIST_DIGITS_STD,
)
from src.data_analysis.fully_connected_network import ACAIGFCN
from src.data_analysis.custom_cnn import MyFirstCNN

DATA_DIR = '../data/'


def class_prediction_acc(y_hat, y_truth):
    return (torch.argmax(y_hat, dim=1) == y_truth).type(torch.FloatTensor).mean()


def get_device_helper():
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.backends.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    return device


def get_fashion_mnist_data_sets() -> tuple[Dataset, Dataset]:
    """
    Load the FashionMNIST dataset from torchvision.datasets.FashionMNIST and apply the necessary transformations.

    :return: tuple[Dataset, Dataset]
    """
    transform_for_data = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((FASHION_MNIST_MEAN,), (FASHION_MNIST_STD,))
        ]
    )

    train_dataset = torchvision.datasets.FashionMNIST(
        DATA_DIR,
        train=True,
        download=True,
        transform=transform_for_data,
    )

    test_dataset = torchvision.datasets.FashionMNIST(
        DATA_DIR,
        train=False,
        download=True,
        transform=transform_for_data,
    )

    return train_dataset, test_dataset


def get_mnist_digits_data_sets():
    """
    Load the MNIST digits dataset from torchvision.datasets.MNIST and apply the necessary transformations.
    """
    transform_for_data = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((MNIST_DIGITS_MEAN,), (MNIST_DIGITS_STD,))
        ]
    )

    train_dataset = torchvision.datasets.MNIST(
        DATA_DIR,
        train=True,
        download=True,
        transform=transform_for_data,
    )

    test_dataset = torchvision.datasets.MNIST(
        DATA_DIR,
        train=False,
        download=True,
        transform=transform_for_data,
    )

    return train_dataset, test_dataset


def get_data_sets(dataset_name: str = "fashion") -> tuple[Dataset, Dataset]:
    """
    Logically determine which dataset to load.

    :param dataset_name: The name of the dataset to load. Either 'fashion' or 'digits'.

    :return: tuple[Dataset, Dataset]
    """
    if dataset_name.lower() == "fashion":
        return get_fashion_mnist_data_sets()
    elif dataset_name.lower() == "digits":
        return get_mnist_digits_data_sets()
    else:
        raise ValueError(f"Invalid dataset name: {dataset_name} expected one of ['fashion', 'digits']")


def get_data_loaders(
    train_dataset: Dataset,
    test_dataset: Dataset,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Prepare the training, validation and testing data loaders for the FashionMNIST dataset.

    :param train_dataset: The training dataset
    :param test_dataset: The testing dataset
    
    :return: tuple[DataLoader, DataLoader, DataLoader]
    """
    split_size = 0.1

    train_indices, val_indices, _, _ = train_test_split(
        range(len(train_dataset)),
        train_dataset.targets,
        stratify=train_dataset.targets,
        test_size=split_size,
    )

    # Generate training and validation subsets based on indices
    train_split = Subset(train_dataset, train_indices)
    val_split = Subset(train_dataset, val_indices)

    # set batches sizes
    train_batch_size = 512 #Define train batch size
    test_batch_size  = 256 #Define test batch size (can be larger than train batch size)

    # Define dataloader objects that help to iterate over batches and samples for
    # training, validation and testing
    train_batches = DataLoader(train_split, batch_size=train_batch_size, shuffle=True)
    val_batches = DataLoader(val_split, batch_size=train_batch_size, shuffle=True)
    test_batches = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=True)
                                            
    return train_batches, val_batches, test_batches


class ModelTrainingArtifactBase():
    def __init__(
        self,
        model: nn.Module,
        num_epochs: int,
        optimizer: torch.optim.Optimizer,
        optimizer_hyperparams: dict[str, Any],
        objective: nn.Module,
        accuracy_metric: callable = class_prediction_acc,
        dataset_name: str = "fashion",
        experiments_directory: str = "experiments",
    ) -> None:
        """
        Initialize the ModelTrainingArtifact object with the necessary params.

        :param model: The model to train
        :param num_epochs: The number of epochs to train the model
        :param optimizer: The optimizer to use for training the model
        :param optimizer_hyperparams: The hyperparameters to use for the optimizer  
        :param objective: The loss function to use for training the model
        :param accuracy_metric: The accuracy metric to use for evaluating the model
        :param dataset_name: The name of the dataset to use for training the model
        :param experiments_directory: The directory to save the experiments

        :return: ModelTrainingArtifact
        """
        self.experiments_directory = experiments_directory

        self.model = model

        self.num_epochs = num_epochs
        self.optimizer_hyperparams = optimizer_hyperparams
        self.optimizer = optimizer(self.model.parameters(), **self.optimizer_hyperparams)
        
        self.objective = objective
        self.accuracy_metric = accuracy_metric
        self.device = get_device_helper()

        train_dataset, test_dataset = get_data_sets(dataset_name=dataset_name)
        train_batches, val_batches, test_batches = get_data_loaders(train_dataset=train_dataset, test_dataset=test_dataset)
        self.train_batches = train_batches
        self.val_batches = val_batches
        self.test_batches = test_batches

    def train(self):
        """
        Train the given model using the training and validation data loaders.

        :return: None
        """
        self.train_loss = []
        self.validation_loss = []
        self.train_accuracy = []
        self.validation_accuracy = []

        loop = tqdm(total=len(self.train_batches) * self.num_epochs, position=0)

        # Iterate over epochs, batches with progress bar and train+ validate the ACAIGFCN
        # Track the loss and validation accuracy
        train_start_time = datetime.now()
        for epoch in range(self.num_epochs):

            # ACAIGFCN Training 
            for train_features, train_labels in self.train_batches:
                # Set model into training mode
                self.model = self.model.to(self.device)
                self.model.train()
                
                # Reshape images into a vector
                train_features = train_features.reshape(-1, 28*28)

                train_features = train_features.to(self.device)
                train_labels = train_labels.to(self.device)

                self.optimizer.zero_grad()

                y_hat = self.model(train_features)
                loss = self.objective(y_hat, train_labels)

                loss.backward()
                self.optimizer.step()
                
                # loop.set_description(f"Epoch {epoch+1}/{epochs}")
                # loop.update(1)

                # Reset gradients, Calculate training loss on model 
                # Perfrom optimization, back propagation
                train_accuracy = self.accuracy_metric(y_hat, train_labels)
                self.train_accuracy.append(train_accuracy)
        
                # Record loss for the epoch
                self.train_loss.append(loss.item())

                loop.set_description('epoch:{} train loss:{:.4f} train accuracy:{:.4f}'.format(epoch, self.train_loss[-1], train_accuracy))
                loop.update(1)

            current_val_acc = []
            current_val_loss = []
            
            # ACAIGFCN Validation
            for val_features, val_labels in self.val_batches:
                
                # Telling PyTorch we aren't passing inputs to network for training purpose
                with torch.no_grad(): 
                    self.model.eval()
                    
                    # Reshape validation images into a vector
                    val_features = val_features.reshape(-1, 28*28)

                    val_features = val_features.to(self.device)
                    val_labels = val_labels.to(self.device)

                    y_hat = self.model(val_features)
                    val_loss = self.objective(y_hat, val_labels)
                
                    # Compute validation outputs (targets) 
                    # and compute accuracy
                    val_accuracy = self.accuracy_metric(y_hat, val_labels)
                    current_val_acc.append(val_accuracy)
                    current_val_loss.append(val_loss.item())

            self.validation_accuracy.append((len(self.train_accuracy), np.mean(current_val_acc)))
            self.validation_loss.append((len(self.train_loss), np.mean(current_val_loss)))

        self.training_time = datetime.now() - train_start_time
        loop.close()

    def test(self):
        """
        Test the ACAIGFCN model using the test data loader.
        
        :param record_experiment: Whether to record the experiment in the experiments.json file

        :return: None
        """
        test_acc = []
        # Telling PyTorch we aren't passing inputs to network for training purpose
        with torch.no_grad():
            
            for test_features, test_labels in self.test_batches:

                test_features = test_features.to(self.device)
                test_labels = test_labels.to(self.device)

                self.model.eval()
                # Reshape test images into a vector
                test_features = test_features.reshape(-1, 28*28)

                # Compute validation outputs (targets) 
                y_hat = self.model(test_features)
                test_acc.append(self.accuracy_metric(y_hat, test_labels))
                
        self.mean_test_accuracy = np.mean(test_acc)
        self.std_test_accuracy = np.std(test_acc)
        print(f"Mean Accuracy Across Each Batch of the test set: {self.mean_test_accuracy:.4f} ± {self.std_test_accuracy:.5f}")

    def record_experiment(self) -> None:
        pass

    def save_experiment_model(self) -> None:
        pass

    def runner(self, record_experiment: bool = True, record_experiment_kwargs: dict[str, Any] = None) -> None:
        """
        Run the entire training, testing, recording and saving process for the ACAIGFCN model.

        :return: None
        """
        self.train()
        self.test()
        if record_experiment:
            if record_experiment_kwargs:
                self.record_experiment(**record_experiment_kwargs)
            else:
                self.record_experiment()
            self.save_experiment_model()


class FCNArtifact(ModelTrainingArtifactBase):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        objective: nn.Module,
        num_epochs: int,
        hidden_layer_dims: list[int],
        dropout_rate_layers: list[float],
        weight_init_method: str,
        batch_norm: bool,
        activation_function: callable,
        optimizer_hyperparams: dict[str, Any],
        accuracy_metric: callable = class_prediction_acc,
        dataset_name: str = "fashion",
        experiments_directory: str = "experiments",
    ):
        """
        Initialize the ModelTrainingArtifact object with the necessary hyperparameters and model architecture.

        :param optimizer: The optimizer to use for training the model
        :param objective: The loss function to use for training the model
        :param num_epochs: The number of epochs to train the model
        :param hidden_layer_dims: The dimensions of the hidden layers in the model
        :param dropout_rate_layers: The dropout rates for each layer in the model
        :param weight_init_method: The weight initialization method to use for the model
        :param batch_norm: Whether to use batch normalization in the model
        :param activation_function: The activation function to use in the model
        :param optimizer_hyperparams: The hyperparameters to use for the optimizer
        :param accuracy_metric: The accuracy metric to use for evaluating the model
        :param dataset_name: The name of the dataset to use for training the model
        :param experiments_directory: The directory to save the experiments

        :return: ModelTrainingArtifact
        """
        model = ACAIGFCN(
            input_dim=FASHION_MNIST_INPUT_DIM,
            output_dim=FASHION_MNIST_OUTPUT_DIM,
            hidden_layer_dims=hidden_layer_dims,
            activation_function=activation_function,
            dropout=dropout_rate_layers,
            batch_norm=batch_norm,
        )
        self.weight_init_method = weight_init_method

        super().__init__(
            model=model,
            num_epochs=num_epochs,
            optimizer=optimizer,
            optimizer_hyperparams=optimizer_hyperparams,
            objective=objective,
            accuracy_metric=accuracy_metric,
            dataset_name=dataset_name,
            experiments_directory=experiments_directory,
        )
        # Initialize weights
        self.model.apply(self.init_weights)

    def init_weights(self, layer: nn.Linear):
        """
        initialize the weights of the model using the specified weight initialization method

        :param layer: The layer to initialize the weights for

        :return: None
        """
        if isinstance(layer, nn.Linear):
            if self.weight_init_method == "xavier_uniform":
                nn.init.xavier_uniform_(layer.weight)
            elif self.weight_init_method == "random_normal":
                nn.init.normal_(layer.weight)
            elif self.weight_init_method == "kaiming_uniform":
                nn.init.kaiming_uniform_(layer.weight)
            else:
                raise ValueError(f"Invalid weight initialization method: {self.weight_init_method} expected one of ['xavier_uniform', 'random_normal', 'kaiming_uniform']")
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)

    def record_experiment(self, to_compare_with_cnn: bool = False, filename: str = 'experiments.json'):
        """
        Record the critical information about the model in the experiments.json file

        Specifically, record the following:
        - The model architecture
        - The optimizer
        - The number of epochs
        - The dropout rate
        - The activation function
        - The optimizer hyperparameters
        - Test Accuracy (mean and standard deviation)
        - In the future, record the batch normalization setup and weight initialization information

        
        :param path: The path to the experiments.json file
        :param to_compare_with_cnn: Whether to compare the model with the CNN model
        :param filename: The name of the experiments.json file

        :return: None
        """
        if not os.path.exists(self.experiments_directory):
                os.makedirs(self.experiments_directory)
        
        self.experiment_filename = filename

        extra_opt_params = self.optimizer_hyperparams.copy()
        extra_opt_params.pop('lr', None)

        if to_compare_with_cnn:
            experiment_info = {
                "model_type": "CNN",
                "num_epochs": self.num_epochs,
                # TODO Verify this is how we want to calculate weights
                "total_num_weights": np.sum(p.numel() for p in self.model.parameters()),
                "training_time": str(self.training_time).split(".")[0],
                "optimizer": self.optimizer.__class__.__name__,
                "learning_rate": self.optimizer.param_groups[0]['lr'],
                "optimizer_hyperparams": extra_opt_params,
                "weight_init": self.weight_init_method,
                "mean_test_accuracy": self.mean_test_accuracy.astype(float),
                "std_test_accuracy": self.std_test_accuracy.astype(float),
            }
        else:
            experiment_info = {
                "num_epochs": self.num_epochs,
                "mean_test_accuracy": self.mean_test_accuracy.astype(float),
                "std_test_accuracy": self.std_test_accuracy.astype(float),
                "model_architecture": [layer.out_features for layer in self.model.layers],
                "dropout_rate": self.model.dropout_rates,
                "activation_function": self.model.activation_function.__name__,
                "optimizer": self.optimizer.__class__.__name__,
                "learning_rate": self.optimizer.param_groups[0]['lr'],
                "optimizer_hyperparams": extra_opt_params,
                "batch_normalization": self.model.do_batch_norm,
                "weight_init": self.weight_init_method,
            }
        try:
            json.dumps(experiment_info)
        except (TypeError, OverflowError) as e:
            raise ValueError("Failed to record experiment.") from e
        
        filepath = os.path.join(self.experiments_directory, self.experiment_filename)
        try:
            with open(filepath, 'r') as file:
                records = json.load(file)
        except FileNotFoundError:
            records = []
        records.append(experiment_info)
        with open(filepath, 'w') as f:
            json.dump(records, f, indent=4)

    def save_experiment_model(self):
        """
        Saves the PyTorch model, training and validation losses and accuracies, 
        test accuracy and standard deviation to a .pth file with a name that uniquely identifies the experiment.

        :param filepath: The path to save the experiment model

        :return: None
        """
        experiment_data = {
            'model_state_dict': self.model.state_dict(),
            'train_loss': self.train_loss,
            'validation_loss': self.validation_loss,
            'train_accuracy': self.train_accuracy,
            'validation_accuracy': self.validation_accuracy,
            'mean_test_accuracy': self.mean_test_accuracy,
            'std_test_accuracy': self.std_test_accuracy,
        }

        try:
            filepath = os.path.join(self.experiments_directory, self.experiment_filename)
        except AttributeError:
            raise ValueError("Experiment filename not set. Please record the experiment first.")

        try:
            with open(filepath, 'r') as file:
                records = json.load(file)
        except FileNotFoundError:
            records = []

        experiment_num = len(records)
        
        experiment_name = f"experiment_number_{experiment_num:04d}.pth"
        
        model_filepath = os.path.join(self.experiments_directory, "models")
        if not os.path.exists(model_filepath):
            os.makedirs(model_filepath)

        # Save the experiment data
        torch.save(experiment_data, os.path.join(model_filepath, experiment_name))


class CNNArtifact(ModelTrainingArtifactBase):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        objective: nn.Module,
        num_epochs: int,
        weight_init_method: str,
        batch_norm: bool,
        optimizer_hyperparams: dict[str, Any],
        accuracy_metric: callable = class_prediction_acc,
        dataset_name: str = "fashion",
        experiments_directory: str = "experiments",
    ) -> None:
        """
        Initialize the ModelTrainingArtifact object with the necessary hyperparameters and model architecture.

        :param optimizer: The optimizer to use for training the model
        :param objective: The loss function to use for training the model
        :param num_epochs: The number of epochs to train the model
        :param weight_init_method: The weight initialization method to use for the model
        :param batch_norm: Whether to use batch normalization in the model
        :param optimizer_hyperparams: The hyperparameters to use for the optimizer
        :param accuracy_metric: The accuracy metric to use for evaluating the model
        :param dataset_name: The name of the dataset to use for training the model

        :return: ModelTrainingArtifact
        """
        # Fill out this section...
        model = MyFirstCNN()
        self.weight_init_method = weight_init_method

        super().__init__(
            model=model,
            num_epochs=num_epochs,
            optimizer=optimizer,
            optimizer_hyperparams=optimizer_hyperparams,
            objective=objective,
            accuracy_metric=accuracy_metric,
            dataset_name=dataset_name,
            experiments_directory=experiments_directory,
        )
        # Initialize weights
        self.model.apply(self.init_weights)

    # TODO: Do we need this function for the cnn?
    def init_weights(self, layer: nn.Linear):
        """
        initialize the weights of the model using the specified weight initialization method

        :param layer: The layer to initialize the weights for

        :return: None
        """
        if isinstance(layer, nn.Linear):
            if self.weight_init_method == "xavier_uniform":
                nn.init.xavier_uniform_(layer.weight)
            elif self.weight_init_method == "random_normal":
                nn.init.normal_(layer.weight)
            elif self.weight_init_method == "kaiming_uniform":
                nn.init.kaiming_uniform_(layer.weight)
            else:
                raise ValueError(f"Invalid weight initialization method: {self.weight_init_method} expected one of ['xavier_uniform', 'random_normal', 'kaiming_uniform']")
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)

    def record_experiment(self, filename: str = 'experiments.json'):
        """
        Record the critical information about the model in the experiments.json file

        Specifically, record the following:
        - The model architecture
        - The optimizer
        - The number of epochs
        - The dropout rate
        - The activation function
        - The optimizer hyperparameters
        - Test Accuracy (mean and standard deviation)
        - In the future, record the batch normalization setup and weight initialization information

        
        :param path: The path to the experiments.json file
        :param filename: The name of the experiments.json file

        :return: None
        """
        self.experiment_filename = filename

        extra_opt_params = self.optimizer_hyperparams.copy()
        extra_opt_params.pop('lr', None)

        experiment_info = {
            "model_type": "CNN",
            "num_epochs": self.num_epochs,
            # TODO Verify this is how we want to calculate weights
            "total_num_weights": np.sum(p.numel() for p in self.model.parameters()),
            "training_time": self.training_time,
            "optimizer": self.optimizer.__class__.__name__,
            "learning_rate": self.optimizer.param_groups[0]['lr'],
            "optimizer_hyperparams": extra_opt_params,
            "weight_init": self.weight_init_method,
            "mean_test_accuracy": self.mean_test_accuracy.astype(float),
            "std_test_accuracy": self.std_test_accuracy.astype(float),
        }
        try:
            json.dumps(experiment_info)
        except (TypeError, OverflowError) as e:
            raise ValueError("Failed to record experiment.") from e
        
        filepath = os.path.join(self.experiments_directory, self.experiment_filename)
        try:
            with open(filepath, 'r') as file:
                records = json.load(file)
        except FileNotFoundError:
            records = []
        records.append(experiment_info)
        with open(filepath, 'w') as f:
            json.dump(records, f, indent=4)

    def save_experiment_model(self):
        """
        Saves the PyTorch model, training and validation losses and accuracies, 
        test accuracy and standard deviation to a .pth file with a name that uniquely identifies the experiment.

        :param filepath: The path to save the experiment model

        :return: None
        """
        experiment_data = {
            'model_state_dict': self.model.state_dict(),
            'train_loss': self.train_loss,
            'validation_loss': self.validation_loss,
            'train_accuracy': self.train_accuracy,
            'validation_accuracy': self.validation_accuracy,
            'mean_test_accuracy': self.mean_test_accuracy,
            'std_test_accuracy': self.std_test_accuracy,
        }

        try:
            filepath = os.path.join(self.experiments_directory, self.experiment_filename)
        except AttributeError:
            raise ValueError("Experiment filename not set. Please record the experiment first.")

        try:
            with open(filepath, 'r') as file:
                records = json.load(file)
        except FileNotFoundError:
            records = []

        experiment_num = len(records)
        
        experiment_name = f"experiment_number_{experiment_num:04d}.pth"
        
        model_filepath = os.path.join(self.experiments_directory, "models")
        if not os.path.exists(model_filepath):
            os.makedirs(model_filepath)

        # Save the experiment data
        torch.save(experiment_data, os.path.join(model_filepath, experiment_name))
