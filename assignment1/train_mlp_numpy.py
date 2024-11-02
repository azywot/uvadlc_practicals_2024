################################################################################
# MIT License
#
# Copyright (c) 2024 University of Amsterdam
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course (UvA) | Fall 2024
# Date Created: 2024-10-28
################################################################################
"""
This module implements training and evaluation of a multi-layer perceptron in NumPy.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import, division, print_function

import argparse
import os
from copy import deepcopy
from datetime import datetime

import cifar10_utils
import matplotlib.pyplot as plt
import numpy as np
import torch
from mlp_numpy import MLP
from modules import CrossEntropyModule
from tqdm import tqdm


def accuracy(predictions, targets):
    """
    Computes the prediction accuracy, i.e. the average of correct predictions
    of the network.

    Args:
      predictions: 2D float array of size [batch_size, n_classes], predictions of the model (logits)
      labels: 1D int array of size [batch_size]. Ground truth labels for
              each sample in the batch
    Returns:
      accuracy: scalar float, the accuracy of predictions between 0 and 1,
                i.e. the average correct predictions over the whole batch

    TODO:
    Implement accuracy computation.
    """

    #######################
    # PUT YOUR CODE HERE  #
    #######################
    accuracy = (predictions.argmax(axis=1) == targets).mean()
    #######################
    # END OF YOUR CODE    #
    #######################

    return accuracy


def evaluate_model(model, data_loader):
    """
    Performs the evaluation of the MLP model on a given dataset.

    Args:
      model: An instance of 'MLP', the model to evaluate.
      data_loader: The data loader of the dataset to evaluate.
    Returns:
      avg_accuracy: scalar float, the average accuracy of the model on the dataset.

    TODO:
    Implement evaluation of the MLP model on a given dataset.

    Hint: make sure to return the average accuracy of the whole dataset,
          independent of batch sizes (not all batches might be the same size).
    """

    #######################
    # PUT YOUR CODE HERE  #
    #######################
    # average of accuracies (weighted by batch size)
    nominator, denominator = 0, 0

    for x, y in data_loader:
        batch_size = len(y)
        x_reshaped = x.reshape(x.shape[0], -1)

        predictions = model.forward(x_reshaped)
        nominator += accuracy(predictions, y) * batch_size
        denominator += batch_size

    avg_accuracy = nominator / denominator
    #######################
    # END OF YOUR CODE    #
    #######################

    return avg_accuracy


def train(hidden_dims, lr, batch_size, epochs, seed, data_dir):
    """
    Performs a full training cycle of MLP model.

    Args:
      hidden_dims: A list of ints, specificying the hidden dimensionalities to use in the MLP.
      lr: Learning rate of the SGD to apply.
      batch_size: Minibatch size for the data loaders.
      epochs: Number of training epochs to perform.
      seed: Seed to use for reproducible results.
      data_dir: Directory where to store/find the CIFAR10 dataset.
    Returns:
      model: An instance of 'MLP', the trained model that performed best on the validation set.
      val_accuracies: A list of scalar floats, containing the accuracies of the model on the
                      validation set per epoch (element 0 - performance after epoch 1)
      test_accuracy: scalar float, average accuracy on the test dataset of the model that
                     performed best on the validation. Between 0.0 and 1.0
      logging_dict: An arbitrary object containing logging information. This is for you to
                    decide what to put in here.

    TODO:
    - Implement the training of the MLP model.
    - Evaluate your model on the whole validation set each epoch.
    - After finishing training, evaluate your model that performed best on the validation set,
      on the whole test dataset.
    - Integrate _all_ input arguments of this function in your training. You are allowed to add
      additional input argument if you assign it a default value that represents the plain training
      (e.g. '..., new_param=False')

    Hint: you can save your best model by deepcopy-ing it.
    """

    # Set the random seeds for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)

    ## Loading the dataset
    cifar10 = cifar10_utils.get_cifar10(data_dir)
    cifar10_loader = cifar10_utils.get_dataloader(
        cifar10, batch_size=batch_size, return_numpy=True
    )

    #######################
    # PUT YOUR CODE HERE  #
    #######################

    # TODO: Initialize model and loss module
    model = MLP(3 * 32 * 32, hidden_dims, 10)
    loss_module = CrossEntropyModule()
    # TODO: Training loop including validation
    val_accuracies = []
    train_losses = []
    best_model = None
    best_val_accuracy = -1  # set to -1 to not have to check for first iteration

    with tqdm(range(epochs), desc="MLP numpy training") as p_bar:
        for epoch in p_bar:
            total_loss = 0
            num_batches = 0

            for x, y in cifar10_loader["train"]:

                x_reshaped = x.reshape(x.shape[0], -1)
                # print("x_reshaped.shape", x_reshaped.shape)

                # forward pass
                predictions = model.forward(x_reshaped)
                # print("predictions.shape", predictions.shape, "y.shape", y.shape)
                loss = loss_module.forward(predictions, y)
                # print("loss", loss)

                # backward pass
                dout = loss_module.backward(predictions, y)
                # print("dout.shape", dout.shape)
                model.backward(dout)

                # update weights
                for layer in model.layers:
                    if hasattr(layer, "params"):
                        for param_name, param in layer.params.items():
                            param -= lr * layer.grads[param_name]

                total_loss += loss
                num_batches += 1

            avg_epoch_loss = total_loss / num_batches
            train_losses.append(avg_epoch_loss)

            # evaluate on validation set
            val_accuracy = evaluate_model(model, cifar10_loader["validation"])
            val_accuracies.append(val_accuracy)
            p_bar.set_postfix(val_acc=val_accuracy, avg_loss=avg_epoch_loss)

            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_model = deepcopy(model)

    # TODO: Test best model
    test_accuracy = evaluate_model(best_model, cifar10_loader["test"])
    # TODO: Add any information you might want to save for plotting
    logging_dict = {
        "train_losses": train_losses,
        "val_accuracies": val_accuracies,
        "test_accuracy": test_accuracy,
        "best_val_accuracy": best_val_accuracy,
    }
    #######################
    # END OF YOUR CODE    #
    #######################

    return model, val_accuracies, test_accuracy, logging_dict


def plot_results(logging_dict: dict, epochs: int) -> None:
    """
    Plot and save the results of the training process.
    """

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    x = np.arange(1, epochs + 1)
    axs[0].plot(x, logging_dict["val_accuracies"])
    axs[0].set_xlabel("epoch")
    axs[0].set_ylabel("validation accuracy")
    axs[0].set_title("Validation accuracy over epochs")

    axs[1].plot(x, logging_dict["train_losses"])
    axs[1].set_xlabel("epoch")
    axs[1].set_ylabel("average loss from all batches")
    axs[1].set_title("Average loss over epochs")

    plt.tight_layout()
    date = datetime.now().strftime("%Y-%m-%d")
    filename = f"out/numpy_metrics_over_epochs_{date}.png"
    plt.savefig(filename)


if __name__ == "__main__":
    # Command line arguments
    parser = argparse.ArgumentParser()

    # Model hyperparameters
    parser.add_argument(
        "--hidden_dims",
        default=[128],
        type=int,
        nargs="+",
        help='Hidden dimensionalities to use inside the network. To specify multiple, use " " to separate them. Example: "256 128"',
    )

    # Optimizer hyperparameters
    parser.add_argument("--lr", default=0.1, type=float, help="Learning rate to use")
    parser.add_argument("--batch_size", default=128, type=int, help="Minibatch size")

    # Other hyperparameters
    parser.add_argument("--epochs", default=10, type=int, help="Max number of epochs")
    parser.add_argument(
        "--seed", default=42, type=int, help="Seed to use for reproducing results"
    )
    parser.add_argument(
        "--data_dir",
        default="data/",
        type=str,
        help="Data directory where to store/find the CIFAR10 dataset.",
    )

    args = parser.parse_args()
    kwargs = vars(args)

    model, val_accuracies, test_accuracy, logging_dict = train(**kwargs)
    print(logging_dict)
    # Feel free to add any additional functions, such as plotting of the loss curve here
    plot_results(logging_dict, args.epochs)
