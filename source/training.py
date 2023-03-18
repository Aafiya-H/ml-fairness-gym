import logging
import os

import numpy as np
import torch
from tqdm import tqdm

from source.datasets import TabularDataset


def train_classification_model(
    model,
    training_features,
    training_targets,
    loss_function,
    learning_rate,
    num_epochs,
    batch_size,
    save_path
):
    save_dir = "/".join(save_path.split('/')[0])
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Training with: {}".format(device))

    model.train()
    model.to(device)

    training_dataset = TabularDataset(
        features=training_features, targets=training_targets, device=device
    )
    training_loader = torch.utils.data.DataLoader(
        training_dataset, batch_size=batch_size, shuffle=True
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    losses = []
    for epoch in range(num_epochs):
        with tqdm(training_loader) as batches:
            batches.set_description("Training - Epoch {}".format(epoch + 1))
            for features, targets in batches:
                optimizer.zero_grad()
                predictions = model(features)
                loss = loss_function(predictions, targets.reshape(-1, 1))
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
                batches.set_postfix({"loss": loss.item()})
    torch.save(model, save_path)
    return losses


def train_reward_model(
    model,
    training_features,
    training_targets,
    loss_function,
    learning_rate,
    num_epochs,
    batch_size,
    save_path
):
    save_dir = "/".join(save_path.split('/')[0])
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Training with: {}".format(device))

    model.train()
    model.to(device)

    training_dataset = TabularDataset(
        features=training_features, targets=training_targets, device=device
    )
    training_loader = torch.utils.data.DataLoader(
        training_dataset, batch_size=batch_size, shuffle=True
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    losses = []
    for epoch in range(num_epochs):
        with tqdm(training_loader) as batches:
            batches.set_description("Training - Epoch {}".format(epoch + 1))
            for features, decisions in batches:
                optimizer.zero_grad()
                features_ones = torch.cat(
                    [
                        features,
                        torch.ones(
                            size=(features.shape[0], 1),
                            device=device,
                        ),
                    ],
                    dim=-1,
                )
                features_zeros = torch.cat(
                    [
                        features,
                        torch.zeros(
                            size=(features.shape[0], 1),
                            device=device,
                        ),
                    ],
                    dim=-1,
                )
                perm = np.random.permutation(features_ones.shape[0])
                features_ones = features_ones[perm, :]
                features_zeros = features_zeros[perm, :]
                features_ones_scores = model(features_ones)
                features_zeros_scores = model(features_zeros)
                loss = loss_function(
                    features_ones_scores, features_zeros_scores, decisions
                )
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
                batches.set_postfix({"loss": np.mean(losses)})
    torch.save(model, save_path)
    return losses
