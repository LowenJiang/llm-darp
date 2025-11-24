import re
import os
from ast import literal_eval
import math
import json
import itertools
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from openai import OpenAI
import matplotlib.pyplot as plt
from tqdm import tqdm
from pprint import pprint

# Default flexibility personalities (can be overridden by data)
# Order matters for mask computation!
flexibility_personalities = [
    "flexible for late dropoff, but inflexible for early pickup",
    "flexible for early pickup, but inflexible for late dropoff",
    "inflexible for any schedule changes",
    "flexible for both early pickup and late dropoff"
]
n_flexibilities = len(flexibility_personalities)

# Try to load data if available
try:
    df_decisions = pd.read_csv("traveler_trip_types.csv")
    n_travelers = len(df_decisions["traveler_id"].unique())
    flexibility_personalities = list(df_decisions["flexibility"].unique())
    n_flexibilities = len(flexibility_personalities)
except FileNotFoundError:
    df_decisions = None
    n_travelers = 30  # Default

class OnlineTravelerDataset(torch.utils.data.Dataset):
    """
    Custom dataset for online updates.
    It calculates the 'ind_matrix' (consistency) on the fly without
    needing to hack column names or overwrite data.
    """
    def __init__(self, df_online, flexibility_personalities, action_space_map=None):
        self.num_samples = len(df_online)
        # Convert mapped IDs to tensor
        self.entity_ids = torch.LongTensor(df_online['traveler_id'].values)

        # Default action space map if not provided
        if action_space_map is None:
            action_space_map = [
                (-30, 0), (-30, 10), (-30, 20), (-30, 30),
                (-20, 0), (-20, 10), (-20, 20), (-20, 30),
                (-10, 0), (-10, 10), (-10, 20), (-10, 30),
                (0, 0), (0, 10), (0, 20), (0, 30),
            ]

        # Pre-compute the consistency matrix (ind_matrix)
        # Shape: (N_samples, N_personalities)
        decision_matrix = []

        for idx, row in df_online.iterrows():
            row_consistencies = []

            # What actually happened?
            actual = "accept" if row['accepted'] else "reject"

            # What was the move?
            action = row['action']
            pickup_shift, dropoff_shift = action_space_map[action]
            early_shift = abs(pickup_shift)
            late_shift = dropoff_shift

            # Check consistency for ALL 4 personalities
            for flex_idx, _ in enumerate(flexibility_personalities):
                # 1. Determine theoretical behavior
                if flex_idx == 0:   # Late Flex
                    would_accept = (early_shift == 0)
                elif flex_idx == 1: # Early Flex
                    would_accept = (late_shift == 0)
                elif flex_idx == 2: # Inflexible
                    would_accept = (early_shift == 0 and late_shift == 0)
                else:               # Both Flex
                    would_accept = True

                theoretical = "accept" if would_accept else "reject"

                # 2. Compare with reality
                # If theoretical matches actual, we put 1.0 (consistent)
                # If not, we put 0.0 (inconsistent)
                is_consistent = 1.0 if theoretical == actual else 0.0
                row_consistencies.append(is_consistent)

            decision_matrix.append(row_consistencies)

        self.ind_matrix = torch.FloatTensor(decision_matrix)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # We only need entity_ids and ind_matrix for the likelihood_loss
        # We return dummies for the other values expected by the training loop if needed,
        # or just unpack correctly in the loop.
        return self.entity_ids[idx], torch.zeros(1), self.ind_matrix[idx], torch.zeros(1)


class TravelerDataset(Dataset):
    def __init__(self, df_decisions = None):
        self.num_samples = df_decisions.shape[0]
        self.num_entities = len(df_decisions["traveler_id"].unique())
        self.entity_ids = torch.from_numpy(np.array(df_decisions["traveler_id"] - 1))
        df_decisions["decision"] = df_decisions.apply(lambda row: row[row["flexibility"]], axis = 1)
        self.decisions = torch.from_numpy(np.array(df_decisions["decision"].apply(lambda x: 1 if x == "accept" else 0)))
        mapping = {cat: i for i, cat in enumerate(flexibility_personalities)}
        df_decisions["flexibility_encoded"] = df_decisions["flexibility"].map(mapping)
        self.flexibility_labels = torch.tensor(df_decisions["flexibility_encoded"].values, dtype=torch.long)
        ## TODO: Implement it
        self.decision_matrix = torch.zeros((df_decisions.shape[0], n_flexibilities))
        flexibility_decisions = df_decisions[flexibility_personalities].copy()
        for i in range(len(flexibility_personalities)):
            flexibility = flexibility_personalities[i]
            flexibility_decisions[flexibility] = (df_decisions[flexibility] == df_decisions["decision"]) + 0
            self.decision_matrix[:,i] = torch.from_numpy(np.array(df_decisions[flexibility].apply(lambda x: 1 if x == "accept" else 0)))
        self.ind_matrix = torch.from_numpy(np.array(flexibility_decisions))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.entity_ids[idx], self.decisions[idx], self.ind_matrix[idx,:], self.decision_matrix[idx,:]

class EmbeddingFFN(nn.Module):
    def __init__(self, num_entities, embed_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(num_entities, embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim),
            nn.Softmax(dim=-1)
        )
    
    def get_embed(self):
        return self.embedding.weight

    def forward(self, entity_ids):
        emb = self.embedding(entity_ids)
        return self.ffn(emb)

def likelihood_loss(beta_matrix: torch.Tensor,
              P_z_given_d: torch.Tensor,
              ind_matrix: torch.Tensor,
              alpha_e: float = 0.5,
              alpha: float = 0.4,
              eps: float = 1e-9,
              unbiased_var: bool = False) -> torch.Tensor:
    """
    Compute loss corresponding to likelihood (we minimize this).

    Loss = - sum_{i=1..N} sum_{l=1..L} w_tilde[i,l] * log(P_z_given_d[i,l])
           + alpha * sum_{m=1..M} ( var(beta[:,m]) / mean_var - 1 )^2

    Args:
        beta_matrix: Tensor shape (G, M) -- embedding parameters; var computed across G for each dimension m
        P_z_given_d: Tensor shape (N, L) -- probabilities P(Z_l | d_i); values in (0,1]
        w_tilde: Tensor shape (N, L) -- modified/sample weights \tilde{w}_{i,l}
        alpha: regularization strength (scalar). In the paper they used α_m; here we use same alpha for sum.
        eps: small float to stabilize log
        unbiased_var: whether to use unbiased var (ddof=1). Default False for population var.

    Returns:
        loss: scalar tensor (to minimize)
    """
    ## Compute w_tilde
    w_tilde = (P_z_given_d * ind_matrix) / torch.sum(P_z_given_d * ind_matrix, dim = 1).reshape((-1, 1)) * (1 - alpha_e * torch.prod(ind_matrix, dim = 1)).reshape((-1, 1))

    # Weighted log-likelihood term: sum_i sum_l w_il * log P(Z_l | d_i)
    # we minimize negative of that (maximize objective in eq(21))
    logP = torch.log(P_z_given_d.clamp(min=eps))
    weighted_loglik = torch.sum(w_tilde * logP)  # scalar

    # Regularization: compute per-dimension variances of beta_matrix
    # beta_matrix shape: (G, M) -> variances over dim 0 -> result shape (M,)
    var_dims = beta_matrix.var(dim=0, unbiased=unbiased_var)  # σ^2(β_m) for each m
    mean_var = var_dims.mean()

    # avoid divide-by-zero
    if mean_var.item() == 0.0:
        # If mean_var is zero (all zeros), regularizer should be zero (no variance across dims)
        reg_term = torch.tensor(0.0, device=beta_matrix.device, dtype=beta_matrix.dtype)
    else:
        ratio = var_dims / (mean_var + 1e-12)
        reg_term = torch.sum((ratio - 1.0) ** 2)  # sum_m (σ^2(β_m)/meanvar - 1)^2

    # full objective from Eq (21) is: maximize weighted_loglik + alpha * reg_term
    # we return loss = - (weighted_loglik + alpha * reg_term)
    loss = - (weighted_loglik + alpha * reg_term)

    return loss

def sample_trips_per_traveler(df: pd.DataFrame, N: int, random_state: int = None) -> pd.DataFrame:
    """
    For each traveler_id in the dataframe, randomly sample exactly N records,
    sampling with replacement if a traveler has fewer than N records.

    Parameters
    ----------
    df : pd.DataFrame
        The input dataframe containing a 'traveler_id' column.
    N : int
        Number of samples per traveler.
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        A dataframe containing exactly N samples per traveler.
    """
    return (
        df.groupby("traveler_id", group_keys=False)
          .apply(lambda x: x.sample(n=N, replace=True, random_state=random_state))
          .reset_index(drop=True)
    )

def binary_classification_metrics(y_true, y_pred):
    """
    Compute accuracy, precision, recall, F1, and confusion matrix for binary labels.
    Both y_true and y_pred are 1D torch tensors of 0s and 1s.
    """
    y_true = y_true.int()
    y_pred = y_pred.int()

    tp = torch.sum((y_true == 1) & (y_pred == 1)).item()
    tn = torch.sum((y_true == 0) & (y_pred == 0)).item()
    fp = torch.sum((y_true == 0) & (y_pred == 1)).item()
    fn = torch.sum((y_true == 1) & (y_pred == 0)).item()

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    confusion_matrix = torch.tensor([[tn, fp],
                                     [fn, tp]])

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": confusion_matrix
    }

if __name__ == "__main__":
    # Only run training when executed directly
    if df_decisions is None:
        print("Error: traveler_trip_types.csv not found. Cannot run standalone training.")
        exit(1)

    NUM_EPOCHS = 500
    BATCH_SIZE = 128
    model = EmbeddingFFN(num_entities=n_travelers, embed_dim=64, hidden_dim=128, output_dim=n_flexibilities)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    df_decisions_train = sample_trips_per_traveler(df_decisions, N=20)
    dataset_train = TravelerDataset(df_decisions=df_decisions_train)
    dataloader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True)
    dataset_test = TravelerDataset(df_decisions=df_decisions)
    dataloader_test = DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=True)

    # Training loop
    train_loss_arr = []
    test_loss_arr = []
    test_error_arr = []
    for epoch in tqdm(range(NUM_EPOCHS)):

        model.train()
        total_loss = 0.0
        for entity_ids, _, ind_matrix, _ in dataloader_train:
            optimizer.zero_grad()
            pred_proba = model(entity_ids)
            beta_matrix = model.get_embed()
            loss = likelihood_loss(beta_matrix, pred_proba, ind_matrix, alpha_e=0.5, alpha=0.4, eps=1e-9, unbiased_var=False)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(entity_ids)
        avg_loss_train = total_loss / len(dataset_train)
        train_loss_arr.append(avg_loss_train)

        model.eval()
        total_loss = 0.0
        total_miss = 0

        decision_lst = []
        selected_value_lst = []
        for entity_ids, decisions, ind_matrix, decision_matrix in dataloader_test:
            with torch.no_grad():
                pred_proba = model(entity_ids)
                beta_matrix = model.get_embed()
                loss = likelihood_loss(beta_matrix, pred_proba, ind_matrix, alpha_e=0.5, alpha=0.4, eps=1e-9, unbiased_var=False)
                total_loss += loss.item() * len(entity_ids)
                ## Compute mis-classification rate
                max_idx = torch.argmax(pred_proba, dim=1)  # shape: [N]
                selected_values = decision_matrix.gather(1, max_idx.unsqueeze(1)).squeeze(1)
                total_miss += torch.sum(selected_values != decisions)
            if epoch == NUM_EPOCHS - 1:
                decision_lst += list(decisions.numpy())
                selected_value_lst += list(selected_values.numpy())
        avg_loss_test = total_loss / len(dataset_test)
        avg_miss = total_miss / len(dataset_test)
        test_loss_arr.append(avg_loss_test)
        test_error_arr.append(avg_miss)

    decisions = torch.tensor(decision_lst)
    selected_values = torch.tensor(selected_value_lst)
    metrics = binary_classification_metrics(decisions, selected_values)
    print(metrics)

    plt.plot(train_loss_arr, label="Training Loss")
    plt.plot(test_loss_arr, label="Test Loss")
    plt.xlabel("Training epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("embedding_loss.png")
    plt.clf()
    plt.close()

    plt.plot(test_error_arr)
    plt.xlabel("Training epoch")
    plt.ylabel("Misclassification Rate For Acceptance/Rejection")
    plt.title(f"Misclassification rate {test_error_arr[-1] * 100:.2f}%")
    plt.savefig("embedding_miss.png")
    plt.clf()
    plt.close()