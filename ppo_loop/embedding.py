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


class OnlineTravelerDataset(Dataset):
    """
    Dataset for online learning from (customer_id, action, accepted) observations.

    Does not require ground-truth flexibility labels. Instead, computes which flexibility
    types would be consistent with each observed (action, accepted) pair.
    """
    def __init__(self, df_online, flexibility_personalities, action_space_map):
        """
        Args:
            df_online: DataFrame with columns ['traveler_id', 'action', 'accepted']
                      (or 'customer_id' which will be converted to 0-indexed 'traveler_id')
                      Note: customer_id/traveler_id can be 1-indexed (will be converted to 0-indexed)
            flexibility_personalities: List of flexibility type descriptions (length L)
            action_space_map: List of (early_shift, late_shift) tuples for each action
        """
        df_online = df_online.copy()

        # Handle both 'customer_id' and 'traveler_id' column names
        if 'customer_id' in df_online.columns:
            # Convert customer_id (1-indexed) to traveler_id (0-indexed)
            # This matches TravelerDataset behavior
            df_online['traveler_id'] = df_online['customer_id'] - 1
        elif 'traveler_id' in df_online.columns:
            # Check if already 0-indexed or needs conversion
            min_id = df_online['traveler_id'].min()
            if min_id >= 1:
                # Appears to be 1-indexed, convert to 0-indexed
                df_online['traveler_id'] = df_online['traveler_id'] - 1

        self.num_samples = df_online.shape[0]
        self.num_entities = len(df_online["traveler_id"].unique())

        # Entity IDs (0-indexed for embedding layer)
        self.entity_ids = torch.from_numpy(np.array(df_online["traveler_id"])).long()

        # Observed decisions (1 = accepted, 0 = rejected)
        self.decisions = torch.from_numpy(np.array(df_online["accepted"].astype(int)))

        # Compute indicator matrix: ind_matrix[i, l] = 1 if flexibility type l
        # would make the same decision as observed for sample i
        self.ind_matrix = self._compute_indicator_matrix(
            df_online["action"].values,
            df_online["accepted"].values,
            flexibility_personalities,
            action_space_map
        )

        # Decision matrix: what each flexibility type would decide for each action
        # decision_matrix[i, l] = 1 if flexibility type l would accept action i
        self.decision_matrix = self._compute_decision_matrix(
            df_online["action"].values,
            flexibility_personalities,
            action_space_map
        )

    def _would_accept_action(self, action_idx, flex_type_idx, action_space_map):
        """
        Determine if a flexibility type would accept a given action.

        Flexibility types:
            0: flexible for late dropoff, inflexible for early pickup
            1: flexible for early pickup, inflexible for late dropoff
            2: inflexible for any schedule changes
            3: flexible for both early pickup and late dropoff
        """
        early_shift, late_shift = action_space_map[action_idx]
        early_shift = abs(early_shift)  # Convert to positive value

        if flex_type_idx == 0:  # Flexible late dropoff, inflexible early pickup
            return early_shift == 0
        elif flex_type_idx == 1:  # Flexible early pickup, inflexible late dropoff
            return late_shift == 0
        elif flex_type_idx == 2:  # Inflexible for any changes
            return early_shift == 0 and late_shift == 0
        elif flex_type_idx == 3:  # Flexible for both
            return True
        else:
            raise ValueError(f"Unknown flexibility type index: {flex_type_idx}")

    def _compute_indicator_matrix(self, actions, accepted, flexibility_personalities, action_space_map):
        """
        Compute indicator matrix where ind_matrix[i, l] = 1 if flexibility type l
        would make the same decision as observed for sample i.
        """
        n_samples = len(actions)
        n_flex_types = len(flexibility_personalities)
        ind_matrix = torch.zeros((n_samples, n_flex_types), dtype=torch.float32)

        for i in range(n_samples):
            action_idx = actions[i]
            observed_decision = accepted[i]  # True/False or 1/0

            for l in range(n_flex_types):
                # What would this flexibility type decide?
                flex_would_accept = self._would_accept_action(action_idx, l, action_space_map)

                # Indicator is 1 if both made the same decision
                ind_matrix[i, l] = float(flex_would_accept == observed_decision)

        return ind_matrix

    def _compute_decision_matrix(self, actions, flexibility_personalities, action_space_map):
        """
        Compute decision matrix where decision_matrix[i, l] = 1 if flexibility type l
        would accept action i.
        """
        n_samples = len(actions)
        n_flex_types = len(flexibility_personalities)
        decision_matrix = torch.zeros((n_samples, n_flex_types), dtype=torch.float32)

        for i in range(n_samples):
            action_idx = actions[i]

            for l in range(n_flex_types):
                decision_matrix[i, l] = float(
                    self._would_accept_action(action_idx, l, action_space_map)
                )

        return decision_matrix

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.entity_ids[idx], self.decisions[idx], self.ind_matrix[idx,:], self.decision_matrix[idx,:]


def update_embedding_model(embedding_model, online_data, flexibility_personalities, action_space_map,
                           num_epochs=50, batch_size=64, lr=1e-3):
    """
    Update embedding model using online data to learn customer flexibility preferences.

    Args:
        embedding_model: EmbeddingFFN model to update
        online_data: List of dicts with keys: customer_id, action, accepted
        flexibility_personalities: List of flexibility type descriptions
        action_space_map: List of (early_shift, late_shift) tuples for each action
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        lr: Learning rate

    Returns:
        Updated embedding model
    """
    # Data quality checks
    if len(online_data) < batch_size:
        print(f"  [Embedding Update] Insufficient data: {len(online_data)} < {batch_size}. Skipping update.")
        return embedding_model

    df_online = pd.DataFrame(online_data)

    # Check per-customer data availability
    customer_counts = df_online['customer_id'].value_counts()
    min_samples_per_customer = 3
    valid_customers = customer_counts[customer_counts >= min_samples_per_customer]

    if len(valid_customers) < 2:
        print(f"  [Embedding Update] Too few customers with sufficient data: {len(valid_customers)}. Skipping update.")
        return embedding_model

    # Filter to only include customers with enough samples
    df_online = df_online[df_online['customer_id'].isin(valid_customers.index)]

    # Check action diversity
    action_counts = df_online['action'].value_counts()
    if len(action_counts) < 3:
        print(f"  [Embedding Update] Low action diversity: {len(action_counts)} unique actions. Skipping update.")
        return embedding_model

    unique_customers = sorted(df_online['customer_id'].unique())

    print(f"  [Embedding Update] Training on {len(df_online)} samples from {len(unique_customers)} customers")
    print(f"  [Embedding Update] Action distribution: {action_counts.head(5).to_dict()}")

    try:
        # Use OnlineTravelerDataset which calculates consistency matrix correctly
        # OnlineTravelerDataset will handle customer_id (1-indexed) -> traveler_id (0-indexed) conversion
        dataset = OnlineTravelerDataset(df_online, flexibility_personalities, action_space_map)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.Adam(embedding_model.parameters(), lr=lr)

        embedding_model.train()

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            for entity_ids, _, ind_matrix, _ in dataloader:
                optimizer.zero_grad()
                pred_proba = embedding_model(entity_ids)
                beta_matrix = embedding_model.get_embed()
                loss = likelihood_loss(beta_matrix, pred_proba, ind_matrix)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            if epoch == 0 or epoch == num_epochs - 1:
                print(f"    Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(dataloader):.4f}")

        embedding_model.eval()

        # Log predictions for tracked customers
        with torch.no_grad():
            tracked_ids = torch.LongTensor([cid - 1 for cid in unique_customers[:5]])
            pred_proba = embedding_model(tracked_ids)
            predicted_types = torch.argmax(pred_proba, dim=1)
            print(f"  [Embedding Update] Sample predictions for customers {unique_customers[:5]}: {predicted_types.tolist()}")

    except Exception as e:
        print(f"  [Embedding Update] Error: {e}")
        import traceback
        traceback.print_exc()

    return embedding_model


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