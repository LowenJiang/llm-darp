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

    Uses CSV ground truth to determine which flexibility types would make the same decision.
    """
    def __init__(self, df_online, flexibility_personalities, action_space_map,
                 csv_path="/Users/jiangwolin/Desktop/Research/llm-rl/rl4co git/ppo_loop/traveler_decisions_augmented.csv"):
        """
        Args:
            df_online: DataFrame with columns ['traveler_id', 'action', 'accepted',
                      'trip_purpose', 'departure_location', 'arrival_location',
                      'departure_time_window', 'arrival_time_window']
            flexibility_personalities: List of flexibility type descriptions (length L)
            action_space_map: List of (early_shift, late_shift) tuples for each action
            csv_path: Path to CSV with ground truth acceptance decisions
        """
        df_online = df_online.copy()

        # Load CSV ground truth
        try:
            self.csv_df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"  [WARNING] Could not load CSV from {csv_path}: {e}")
            print(f"  [WARNING] Falling back to hard-coded rules")
            self.csv_df = None

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

        # Store df_online for CSV lookup
        self.df_online = df_online

        # Entity IDs (0-indexed for embedding layer)
        self.entity_ids = torch.from_numpy(np.array(df_online["traveler_id"])).long()

        # Observed decisions (1 = accepted, 0 = rejected)
        self.decisions = torch.from_numpy(np.array(df_online["accepted"].astype(int)))

        # Compute indicator matrix: ind_matrix[i, l] = 1 if flexibility type l
        # would make the same decision as observed for sample i
        self.ind_matrix = self._compute_indicator_matrix(
            df_online,
            df_online["action"].values,
            df_online["accepted"].values,
            flexibility_personalities,
            action_space_map
        )

        # Decision matrix: what each flexibility type would decide for each action
        # decision_matrix[i, l] = 1 if flexibility type l would accept action i
        self.decision_matrix = self._compute_decision_matrix(
            df_online,
            df_online["action"].values,
            flexibility_personalities,
            action_space_map
        )

    def _would_accept_action_csv(self, sample_row, flex_type_idx, action_idx,
                                   action_space_map, flexibility_personalities):
        """
        Look up from CSV what a flexibility type would decide for this action.

        Args:
            sample_row: Row from df_online with trip context
            flex_type_idx: Index of flexibility type (0-3)
            action_idx: Action index
            action_space_map: List of (pickup_shift, dropoff_shift) tuples
            flexibility_personalities: List of flexibility type strings

        Returns:
            True if this flexibility type would accept, False otherwise
        """
        if self.csv_df is None:
            # Fallback to hard-coded rules if CSV not available
            return self._would_accept_action_hardcoded(action_idx, flex_type_idx, action_space_map)

        # Get trip context from sample
        traveler_id = sample_row.get('traveler_id', sample_row.get('customer_id'))
        trip_purpose = sample_row.get('trip_purpose')
        departure_location = sample_row.get('departure_location')
        arrival_location = sample_row.get('arrival_location')
        departure_tw = sample_row.get('departure_time_window')
        arrival_tw = sample_row.get('arrival_time_window')

        # Get action details
        pickup_shift, dropoff_shift = action_space_map[action_idx]
        pickup_shift_abs = abs(pickup_shift)
        dropoff_shift_abs = abs(dropoff_shift)

        # Check if we have required context
        if any(v is None for v in [trip_purpose, departure_location, arrival_location,
                                     departure_tw, arrival_tw]):
            # Missing context, fall back to hard rules
            return self._would_accept_action_hardcoded(action_idx, flex_type_idx, action_space_map)

        # Look up in CSV
        mask = (
            (self.csv_df['traveler_id'] == traveler_id) &
            (self.csv_df['trip_purpose'] == trip_purpose) &
            (self.csv_df['departure_location'] == departure_location) &
            (self.csv_df['arrival_location'] == arrival_location) &
            (self.csv_df['departure_time_window'] == departure_tw) &
            (self.csv_df['arrival_time_window'] == arrival_tw) &
            (self.csv_df['pickup_shift_min'] == pickup_shift_abs) &
            (self.csv_df['dropoff_shift_min'] == dropoff_shift_abs)
        )

        matching_rows = self.csv_df[mask]

        if len(matching_rows) == 0:
            # No match in CSV, fall back to hard rules
            return self._would_accept_action_hardcoded(action_idx, flex_type_idx, action_space_map)

        # Get the flexibility type column name
        flex_column = flexibility_personalities[flex_type_idx]

        # Get decision from CSV
        decision = matching_rows.iloc[0][flex_column]
        return decision == "accept"

    def _would_accept_action_hardcoded(self, action_idx, flex_type_idx, action_space_map):
        """
        Fallback hard-coded rules (for when CSV lookup fails).

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

    def _compute_indicator_matrix(self, df_online, actions, accepted, flexibility_personalities, action_space_map):
        """
        Compute indicator matrix where ind_matrix[i, l] = 1 if flexibility type l
        would make the same decision as observed for sample i.

        Uses CSV lookup to determine what each flexibility type would decide.
        """
        n_samples = len(actions)
        n_flex_types = len(flexibility_personalities)
        ind_matrix = torch.zeros((n_samples, n_flex_types), dtype=torch.float32)

        for i in range(n_samples):
            action_idx = actions[i]
            observed_decision = accepted[i]  # True/False or 1/0
            sample_row = df_online.iloc[i]

            for l in range(n_flex_types):
                # What would this flexibility type decide? (CSV lookup)
                flex_would_accept = self._would_accept_action_csv(
                    sample_row, l, action_idx, action_space_map, flexibility_personalities
                )

                # Indicator is 1 if both made the same decision
                ind_matrix[i, l] = float(flex_would_accept == observed_decision)

        return ind_matrix

    def _compute_decision_matrix(self, df_online, actions, flexibility_personalities, action_space_map):
        """
        Compute decision matrix where decision_matrix[i, l] = 1 if flexibility type l
        would accept action i.

        Uses CSV lookup to determine what each flexibility type would decide.
        """
        n_samples = len(actions)
        n_flex_types = len(flexibility_personalities)
        decision_matrix = torch.zeros((n_samples, n_flex_types), dtype=torch.float32)

        for i in range(n_samples):
            action_idx = actions[i]
            sample_row = df_online.iloc[i]

            for l in range(n_flex_types):
                decision_matrix[i, l] = float(
                    self._would_accept_action_csv(
                        sample_row, l, action_idx, action_space_map, flexibility_personalities
                    )
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

        # Diagnostics: Check indicator matrix quality
        sample_ind_matrix = dataset.ind_matrix[:min(100, len(dataset))]
        ind_mean_per_flex = sample_ind_matrix.mean(dim=0)
        print(f"  [Embedding Update] Indicator matrix mean per flexibility type: {ind_mean_per_flex.tolist()}")

        # Check if CSV lookup was used
        if dataset.csv_df is not None:
            print(f"  [Embedding Update] Using CSV ground truth for indicator matrix (not hard rules!)")
        else:
            print(f"  [WARNING] CSV not loaded, using hard-coded rules (may be incorrect!)")

        # Check for degenerate cases
        all_zeros_samples = (sample_ind_matrix.sum(dim=1) == 0).sum().item()
        if all_zeros_samples > 0:
            print(f"  [WARNING] {all_zeros_samples} samples have all-zero indicator (no flexibility agrees)")

        # Check if any flexibility type is never consistent (always 0 in ind_matrix)
        full_ind_matrix = dataset.ind_matrix
        ind_sum_per_flex = full_ind_matrix.sum(dim=0)
        for flex_idx, total in enumerate(ind_sum_per_flex.tolist()):
            if total == 0:
                print(f"  [WARNING] Flexibility type {flex_idx} is NEVER consistent with observed data!")
            elif total < len(dataset) * 0.01:  # Less than 1% consistency
                print(f"  [WARNING] Flexibility type {flex_idx} is rarely consistent ({total}/{len(dataset)} = {100*total/len(dataset):.1f}%)")

        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.Adam(embedding_model.parameters(), lr=lr)

        embedding_model.train()

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            num_nan_batches = 0

            for entity_ids, _, ind_matrix, _ in dataloader:
                optimizer.zero_grad()
                pred_proba = embedding_model(entity_ids)
                beta_matrix = embedding_model.get_embed()
                loss = likelihood_loss(beta_matrix, pred_proba, ind_matrix)

                # Check for NaN loss
                if torch.isnan(loss):
                    num_nan_batches += 1
                    continue  # Skip this batch

                loss.backward()

                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(embedding_model.parameters(), max_norm=1.0)

                optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / max(len(dataloader) - num_nan_batches, 1)
            if epoch == 0 or epoch == num_epochs - 1:
                if num_nan_batches > 0:
                    print(f"    Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f} ({num_nan_batches} NaN batches skipped)")
                else:
                    print(f"    Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

        embedding_model.eval()

        # Log predictions for tracked customers
        with torch.no_grad():
            tracked_ids = torch.LongTensor([cid - 1 for cid in unique_customers[:5]])
            pred_proba = embedding_model(tracked_ids)
            #predicted_types = torch.argmax(pred_proba, dim=1)

            dist = torch.distributions.Categorical(probs=pred_proba)
            predicted_flexibilities = dist.sample() 
            print(f"  [Embedding Update] Sample predictions for customers {unique_customers[:5]}: {predicted_types.tolist()}")

            # Also show probability distributions to diagnose collapse
            print(f"  [Embedding Update] Sample probability distributions:")
            for i, cid in enumerate(unique_customers[:3]):
                probs = pred_proba[i].tolist()
                print(f"    Customer {cid}: {[f'{p:.3f}' for p in probs]}")

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
              entropy_coef: float = 0.05,  # NEW: controls exploration
              eps: float = 1e-9,
              unbiased_var: bool = False) -> torch.Tensor:
    """
    Compute loss corresponding to likelihood (we minimize this).
    
    Now includes Entropy Regularization to prevent mode collapse.
    """
    
    # 1. CLAMPING: Prevent probabilities from being exactly 0 or 1.
    # This prevents log(0) and ensures the denominator in w_tilde is never 0 
    # (unless ind_matrix is all zeros, which implies bad data).
    P_z_given_d = P_z_given_d.clamp(min=1e-4, max=1.0 - 1e-4)
    
    # Re-normalize to ensure sum is 1.0 after clamping
    P_z_given_d = P_z_given_d / P_z_given_d.sum(dim=1, keepdim=True)

    # 2. Compute w_tilde
    # Numerator: element-wise product (P(z) * Ind(z))
    numerator = P_z_given_d * ind_matrix

    # Denominator: sum across flexibility types
    # Since P is clamped > 0, this can only be 0 if ind_matrix is all 0s.
    denominator = torch.sum(numerator, dim=1, keepdim=True)
    denominator_safe = denominator.clamp(min=eps)

    # Compute normalized weights P(z | d, accepted)
    w_normalized = numerator / denominator_safe

    # Compute adjustment term (1 - alpha_e * prod(ind_matrix))
    ind_prod = torch.prod(ind_matrix, dim=1, keepdim=True)
    adjustment = (1 - alpha_e * ind_prod).clamp(min=0.0)

    # Final sample weights
    w_tilde = w_normalized * adjustment

    # 3. Weighted Log-Likelihood
    # sum_i sum_l w_il * log P(Z_l | d_i)
    logP = torch.log(P_z_given_d)
    weighted_loglik = torch.sum(w_tilde * logP)

    # 4. Variance Regularization (Beta matrix)
    var_dims = beta_matrix.var(dim=0, unbiased=unbiased_var)
    mean_var = var_dims.mean()

    if mean_var.item() == 0.0:
        reg_term = torch.tensor(0.0, device=beta_matrix.device)
    else:
        ratio = var_dims / (mean_var + 1e-12)
        reg_term = torch.sum((ratio - 1.0) ** 2)

    # 5. Entropy Regularization (NEW)
    # Penalize the model for being 100% sure (to keep exploration alive).
    # H = - sum p * log(p)
    entropy = -torch.sum(P_z_given_d * logP, dim=1).mean()

    # 6. Total Loss
    # We want to:
    #  - Maximize Likelihood (minimize -Likelihood)
    #  - Minimize Variance Heterogeneity (minimize +Reg)
    #  - Maximize Entropy (minimize -Entropy)
    loss = -weighted_loglik + (alpha * reg_term) - (entropy_coef * entropy)

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