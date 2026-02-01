"""
CRITICAL TEST: Improved LSTM + Raw Sensors (PyTorch)

This script runs the EXACT same LSTM architecture on raw sensor data
to determine whether PRISM features add value at the LSTM level.

Architecture (identical to PRISM run):
- Bidirectional LSTM (64 units)
- Attention mechanism
- RUL capped at 125 (benchmark standard)
- Sequence length: 50
- 100 epochs
- GroupKFold by engine (3 folds)

Expected results interpretation:
- Raw ≈ 14-15: PRISM adds nothing
- Raw ≈ 15-16: PRISM helps marginally
- Raw ≈ 20+: PRISM helps significantly
"""

import numpy as np
import polars as pl
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")


class Attention(nn.Module):
    """Simple attention mechanism for LSTM outputs."""

    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, lstm_output):
        # lstm_output: (batch, seq_len, hidden_size)
        attention_weights = self.attention(lstm_output)  # (batch, seq_len, 1)
        attention_weights = torch.softmax(attention_weights, dim=1)
        # Weighted sum
        context = torch.sum(lstm_output * attention_weights, dim=1)  # (batch, hidden_size)
        return context


class ImprovedLSTM(nn.Module):
    """Bidirectional LSTM with attention for RUL prediction."""

    def __init__(self, n_features, hidden_size=64, dropout=0.2):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0
        )

        # Bidirectional doubles hidden size
        self.attention = Attention(hidden_size * 2)

        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        # x: (batch, seq_len, n_features)
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden*2)
        context = self.attention(lstm_out)  # (batch, hidden*2)
        output = self.fc(context)  # (batch, 1)
        return output.squeeze(-1)


def create_sequences(X, y, seq_len):
    """Create sequences for LSTM."""
    sequences = []
    targets = []

    for i in range(len(X) - seq_len + 1):
        sequences.append(X[i:i+seq_len])
        targets.append(y[i+seq_len-1])

    return np.array(sequences), np.array(targets)


def load_raw_data():
    """Load raw sensor data and create RUL labels."""
    print("Loading raw sensor data...")

    # Load observations
    obs = pl.read_parquet('/Users/jasonrudder/prism/data/cmapss/observations.parquet')

    # Pivot to wide format
    wide = obs.pivot(
        index=['unit_id', 'I'],
        on='signal_id',
        values='value'
    ).sort(['unit_id', 'I'])

    # Get sensor columns (sorted for consistency)
    sensor_cols = sorted([c for c in wide.columns if c.startswith('sensor_')])
    print(f"Sensors: {len(sensor_cols)}")

    # Create RUL labels
    max_cycles = wide.group_by('unit_id').agg(pl.col('I').max().alias('max_cycle'))
    wide = wide.join(max_cycles, on='unit_id')
    wide = wide.with_columns((pl.col('max_cycle') - pl.col('I')).alias('RUL'))

    # Cap RUL at 125 (benchmark standard)
    wide = wide.with_columns(pl.col('RUL').clip(upper_bound=125).alias('RUL_capped'))

    print(f"Total rows: {len(wide)}")
    print(f"Units: {wide['unit_id'].n_unique()}")

    return wide, sensor_cols


def train_epoch(model, loader, optimizer, criterion):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    n_batches = 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        predictions = model(X_batch)
        loss = criterion(predictions, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


def evaluate(model, loader, criterion):
    """Evaluate model."""
    model.eval()
    total_loss = 0
    n_batches = 0

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)

            total_loss += loss.item()
            n_batches += 1

    return total_loss / n_batches


def run_lstm_cv(wide_df, feature_cols, n_folds=3, seq_len=50, epochs=100):
    """Run GroupKFold cross-validation with Improved LSTM."""

    # Prepare data by unit
    units = wide_df['unit_id'].unique().sort().to_list()

    # Create unit-level data
    X_by_unit = {}
    y_by_unit = {}

    for unit in units:
        unit_data = wide_df.filter(pl.col('unit_id') == unit).sort('I')
        X = unit_data.select(feature_cols).to_numpy().astype(np.float32)
        y = unit_data['RUL_capped'].to_numpy().astype(np.float32)

        if len(X) >= seq_len:
            X_by_unit[unit] = X
            y_by_unit[unit] = y

    valid_units = list(X_by_unit.keys())
    print(f"Valid units (len >= {seq_len}): {len(valid_units)}")

    n_features = len(feature_cols)

    # GroupKFold by unit
    gkf = GroupKFold(n_splits=n_folds)
    unit_array = np.array(valid_units)

    fold_results = []

    for fold, (train_idx, test_idx) in enumerate(gkf.split(unit_array, groups=unit_array)):
        print(f"\n=== Fold {fold+1}/{n_folds} ===")

        train_units = unit_array[train_idx]
        test_units = unit_array[test_idx]

        print(f"Train units: {len(train_units)}, Test units: {len(test_units)}")

        # Collect training data
        X_train_list = []
        y_train_list = []
        for unit in train_units:
            X_seq, y_seq = create_sequences(X_by_unit[unit], y_by_unit[unit], seq_len)
            X_train_list.append(X_seq)
            y_train_list.append(y_seq)

        X_train = np.vstack(X_train_list)
        y_train = np.concatenate(y_train_list)

        # Scale features (fit on train, transform test)
        n_samples, n_steps, n_feat = X_train.shape
        X_train_2d = X_train.reshape(-1, n_feat)
        scaler = StandardScaler()
        X_train_2d = scaler.fit_transform(X_train_2d)
        X_train = X_train_2d.reshape(n_samples, n_steps, n_feat).astype(np.float32)

        print(f"Train sequences: {len(X_train)}")

        # Create validation split
        val_size = int(0.1 * len(X_train))
        X_val = X_train[-val_size:]
        y_val = y_train[-val_size:]
        X_train = X_train[:-val_size]
        y_train = y_train[:-val_size]

        # Create data loaders
        train_dataset = TensorDataset(
            torch.from_numpy(X_train),
            torch.from_numpy(y_train)
        )
        val_dataset = TensorDataset(
            torch.from_numpy(X_val),
            torch.from_numpy(y_val)
        )

        train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=256)

        # Create model
        model = ImprovedLSTM(n_features, hidden_size=64, dropout=0.2).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
        )
        criterion = nn.MSELoss()

        # Train with early stopping
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0
        best_model_state = None

        for epoch in range(epochs):
            train_loss = train_epoch(model, train_loader, optimizer, criterion)
            val_loss = evaluate(model, val_loader, criterion)

            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

            if (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch+1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

        # Restore best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        print(f"Trained for {epoch+1} epochs, best val_loss: {best_val_loss:.4f}")

        # Evaluate on test units
        model.eval()
        test_predictions = []
        test_actuals = []

        with torch.no_grad():
            for unit in test_units:
                X_seq, y_seq = create_sequences(X_by_unit[unit], y_by_unit[unit], seq_len)

                # Scale with same scaler
                X_seq_2d = X_seq.reshape(-1, n_feat)
                X_seq_2d = scaler.transform(X_seq_2d)
                X_seq = X_seq_2d.reshape(-1, n_steps, n_feat).astype(np.float32)

                X_tensor = torch.from_numpy(X_seq).to(device)
                preds = model(X_tensor).cpu().numpy()

                test_predictions.extend(preds)
                test_actuals.extend(y_seq)

        test_predictions = np.array(test_predictions)
        test_actuals = np.array(test_actuals)

        # Compute RMSE
        rmse = np.sqrt(np.mean((test_predictions - test_actuals) ** 2))
        fold_results.append(rmse)

        print(f"Fold {fold+1} RMSE: {rmse:.2f}")

    return fold_results


def main():
    print("=" * 60)
    print("CRITICAL TEST: Improved LSTM + Raw Sensors")
    print("=" * 60)
    print("\nThis determines whether PRISM adds value at the LSTM level.")
    print("Comparing to: Improved LSTM + PRISM = 14.80 ± 0.07\n")

    # Load data
    wide_df, sensor_cols = load_raw_data()

    # Run cross-validation
    print("\nRunning 3-fold GroupKFold CV...")
    fold_results = run_lstm_cv(
        wide_df,
        sensor_cols,
        n_folds=3,
        seq_len=50,
        epochs=100
    )

    # Results
    mean_rmse = np.mean(fold_results)
    std_rmse = np.std(fold_results)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"\nImproved LSTM + Raw: {mean_rmse:.2f} ± {std_rmse:.2f}")
    print(f"Improved LSTM + PRISM: 14.80 ± 0.07")
    print(f"\nDifference: {mean_rmse - 14.80:.2f} cycles")

    print("\n" + "-" * 40)
    print("INTERPRETATION")
    print("-" * 40)

    if mean_rmse < 15.5:
        print("→ Raw ≈ PRISM: PRISM adds NOTHING at LSTM level")
        print("  The improved model extracts all information from raw data")
    elif mean_rmse < 17:
        print("→ Raw slightly worse: PRISM helps MARGINALLY")
        print(f"  PRISM improves by {mean_rmse - 14.80:.1f} cycles")
    elif mean_rmse < 20:
        print("→ Raw noticeably worse: PRISM helps MODERATELY")
        print(f"  PRISM improves by {mean_rmse - 14.80:.1f} cycles")
    else:
        print("→ Raw much worse: PRISM helps SIGNIFICANTLY")
        print(f"  PRISM improves by {mean_rmse - 14.80:.1f} cycles")

    print("\n" + "=" * 60)

    return mean_rmse, std_rmse


if __name__ == '__main__':
    main()
