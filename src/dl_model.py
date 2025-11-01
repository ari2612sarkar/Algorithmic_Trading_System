# src/dl_model.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import logging
from src.data_loader import fetch_ohlcv
from src.features import create_features

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)


# ==== Deep Learning Models ====

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return self.sigmoid(out)


class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])
        return self.sigmoid(out)


# ==== Training and Evaluation ====

def train_model(model, X_train, y_train, X_val, y_val, epochs=50, lr=0.001):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val)
                val_loss = criterion(val_outputs, y_val)
            logger.info(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {loss.item():.4f} | Val Loss: {val_loss.item():.4f}")

    return model


def evaluate_model(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        preds = model(X_test)
        preds = (preds > 0.5).float()
    y_true = y_test.cpu().numpy().flatten()
    y_pred = preds.cpu().numpy().flatten()
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0)
    }


# ==== Main Execution ====

if __name__ == "__main__":
    df = fetch_ohlcv("RELIANCE", "01-01-2024", "30-10-2025")
    df = create_features(df)

    # Data prep
    X = df.drop("Target", axis=1).values
    y = df["Target"].values.reshape(-1, 1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Reshape for RNN input: (samples, timesteps=5, features)
    seq_len = 5
    X_seq = np.array([X_scaled[i - seq_len:i] for i in range(seq_len, len(X_scaled))])
    y_seq = y[seq_len:]

    split = int(len(X_seq) * 0.8)
    X_train, X_test = X_seq[:split], X_seq[split:]
    y_train, y_test = y_seq[:split], y_seq[split:]

    # Convert to tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    input_size = X_train.shape[2]

    # Train LSTM
    lstm_model = LSTMModel(input_size)
    logger.info("Training LSTM...")
    lstm_model = train_model(lstm_model, X_train, y_train, X_test, y_test)
    lstm_results = evaluate_model(lstm_model, X_test, y_test)
    logger.info(f"LSTM Results: {lstm_results}")

    # Train GRU
    gru_model = GRUModel(input_size)
    logger.info("Training GRU...")
    gru_model = train_model(gru_model, X_train, y_train, X_test, y_test)
    gru_results = evaluate_model(gru_model, X_test, y_test)
    logger.info(f"GRU Results: {gru_results}")

    # === Summary ===
    results_df = pd.DataFrame([{"Model": "LSTM", **lstm_results}, {"Model": "GRU", **gru_results}])
    logger.info("\n🏆 Deep Learning Model Comparison:\n%s", results_df)

    print("\n📈 Deep Learning Summary:")
    print(results_df)

    best_model = results_df.sort_values(by="F1", ascending=False).iloc[0]
    print(f"\n💡 Business Insight:")
    print(f"{best_model['Model']} achieved {best_model['Accuracy']*100:.2f}% accuracy and "
          f"{best_model['F1']*100:.2f}% F1-score — useful for short-term price direction modeling "
          f"in FinTech analytics or signal generation dashboards.")
