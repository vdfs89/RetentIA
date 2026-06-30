import logging
import os
import pickle

import mlflow
import numpy as np
import torch
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from src.cost.threshold import find_optimal_threshold
from src.data.ingest import ingest_data
from src.features.preprocessor import ChurnPreprocessor
from src.models.mlp import ChurnMLP
from src.validation.schemas import get_pandera_schema

logger = logging.getLogger(__name__)


def compute_metrics(y_true: np.ndarray, y_preds: np.ndarray, y_probs: np.ndarray | None = None):
    metrics = {
        "accuracy": accuracy_score(y_true, y_preds),
        "precision": precision_score(y_true, y_preds, zero_division=0),
        "recall": recall_score(y_true, y_preds, zero_division=0),
        "f1": f1_score(y_true, y_preds, zero_division=0),
    }
    if y_probs is not None:
        metrics["roc_auc"] = roc_auc_score(y_true, y_probs)
        metrics["pr_auc"] = average_precision_score(y_true, y_probs)
    return metrics


def train_pipeline():
    # Reprodutibilidade: define a seed do numpy e torch para o treino do MLP ser determinístico.
    np.random.seed(42)
    torch.manual_seed(42)

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("RetentIA_Churn_Prediction")

    with mlflow.start_run():
        df = ingest_data()
        schema = get_pandera_schema(is_training=True)
        schema.validate(df)

        X = df.drop(columns=["Churn", "customerID"])
        y = (df["Churn"] == "Yes").astype(int).values

        # Split: Train 60%, Val 20%, Test 20%
        X_temp, X_test_df, y_temp, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        X_train_df, X_val_df, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
        )

        preprocessor = ChurnPreprocessor()
        preprocessor.fit(X_train_df)

        X_train = preprocessor.transform(X_train_df)
        X_val = preprocessor.transform(X_val_df)
        X_test = preprocessor.transform(X_test_df)

        os.makedirs("models", exist_ok=True)
        preprocessor.save("models/preprocessor.pkl")
        mlflow.log_artifact("models/preprocessor.pkl", artifact_path="preprocessor")

        # 1. Baseline: Dummy Classifier
        dummy = DummyClassifier(strategy="most_frequent")
        dummy.fit(X_train, y_train)
        dummy_preds = dummy.predict(X_test)
        dummy_probs = dummy.predict_proba(X_test)[:, 1]
        dummy_metrics = compute_metrics(y_test, dummy_preds, dummy_probs)
        for k, v in dummy_metrics.items():
            mlflow.log_metric(f"dummy_{k}", v)

        # 2. Baseline: Logistic Regression
        lr = LogisticRegression(class_weight="balanced", random_state=42, max_iter=1000)
        lr.fit(X_train, y_train)
        lr_probs = lr.predict_proba(X_test)[:, 1]
        lr_preds = (lr_probs >= 0.5).astype(int)
        lr_metrics = compute_metrics(y_test, lr_preds, lr_probs)
        for k, v in lr_metrics.items():
            mlflow.log_metric(f"logreg_{k}", v)

        # 3. Baseline: XGBoost
        scale_pos_weight = (len(y_train) - int(np.sum(y_train))) / max(int(np.sum(y_train)), 1)
        xgb = XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            eval_metric="logloss",
            verbosity=0,
        )
        xgb.fit(X_train, y_train)
        xgb_probs = xgb.predict_proba(X_test)[:, 1]
        xgb_preds = (xgb_probs >= 0.5).astype(int)
        xgb_metrics = compute_metrics(y_test, xgb_preds, xgb_probs)
        for k, v in xgb_metrics.items():
            mlflow.log_metric(f"xgb_{k}", v)

        with open("models/xgboost.pkl", "wb") as f:
            pickle.dump(xgb, f)
        mlflow.log_artifact("models/xgboost.pkl", artifact_path="xgboost")

        # 4. PyTorch MLP
        input_dim = X_train.shape[1]
        model = ChurnMLP(input_dim)

        # Imbalance weight
        num_pos = int(np.sum(y_train))
        num_neg = len(y_train) - num_pos
        pos_weight = torch.tensor([num_neg / max(num_pos, 1)], dtype=torch.float32)
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

        train_dataset = TensorDataset(
            torch.FloatTensor(X_train), torch.FloatTensor(y_train).unsqueeze(1)
        )
        loader_generator = torch.Generator()
        loader_generator.manual_seed(42)
        train_loader = DataLoader(
            train_dataset,
            batch_size=64,
            shuffle=True,
            drop_last=True,
            generator=loader_generator,
        )

        X_val_t = torch.FloatTensor(X_val)
        y_val_t = torch.FloatTensor(y_val).unsqueeze(1)

        best_val_loss = float("inf")
        patience = 10
        patience_counter = 0
        best_state = None

        for epoch in range(100):
            model.train()
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                loss = criterion(model(batch_x), batch_y)
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                val_loss = criterion(model(X_val_t), y_val_t).item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info("Parada antecipada no epoch %d", epoch)
                    break

        # Carrega os melhores pesos
        if best_state is not None:
            model.load_state_dict(best_state)
        torch.save(model.state_dict(), "models/mlp_weights.pt")

        model.eval()

        # Tune threshold on Validation Set
        with torch.no_grad():
            val_logits = model(X_val_t)
            val_probs = torch.sigmoid(val_logits).numpy().squeeze()

        optimal_t = find_optimal_threshold(y_val, val_probs)

        # Evaluate MLP on Test Set
        with torch.no_grad():
            test_logits = model(torch.FloatTensor(X_test))
            test_probs = torch.sigmoid(test_logits).numpy().squeeze()

        mlp_preds = (test_probs >= optimal_t).astype(int)
        mlp_metrics = compute_metrics(y_test, mlp_preds, test_probs)

        # Log MLP parameters and metrics
        mlflow.log_param("optimal_threshold", optimal_t)
        mlflow.log_param("pos_weight", pos_weight.item())
        mlflow.log_param("patience", patience)
        mlflow.log_param("batch_size", 64)
        mlflow.log_param("lr", 0.005)
        for k, v in mlp_metrics.items():
            mlflow.log_metric(f"mlp_{k}", v)

        mlflow.log_artifact("models/mlp_weights.pt", artifact_path="model_weights")

        with open("models/threshold.pkl", "wb") as f:
            pickle.dump(optimal_t, f)
        mlflow.log_artifact("models/threshold.pkl", artifact_path="threshold")

        logger.info("=== Pipeline executado com sucesso ===")
        logger.info("Threshold ótimo (ajustado na validação): %.2f", optimal_t)
        logger.info("Dummy    : %s", dummy_metrics)
        logger.info("LogReg   : %s", lr_metrics)
        logger.info("XGBoost  : %s", xgb_metrics)
        logger.info("MLP      : %s", mlp_metrics)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    train_pipeline()
