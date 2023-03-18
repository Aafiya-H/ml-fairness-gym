import logging
import os
import sys
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

sys.path.insert(1, os.path.join(sys.path[0], ".."))

from source.evaluate_models import evaluate_model
from source.mlp import MLP
from source.training import train_classification_model
from source.utils import load_csv_data_splits

logging.basicConfig(
    format="%(asctime)s %(levelname)s:%(message)s",
    level=logging.INFO,
    datefmt="%I:%M:%S",
)

decision_data_path = "./data/trajectory_decisions.csv"
synthetic_decisions_path = "./data/synthetic_decisions.csv"

X_train, X_test, y_train, y_test = load_csv_data_splits(
    data_path=decision_data_path, target_column=2, dataset_size=50
)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
input_dim = X_train.shape[1]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
decision_maker = MLP("decision_maker", layer_dims=[input_dim, 100, 1], out_act="sigmoid")

losses = train_classification_model(
    decision_maker,
    X_train,
    y_train,
    loss_function=torch.nn.BCELoss(),
    learning_rate=0.001,
    num_epochs=3,
    batch_size=128,
    save_dir="./models/",
)
predictions, score = evaluate_model(decision_maker, X_test, y_test, device)
logging.info(f"Classification score of trained decision maker: {score}")

feature_decision_dataframe = pd.DataFrame(X_test)
feature_decision_dataframe[input_dim-1] = (feature_decision_dataframe[input_dim-1] > 0).astype(int)
feature_decision_dataframe["decision"] = (predictions.detach().cpu() > 0.5).int()
feature_decision_dataframe.to_csv(synthetic_decisions_path, index=False)
logging.info(f"Decision dataframe stored in ./data/synthetic_decisions.csv")
