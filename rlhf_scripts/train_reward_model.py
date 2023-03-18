import logging
import os
import sys
import pandas as pd
import torch
sys.path.insert(1, os.path.join(sys.path[0], ".."))

from source.losses import preference_loss_function
from source.mlp import MLP
from source.training import train_reward_model

# trajectory_path = "./data/synthetic_decisions.csv"
# trajectory_path = "./data/trajectory_decisions.csv"

logging.basicConfig(
    format="%(asctime)s %(levelname)s:%(message)s",
    level=logging.INFO,
    datefmt="%I:%M:%S",
)

drop_features = ["timestep", "sample_id"]

def train_model(trajectory_path, 
                target_col="agent_action", 
                model_hidden_layers=[100, 100], 
                save_path="./models/reward_model.pt",
                loss_path="./models/losses.csv"
                ):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    decisions_df = pd.read_csv(trajectory_path)
    drop_features_filtered = [i for i in drop_features if i in decisions_df.columns]
    decisions_df = decisions_df.drop(drop_features_filtered, axis=1)

    print(decisions_df)
    features = decisions_df.drop([target_col], axis=1).to_numpy()
    decisions = decisions_df[target_col].to_numpy()
    input_dim = features.shape[1]

    reward_model = MLP(name="reward_model", layer_dims=[input_dim+1] + model_hidden_layers + [1], out_act=None)
    losses = train_reward_model(
        reward_model,
        features,
        decisions,
        loss_function=preference_loss_function,
        learning_rate=0.0001,
        num_epochs=30,
        batch_size=256,
        save_path=save_path,
    )

    # export the losses
    losses_df = pd.DataFrame(losses, columns=['loss'])
    losses_df.to_csv(loss_path)
