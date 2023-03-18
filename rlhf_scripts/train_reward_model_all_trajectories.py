import logging
import os
import sys
import pandas as pd
import torch
sys.path.insert(1, os.path.join(sys.path[0], ".."))

from source.losses import preference_loss_function
from source.mlp import MLP
from source.training import train_reward_model

"""
Input features:
"acceptance_rate-group_1", "acceptance_rate-group_2",
"defaulter_rate--group_1", "defaulter_rate--group_2", 
"average_credit_score-group_1", "average_credit_score--group_2", 
"group_membership", "credit_score"

Output feauture: "decision"
"""

model_save_path="./models/reward_model.pt"
losses_save_path="./models/losses.csv"

trajectory_folder = "./data/metrics/"
trajectory_paths = [f"{trajectory_folder}{i}" for i in os.listdir(trajectory_folder) if ('.csv' in i) and ('trajectories' in i)]

target_feature = "agent_action"
drop_features = ["timestep", "sample_id"]

logging.basicConfig(
    format="%(asctime)s %(levelname)s:%(message)s",
    level=logging.INFO,
    datefmt="%I:%M:%S",
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

df_list = []
for tpm_path in trajectory_paths:
    df_list.append(pd.read_csv(tpm_path))
decisions_df = pd.concat(df_list, axis=0)

drop_features = [i.lower() for i in drop_features]
drop_cols = [i for i in decisions_df.columns if i.lower() in drop_features]
if len(drop_cols) > 0:
    decisions_df = decisions_df.drop(drop_cols, axis=1)

features = decisions_df.drop(target_feature, axis=1).to_numpy()
decisions = decisions_df[target_feature].to_numpy()
input_dim = features.shape[1]

reward_model = MLP(name=f"reward_model", layer_dims=[input_dim+1, 100, 100, 1], out_act=None)
losses = train_reward_model(
    reward_model,
    features,
    decisions,
    loss_function=preference_loss_function,
    learning_rate=0.0001,
    num_epochs=30,
    batch_size=256,
    save_path=model_save_path
)
# export the losses
losses_df = pd.DataFrame(losses, columns=['loss'])
losses_df.to_csv(losses_save_path)
