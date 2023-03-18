from typing import Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer, StandardScaler


def load_csv_data_splits(
    data_path: str, target_column: Union[int, str], dataset_size: int
):
    subsample = np.random.permutation(dataset_size)
    data = pd.read_csv(data_path).to_numpy()[subsample, :]
    X = data[:, [i for i in range(data.shape[1]) if i != target_column]]
    discretizer = KBinsDiscretizer(n_bins=2, strategy="quantile", encode="ordinal")
    y = discretizer.fit_transform(data[:, target_column].reshape(-1, 1))
    return train_test_split(X, y, test_size=0.33, random_state=42)
