import torch
from sklearn.metrics import accuracy_score


def evaluate_model(model, test_features, test_targets, device):
    with torch.inference_mode():
        predictions = model(
            torch.tensor(test_features, dtype=torch.float32, device=device)
        )
    return predictions, accuracy_score(predictions.detach().cpu() > 0.5, test_targets)
