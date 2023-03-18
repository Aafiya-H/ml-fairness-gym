import torch


def preference_loss_function(features_ones_scores, features_zeros_scores, decisions):
    multiplicative_factor = (2 * decisions) - 1
    score_diff = multiplicative_factor * (features_ones_scores - features_zeros_scores)
    return -torch.mean(torch.log(torch.sigmoid(score_diff)))
